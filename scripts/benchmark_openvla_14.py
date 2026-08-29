#!/usr/bin/env python3
"""Run fixed OpenVLA synthetic diagnostics for one worker's config subset.

This is a compatibility/latency sweep, not a LIBERO success evaluation.
Each Slurm worker loads one OpenVLA model and evaluates several independent
configuration names sequentially so four GPUs can be used concurrently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/vla-cache/src/openvla"))

from transformers import (  # noqa: E402
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    DynamicCache,
)

from experiments.robot.vla_cache_utils import (  # noqa: E402
    get_layer_mask_schedule,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from prismatic.extern.hf.processing_prismatic import (  # noqa: E402
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from vla_tricks.depth import (  # noqa: E402
    StaticDepthPruner,
    collect_block_influence,
    find_decoder_layers,
    select_non_adjacent,
)
from vla_tricks.foveation import foveate_blur  # noqa: E402
from vla_tricks.perception import InteractionAwareTemporalFusion  # noqa: E402
from vla_tricks.temporal import ConservativeActionReuse  # noqa: E402
from vla_tricks.vla_cache import visual_task_relevance  # noqa: E402


ALL_CONFIGS = (
    "original",
    "fixed_foveation_keep20",
    "fixed_foveation_keep50",
    "action_repeat2",
    "action_repeat4",
    "depth_pruning1",
    "depth_pruning2",
    "depth_pruning4",
    "guarded_reuse_strict",
    "guarded_reuse_moderate",
    "guarded_reuse_aggressive",
    "temporal_fusion_motion_entropy",
    "temporal_fusion_task_aware",
    "temporal_fusion_conservative_adaptive",
)

GUARDED = {
    "guarded_reuse_strict": (0.01, 0.03, 0.995, 1),
    "guarded_reuse_moderate": (0.015, 0.04, 0.99, 1),
    "guarded_reuse_aggressive": (0.02, 0.05, 0.98, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/openvla-7b"))
    parser.add_argument("--configs", required=True, help="comma-separated config names")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/results/openvla_7b")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", default="unknown")
    return parser.parse_args()


def register_openvla() -> None:
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True
    )


def make_scene(shift: int = 0) -> Image.Image:
    image = Image.new("RGB", (224, 224), (174, 148, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 148, 224, 224), fill=(115, 85, 58))
    draw.ellipse((58 + shift, 78, 100 + shift, 120), fill=(30, 30, 30))
    draw.rectangle((137, 88, 176, 128), fill=(215, 52, 42))
    draw.rectangle((151, 65, 162, 91), fill=(210, 210, 210))
    return image


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def action_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "gripper_changed": bool(np.sign(reference[-1]) != np.sign(candidate[-1])),
        "candidate_action": np.asarray(candidate).tolist(),
    }


def timed(call: Callable[[], Any], warmup: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values: list[float] = []
    output: Any = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return output, values


class OpenVLAWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA benchmark requires CUDA")
        register_openvla()
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(
            args.model, config=config, trust_remote_code=False, use_fast=False
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            config=config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).eval().to("cuda:0")
        self.model.language_model.config.proportion_attn_var = None
        self.model.language_model.config.reusable_patches = None
        self.unnorm_key = args.unnorm_key
        self.prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
        self.scene = make_scene()
        self.args = args

    def inputs_for(self, image: Image.Image):
        return self.processor(self.prompt, image).to("cuda:0", dtype=torch.bfloat16)

    def _generate(self, image: Image.Image, output_attentions: bool = False):
        inputs = dict(self.inputs_for(image))
        input_ids = inputs.pop("input_ids")
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.full((1, 1), 29871, dtype=torch.long, device=input_ids.device)),
                dim=1,
            )
        return self.model.generate(
            input_ids,
            max_new_tokens=self.model.get_action_dim(self.unnorm_key),
            **inputs,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=output_attentions,
            past_key_values=DynamicCache(),
        )

    def _decode_action(self, generated_ids: torch.Tensor) -> np.ndarray:
        action_dim = self.model.get_action_dim(self.unnorm_key)
        token_ids = generated_ids[0, -action_dim:].cpu().numpy()
        discretized = self.model.vocab_size - token_ids
        discretized = np.clip(discretized - 1, 0, self.model.bin_centers.shape[0] - 1)
        normalized = self.model.bin_centers[discretized]
        stats = self.model.get_action_stats(self.unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
        return np.asarray(
            np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized),
            dtype=np.float32,
        ).reshape(-1)

    @torch.inference_mode()
    def infer(self, image: Image.Image) -> np.ndarray:
        self.model.language_model.config.proportion_attn_var = None
        self.model.language_model.config.reusable_patches = None
        return self._decode_action(self._generate(image).sequences)

    @torch.inference_mode()
    def infer_with_attention(self, image: Image.Image) -> tuple[np.ndarray, dict]:
        self.model.language_model.config.proportion_attn_var = None
        self.model.language_model.config.reusable_patches = None
        results = self._generate(image, output_attentions=True)
        if results.attentions is None:
            raise RuntimeError("OpenVLA generation returned no attention metadata")
        return self._decode_action(results.sequences), {
            "past_key_values": results.past_key_values,
            "attentions": results.attentions[0],
        }

    def run(self, name: str, baseline_action: np.ndarray, baseline_times: list[float]) -> dict[str, Any]:
        if name == "original":
            action, times = timed(lambda: self.infer(self.scene), self.args.warmup, self.args.repeats)
            return self.result(name, action, times, baseline_action, baseline_times, {})

        if name.startswith("fixed_foveation_keep"):
            ratio = 0.20 if name.endswith("20") else 0.50
            image = Image.fromarray(foveate_blur(np.asarray(self.scene), ratio))
            action, times = timed(lambda: self.infer(image), self.args.warmup, self.args.repeats)
            return self.result(name, action, times, baseline_action, baseline_times, {"fovea_keep_ratio": ratio})

        if name in {"action_repeat2", "action_repeat4"}:
            repeat = 2 if name.endswith("2") else 4
            action, times = timed(lambda: self.infer(self.scene), self.args.warmup, self.args.repeats)
            result = self.result(name, action, times, baseline_action, baseline_times, {"action_repeat": repeat})
            result["effective_control_step_latency_ms"] = float(np.median(times) / repeat)
            result["model_calls_per_control_step"] = 1.0 / repeat
            return result

        if name.startswith("depth_pruning"):
            count = int(name.removeprefix("depth_pruning"))
            layers = find_decoder_layers(self.model)
            if layers is None:
                raise RuntimeError("OpenVLA decoder layers not found")
            calibration_images = [make_scene(shift) for shift in (-8, -3, 3, 8)]
            influence = collect_block_influence(
                layers,
                [lambda image=image: self.infer(image) for image in calibration_images],
            )
            selected = select_non_adjacent(
                influence, count, min_layer_fraction=0.25, protect_last=1, min_gap=1
            )
            pruner = StaticDepthPruner(self.model)
            try:
                pruner.apply(selected)
                action, times = timed(lambda: self.infer(self.scene), self.args.warmup, self.args.repeats)
            finally:
                pruner.restore()
            return self.result(
                name,
                action,
                times,
                baseline_action,
                baseline_times,
                {"removed_layers": list(selected), "influence": influence},
            )

        if name in GUARDED:
            frame_mae, local_mae, cosine, max_consecutive = GUARDED[name]
            records: list[dict[str, Any]] = []
            for _ in range(self.args.repeats):
                controller = ConservativeActionReuse(
                    max_frame_mae=frame_mae,
                    max_local_patch_mae=local_mae,
                    min_action_cosine=cosine,
                    min_translation_norm=0.01,
                    max_consecutive_reuse=max_consecutive,
                )
                step_times: list[float] = []
                action = None
                for _step in range(3):
                    started = time.perf_counter()
                    action, reused = controller.step(self.scene, lambda: self.infer(self.scene))
                    torch.cuda.synchronize()
                    step_times.append((time.perf_counter() - started) * 1000.0)
                    records.append({"reused": reused, "latency_ms": step_times[-1]})
            assert action is not None
            times = [record["latency_ms"] for record in records]
            result = self.result(
                name,
                action,
                times,
                baseline_action,
                baseline_times,
                {"thresholds": {"frame_mae": frame_mae, "local_patch_mae": local_mae, "action_cosine": cosine, "max_consecutive": max_consecutive}},
            )
            result["model_calls"] = sum(not record["reused"] for record in records) // self.args.repeats
            result["reuses"] = sum(record["reused"] for record in records) // self.args.repeats
            result["records_per_sequence"] = records[:3]
            return result

        if name in {
            "temporal_fusion_motion_entropy",
            "temporal_fusion_task_aware",
            "temporal_fusion_conservative_adaptive",
        }:
            task_aware = name == "temporal_fusion_task_aware"
            conservative = name == "temporal_fusion_conservative_adaptive"
            fusion = InteractionAwareTemporalFusion(
                keyframe_interval=2 if conservative else 3,
                grid_size=(16, 16),
                motion_threshold=0.01,
                entropy_protect_fraction=0.15,
                task_protect_fraction=0.20,
                protect_radius=1,
                max_reuse_fraction=0.25 if conservative else 0.50,
            )
            fusion.attach(self.model.projector)
            try:
                action = None
                times: list[float] = []
                reusable: list[int] = []
                for _ in range(self.args.repeats):
                    fusion.reset()
                    fusion.prepare(self.scene)
                    if task_aware:
                        _, context = self.infer_with_attention(self.scene)
                        relevance = visual_task_relevance(context["attentions"]).detach().cpu().numpy()
                        fusion.prepare(self.scene, task_relevance=relevance)
                    else:
                        self.infer(self.scene)
                        decision = fusion.prepare(self.scene)
                        reusable.append(len(decision.reusable_patch_ids) if decision is not None else 0)

                    def fused_call():
                        nonlocal action
                        if task_aware:
                            decision = fusion.prepare(self.scene, task_relevance=relevance)
                            reusable.append(len(decision.reusable_patch_ids) if decision is not None else 0)
                            action, _ = self.infer_with_attention(self.scene)
                        else:
                            action = self.infer(self.scene)
                        return action

                    _, measured = timed(fused_call, 0, 1)
                    times.extend(measured)
            finally:
                fusion.detach()
            assert action is not None
            return self.result(
                name,
                action,
                times,
                baseline_action,
                baseline_times,
                {
                    "keyframe_interval": 2 if conservative else 3,
                    "max_reuse_fraction": 0.25 if conservative else 0.50,
                    "task_relevance": task_aware,
                    "median_reusable_visual_tokens": int(np.median(reusable)) if reusable else 0,
                    "event_motion_threshold": 0.03 if conservative else None,
                    "model_speed_mechanism": False,
                },
            )

        raise ValueError(f"unknown OpenVLA config: {name}")

    def result(
        self,
        name: str,
        action: np.ndarray,
        times: list[float],
        baseline_action: np.ndarray,
        baseline_times: list[float],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        result = {
            "status": "completed",
            "model_name": "openvla_7b",
            "config_name": name,
            "model": self.args.model,
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "attention_backend": "sdpa",
                "worker": self.args.worker,
            },
            "scenario": "synthetic OpenVLA compatibility/latency diagnostic",
            "latency_ms": summarize(times),
            "dense_reference_latency_ms": summarize(baseline_times),
            "speedup_vs_dense": float(np.median(baseline_times) / np.median(times)),
            "parameters": parameters,
            "scope": "Synthetic action agreement and latency only; not robot-task accuracy.",
        }
        result.update(action_difference(baseline_action, action))
        result["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
        return result


def main() -> None:
    args = parse_args()
    names = tuple(name.strip() for name in args.configs.split(",") if name.strip())
    invalid = sorted(set(names) - set(ALL_CONFIGS))
    if invalid:
        raise ValueError(f"unknown configs: {invalid}")
    if not names:
        raise ValueError("at least one config is required")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    worker = OpenVLAWorker(args)
    baseline_action, baseline_times = timed(
        lambda: worker.infer(worker.scene), args.warmup, args.repeats
    )
    for name in names:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = worker.run(name, baseline_action, baseline_times)
        except Exception as exc:
            traceback.print_exc()
            result = {
                "status": "error",
                "model_name": "openvla_7b",
                "config_name": name,
                "model": args.model,
                "worker": args.worker,
                "scenario": "synthetic OpenVLA compatibility/latency diagnostic",
                "error": repr(exc),
                "scope": "Synthetic action agreement and latency only; not robot-task accuracy.",
            }
        output = args.output_root / name / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "config": name,
            "status": result["status"],
            "output": str(output),
            "median_ms": result.get("latency_ms", {}).get("median_ms"),
            "speedup_vs_dense": result.get("speedup_vs_dense"),
        }), flush=True)


if __name__ == "__main__":
    main()
