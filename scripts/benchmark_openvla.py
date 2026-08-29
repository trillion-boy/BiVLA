#!/usr/bin/env python3
"""Local OpenVLA compatibility/latency audit for the proposed tricks.

This is deliberately not called an accuracy evaluation: no static image test
can substitute for paired robot rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_tricks.depth import (
    StaticDepthPruner,
    collect_block_influence,
    find_decoder_layers,
    select_non_adjacent,
)
from vla_tricks.foveation import foveate_blur
from vla_tricks.perception import InteractionAwareTemporalFusion
from vla_tricks.temporal import ConservativeActionReuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--backend", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--prune-layers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_scene(shift: int = 0) -> Image.Image:
    image = Image.new("RGB", (256, 256), (174, 148, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 170, 256, 256), fill=(115, 85, 58))
    draw.ellipse((68 + shift, 92, 116 + shift, 140), fill=(30, 30, 30))
    draw.rectangle((157, 105, 202, 151), fill=(215, 52, 42))
    draw.rectangle((174, 78, 185, 108), fill=(210, 210, 210))
    return image


def timed(run, warmup: int, repeats: int) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    values = []
    output = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = run()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(output), values


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "candidate_action": np.asarray(candidate).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation=args.backend,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().to("cuda:0")
    # The official VLA-Cache Transformers fork expects these runtime fields on
    # every Llama config, including when cache reuse is disabled.
    model.language_model.config.proportion_attn_var = None
    model.language_model.config.reusable_patches = None

    prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"

    def infer(image: Image.Image) -> np.ndarray:
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            return model.predict_action(
                **inputs, unnorm_key=args.unnorm_key, do_sample=False
            )

    scene = make_scene()
    baseline_action, baseline_times = timed(
        lambda: infer(scene), args.warmup, args.repeats
    )
    result: dict[str, object] = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "backend": args.backend,
        },
        "model": args.model,
        "baseline": {
            **summarize(baseline_times),
            "action": baseline_action.tolist(),
        },
    }

    foveated = Image.fromarray(foveate_blur(np.asarray(scene), 0.20))
    foveated_action, foveated_times = timed(
        lambda: infer(foveated), args.warmup, args.repeats
    )
    result["fixed_foveation"] = {
        **summarize(foveated_times),
        **difference(baseline_action, foveated_action),
        "model_speed_mechanism": False,
    }

    # TTF-style projected-token fusion is an accuracy/robustness candidate.
    # Its patch decision is also the interface for future selective KV reuse;
    # fusion alone does not remove any model computation.
    fusion = InteractionAwareTemporalFusion(
        keyframe_interval=3,
        grid_size=(16, 16),
        motion_threshold=0.01,
        entropy_protect_fraction=0.15,
        protect_radius=1,
        max_reuse_fraction=0.50,
    )
    fusion.attach(model.projector)
    fusion_times: list[float] = []
    fusion_action = None
    reusable_counts: list[int] = []
    try:
        for _ in range(args.repeats):
            fusion.reset()
            array = np.asarray(scene)
            fusion.prepare(array)
            infer(scene)  # Dense keyframe seeds the historical representation.
            torch.cuda.synchronize()
            start = time.perf_counter()
            decision = fusion.prepare(array)
            fusion_action = infer(scene)
            torch.cuda.synchronize()
            fusion_times.append((time.perf_counter() - start) * 1000.0)
            reusable_counts.append(
                len(decision.reusable_patch_ids) if decision is not None else 0
            )
    finally:
        fusion.detach()
    assert fusion_action is not None
    result["interaction_aware_temporal_fusion"] = {
        **summarize(fusion_times),
        **difference(baseline_action, fusion_action),
        "reusable_visual_tokens": int(np.median(reusable_counts)),
        "model_speed_mechanism": False,
        "cache_acceleration_interface": True,
        "note": "projected-token denoising compatibility only; paired rollouts are required",
    }

    layers = find_decoder_layers(model)
    if layers is None:
        result["depth_pruning"] = {"error": "decoder stack not found"}
    elif args.prune_layers == 0:
        result["depth_pruning"] = {"skipped": True}
    else:
        calibration_images = [make_scene(shift) for shift in (-8, -3, 3, 8)]
        run_fns = [lambda image=image: infer(image) for image in calibration_images]
        influence = collect_block_influence(layers, run_fns)
        selected = select_non_adjacent(
            influence,
            args.prune_layers,
            min_layer_fraction=0.25,
            protect_last=1,
            min_gap=1,
        )
        pruner = StaticDepthPruner(model)
        try:
            pruner.apply(selected)
            pruned_action, pruned_times = timed(
                lambda: infer(scene), args.warmup, args.repeats
            )
            result["depth_pruning"] = {
                **summarize(pruned_times),
                **difference(baseline_action, pruned_action),
                "selected_layers": list(selected),
                "influence": influence,
                "speedup_vs_baseline": float(
                    np.median(baseline_times) / np.median(pruned_times)
                ),
            }
        finally:
            pruner.restore()

    reuse = ConservativeActionReuse(
        max_frame_mae=0.002,
        min_action_cosine=0.995,
        min_translation_norm=0.0,
        max_consecutive_reuse=1,
    )
    sequence = [np.asarray(make_scene(0)), np.asarray(make_scene(0)), np.asarray(make_scene(0))]
    reuse_records = []
    for array in sequence:
        start = time.perf_counter()
        action, reused = reuse.step(array, lambda array=array: infer(Image.fromarray(array)))
        torch.cuda.synchronize()
        reuse_records.append(
            {"reused": reused, "latency_ms": (time.perf_counter() - start) * 1000, "action": action.tolist()}
        )
    result["conservative_action_reuse"] = {
        "model_calls": reuse.calls,
        "reuses": reuse.reuses,
        "records": reuse_records,
        "note": "compatibility demonstration only; thresholds require held-out calibration",
    }
    result["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
