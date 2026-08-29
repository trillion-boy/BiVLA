#!/usr/bin/env python3
"""Static-frame compatibility and latency audit for the official VLA-Cache.

This intentionally reports action agreement, not task accuracy. Paper claims
must come from paired LIBERO rollouts with the benchmark protocol in
docs/experiment_protocol.md.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Both OpenVLA and OpenVLA-OFT install a top-level ``prismatic`` package. Put
# the audited VLA-Cache checkout first so import order cannot select OFT.
sys.path.insert(0, str(PROJECT_ROOT / "third_party/vla-cache/src/openvla"))

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    DynamicCache,
)

from experiments.robot.vla_cache_utils import (
    find_static_patches,
    get_layer_mask_schedule,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from vla_tricks.vla_cache import task_relevant_static_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--static-patches", type=int, default=130)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def register_openvla() -> None:
    """Use the checked-out official implementation, not checkpoint remote code."""
    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(
        OpenVLAConfig, PrismaticImageProcessor, exist_ok=True
    )
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(
        OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True
    )


def make_scene(shift: int = 0) -> Image.Image:
    # VLA-Cache assumes a 16 x 16 grid of 14-pixel patches.
    image = Image.new("RGB", (224, 224), (174, 148, 112))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 148, 224, 224), fill=(115, 85, 58))
    draw.ellipse((58 + shift, 78, 100 + shift, 120), fill=(30, 30, 30))
    draw.rectangle((137, 88, 176, 128), fill=(215, 52, 42))
    draw.rectangle((151, 65, 162, 91), fill=(210, 210, 210))
    return image


def timed(callable_) -> tuple[tuple[np.ndarray, dict], float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = callable_()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.percentile(array, 95)),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def action_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
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
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    register_openvla()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(
        args.model, config=config, trust_remote_code=False, use_fast=False
    )
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        config=config,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval().to("cuda:0")

    prompt = "In: What action should the robot take to pick up the black bowl?\nOut:"
    previous_image = make_scene(0)
    current_image = make_scene(0)

    def inputs_for(image: Image.Image):
        return processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    @torch.inference_mode()
    def dense(image: Image.Image) -> tuple[np.ndarray, dict]:
        model.language_model.config.proportion_attn_var = None
        model.language_model.config.reusable_patches = None
        return model.predict_action(
            **inputs_for(image),
            unnorm_key=args.unnorm_key,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
            past_key_values=DynamicCache(),
        )

    @torch.inference_mode()
    def cached(image: Image.Image, previous_cache: dict) -> tuple[np.ndarray, dict, int]:
        if not isinstance(previous_cache, dict):
            raise TypeError(
                f"expected cache metadata dict, got {type(previous_cache).__name__}"
            )
        static = find_static_patches(
            image, previous_image, top_k=args.static_patches
        )
        reusable = task_relevant_static_tokens(
            previous_cache["attentions"], static
        )
        model.language_model.config.reusable_patches = torch.tensor(
            reusable, device="cuda:0", dtype=torch.long
        )
        model.language_model.config.proportion_attn_var = get_layer_mask_schedule(
            previous_cache["attentions"]
        )
        action, next_cache = model.predict_action(
            **inputs_for(image),
            unnorm_key=args.unnorm_key,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
            past_key_values=previous_cache["past_key_values"],
        )
        return action, next_cache, len(reusable)

    # Warm both paths. A new dense context is required because DynamicCache is
    # mutated in place by the subsequent cached call.
    for _ in range(args.warmup):
        _, context = dense(previous_image)
        cached(current_image, context)
        dense(current_image)

    dense_times: list[float] = []
    cached_times: list[float] = []
    dense_action = None
    cached_action = None
    reusable_count = 0
    for _ in range(args.repeats):
        # Context creation is the normal first frame of a cache sequence and is
        # not included in steady-state cached latency.
        _, context = dense(previous_image)
        (cached_action, _, reusable_count), cache_ms = timed(
            lambda context=context: cached(current_image, context)
        )
        (dense_action, _), dense_ms = timed(lambda: dense(current_image))
        cached_times.append(cache_ms)
        dense_times.append(dense_ms)

    assert dense_action is not None and cached_action is not None
    result = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "attention_backend": "eager (required for attention maps)",
        },
        "model": args.model,
        "scenario": "identical consecutive synthetic frames",
        "dense": {
            **summarize(dense_times),
            "action": np.asarray(dense_action).tolist(),
        },
        "vla_cache_steady_state": {
            **summarize(cached_times),
            **action_difference(dense_action, cached_action),
            "reusable_visual_tokens": reusable_count,
            "speedup_vs_dense": float(
                np.median(dense_times) / np.median(cached_times)
            ),
        },
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "scope": (
            "Implementation and steady-state latency check only; paired "
            "LIBERO rollouts are required for accuracy claims."
        ),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
