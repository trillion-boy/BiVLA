#!/usr/bin/env python3
"""Run MiniVLA on SimplerEnv WidowX with matched training-free tricks.

This evaluator intentionally uses the local Bridge MiniVLA checkpoint and the
official SimplerEnv WidowX visual-matching tasks.  It reports task success,
environment steps, model calls, query latency, and end-to-end episode time.
The VLA-Cache condition is emitted as not-applicable because the released
cache implementation targets the OpenVLA-7B Hugging Face architecture, while
MiniVLA is a native Prismatic model with a different cache contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT.parent / "AWARe-VLA"
MINIVLA_SOURCE = EXTERNAL_ROOT / "utilities" / "openvla-mini"
SIMPLER_SOURCE = EXTERNAL_ROOT / "utilities" / "SimplerEnv"
for source in (ROOT / "src", EXTERNAL_ROOT, MINIVLA_SOURCE, SIMPLER_SOURCE):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from vla_tricks.depth import (  # noqa: E402
    collect_block_influence,
    find_decoder_layers,
    select_non_adjacent,
    StaticDepthPruner,
)
from vla_tricks.foveation import foveate_blur  # noqa: E402
from vla_tricks.perception import (  # noqa: E402
    InteractionAwareTemporalFusion,
    patch_motion,
)
from vla_tricks.temporal import ConservativeActionReuse  # noqa: E402


TASKS = (
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
)
CONDITIONS = (
    "original",
    "fixed_foveation",
    "action_repeat",
    "depth_pruning",
    "guarded_action_reuse",
    "temporal_fusion",
    "vla_cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model-name", default="minivla_simplerenv")
    parser.add_argument("--config-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--depth-layers", type=int, default=1)
    parser.add_argument("--fovea-keep-ratio", type=float, default=0.20)
    parser.add_argument("--fusion-collect-relevance", action="store_true")
    parser.add_argument("--fusion-keyframe-interval", type=int, default=3)
    parser.add_argument("--fusion-motion-threshold", type=float, default=0.01)
    parser.add_argument("--fusion-entropy-protect-fraction", type=float, default=0.15)
    parser.add_argument("--fusion-task-protect-fraction", type=float, default=0.20)
    parser.add_argument("--fusion-protect-radius", type=int, default=1)
    parser.add_argument("--fusion-max-reuse-fraction", type=float, default=0.50)
    parser.add_argument(
        "--fusion-event-motion-threshold",
        type=float,
        default=None,
        help="force a dense keyframe when any 16x16 patch exceeds this motion",
    )
    parser.add_argument("--reuse-max-frame-mae", type=float, default=0.01)
    parser.add_argument("--reuse-max-local-patch-mae", type=float, default=0.03)
    parser.add_argument("--reuse-min-action-cosine", type=float, default=0.995)
    parser.add_argument("--reuse-min-translation-norm", type=float, default=0.01)
    parser.add_argument("--reuse-max-consecutive", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_policy(checkpoint: Path, device: str):
    if not MINIVLA_SOURCE.is_dir():
        raise FileNotFoundError(
            f"MiniVLA source not found at {MINIVLA_SOURCE}; use the existing "
            "AWARe-VLA utilities or run its setup script first."
        )
    from aware.awre_vla.libero_eval import EvalConfig, MiniVLAPolicy

    # The Bridge VQ tokenizer stores its path relative to the openvla-mini
    # repository, so preserve the same working-directory contract as the
    # maintained MiniVLA evaluator.
    os.chdir(MINIVLA_SOURCE)
    config = EvalConfig(
        checkpoint=str(checkpoint),
        task_suite_name="simpler_widowx",
        unnorm_key="bridge_dataset",
        device=device,
        center_crop=False,
        obs_history=1,
    )
    policy = MiniVLAPolicy(config)
    resolved = Path(policy.checkpoint_path).resolve()
    if "minivla-vq-bridge" not in str(checkpoint).lower():
        raise RuntimeError(
            "The requested checkpoint is not the Bridge MiniVLA checkpoint. "
            f"Refusing to run WidowX with {checkpoint}."
        )
    if "bridge_dataset" not in policy.model.norm_stats:
        raise RuntimeError(
            "Loaded MiniVLA checkpoint has no bridge_dataset action statistics; "
            f"available keys: {sorted(policy.model.norm_stats)}"
        )
    manifest = {
        "requested_checkpoint": str(checkpoint.resolve()),
        "resolved_checkpoint": str(resolved),
        "resolved_checkpoint_size_bytes": resolved.stat().st_size,
        "model_id": getattr(policy.model, "model_id", None),
        "base_vlm": "prism-qwen25-extra-dinosiglip-224px+0_5b",
        "action_statistics_key": policy.unnorm_key,
        "task_suite": "simpler_widowx",
        "image_resolution": 224,
        "center_crop": False,
    }
    print("CHECKPOINT_MANIFEST " + json.dumps(manifest, sort_keys=True))
    return policy, manifest


def collect_depth_calibration(policy, tasks: tuple[str, ...], seed: int) -> tuple[list[float], tuple[int, ...]]:
    """Collect one frame per task on disjoint seeds and rank MiniVLA blocks."""
    import simpler_env
    from experiments.robot.simpler.simpler_utils import get_simpler_img

    samples: list[tuple[np.ndarray, str]] = []
    for index, task_name in enumerate(tasks):
        env = simpler_env.make(task_name)
        try:
            obs, _ = env.reset(seed=seed + 10000 + index)
            samples.append((get_simpler_img(env, obs, 224), env.get_language_instruction()))
        finally:
            env.close()
    layers = find_decoder_layers(policy.model)
    if layers is None:
        raise RuntimeError("Could not find MiniVLA Qwen decoder layers for depth calibration")
    run_fns: list[Callable[[], Any]] = [
        lambda image=image, instruction=instruction: policy.predict(
            [image], instruction, collect_relevance=False
        )
        for image, instruction in samples
    ]
    influence = collect_block_influence(layers, run_fns)
    return influence, tuple()


def run_query(policy, image: np.ndarray, instruction: str, collect_relevance: bool) -> tuple[np.ndarray, Any, float]:
    cuda_sync()
    started = time.perf_counter()
    actions, feedback = policy.predict(
        [image], instruction, collect_relevance=collect_relevance
    )
    cuda_sync()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(actions[0], dtype=np.float32).reshape(-1), feedback, elapsed_ms


def evaluate(args: argparse.Namespace) -> None:
    if args.num_trials_per_task < 1 or args.max_steps < 1:
        raise ValueError("num-trials-per-task and max-steps must be positive")
    if args.action_repeat < 1:
        raise ValueError("action-repeat must be positive")
    if args.depth_layers < 0:
        raise ValueError("depth-layers must be non-negative")
    if args.fusion_keyframe_interval < 1:
        raise ValueError("fusion-keyframe-interval must be positive")
    if args.fusion_protect_radius < 0:
        raise ValueError("fusion-protect-radius must be non-negative")
    if not 0.0 <= args.fusion_max_reuse_fraction <= 1.0:
        raise ValueError("fusion-max-reuse-fraction must be in [0, 1]")
    if args.fusion_event_motion_threshold is not None and args.fusion_event_motion_threshold < 0:
        raise ValueError("fusion-event-motion-threshold must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.condition == "vla_cache":
        summary = {
            "status": "not_applicable",
            "condition": args.condition,
            "checkpoint": str(args.checkpoint.resolve()),
            "task_suite_name": "simpler_widowx",
            "reason": (
                "The released VLA-Cache path targets the OpenVLA-7B Hugging "
                "Face implementation. MiniVLA uses native Prismatic Qwen2.5 "
                "and has a different visual/cache contract; running it as if "
                "compatible would be an invalid comparison."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    set_seed(args.seed)
    policy, manifest = load_policy(args.checkpoint, args.device)

    depth_influence: list[float] | None = None
    selected_depth: tuple[int, ...] = tuple()
    pruner: StaticDepthPruner | None = None
    if args.condition == "depth_pruning":
        depth_influence, _ = collect_depth_calibration(policy, TASKS, args.seed)
        selected_depth = select_non_adjacent(
            depth_influence,
            args.depth_layers,
            min_layer_fraction=0.25,
            protect_last=1,
            min_gap=1,
        )
        pruner = StaticDepthPruner(policy.model)
        pruner.apply(selected_depth)
        print(f"DEPTH_CALIBRATION selected={list(selected_depth)} influence={depth_influence}")

    fusion: InteractionAwareTemporalFusion | None = None
    if args.condition == "temporal_fusion":
        fusion = InteractionAwareTemporalFusion(
            keyframe_interval=args.fusion_keyframe_interval,
            grid_size=(16, 16),
            motion_threshold=args.fusion_motion_threshold,
            entropy_protect_fraction=args.fusion_entropy_protect_fraction,
            task_protect_fraction=args.fusion_task_protect_fraction,
            protect_radius=args.fusion_protect_radius,
            max_reuse_fraction=args.fusion_max_reuse_fraction,
        )
        if not hasattr(policy.model, "projector"):
            raise RuntimeError("MiniVLA model does not expose the projector hook required for fusion")
        fusion.attach(policy.model.projector)

    import simpler_env
    from experiments.robot.simpler.simpler_utils import convert_maniskill, get_simpler_img

    records: list[dict[str, Any]] = []
    all_query_ms: list[float] = []
    all_inference_ms: list[float] = []
    total_calls = total_reuses = total_steps = total_successes = 0
    fusion_reusable_tokens: list[int] = []
    fusion_keyframes = 0

    try:
        with episodes_path.open("w", encoding="utf-8") as output:
            for task_index, task_name in enumerate(TASKS):
                env = simpler_env.make(task_name)
                try:
                    for episode_index in range(args.num_trials_per_task):
                        episode_seed = args.seed + episode_index
                        policy.reset()
                        if fusion is not None:
                            fusion.reset()
                        reuse_controller = (
                            ConservativeActionReuse(
                                max_frame_mae=args.reuse_max_frame_mae,
                                max_local_patch_mae=args.reuse_max_local_patch_mae,
                                min_action_cosine=args.reuse_min_action_cosine,
                                min_translation_norm=args.reuse_min_translation_norm,
                                max_consecutive_reuse=args.reuse_max_consecutive,
                            )
                            if args.condition == "guarded_action_reuse"
                            else None
                        )
                        if reuse_controller is not None:
                            reuse_controller.reset()
                        obs, _ = env.reset(seed=episode_seed)
                        instruction = env.get_language_instruction()
                        action_repeat_left = 0
                        held_action: np.ndarray | None = None
                        previous_relevance: np.ndarray | None = None
                        previous_fusion_image: np.ndarray | None = None
                        episode_query_ms: list[float] = []
                        episode_inference_ms: list[float] = []
                        episode_calls = episode_reuses = 0
                        episode_steps = 0
                        success = False
                        truncated = False
                        error: str | None = None
                        error_traceback: str | None = None
                        episode_started = time.perf_counter()
                        try:
                            for _step in range(args.max_steps):
                                inference_started = time.perf_counter()
                                image = get_simpler_img(env, obs, 224)
                                if args.condition == "fixed_foveation":
                                    image = foveate_blur(image, args.fovea_keep_ratio)

                                feedback = None
                                reused = False
                                decision = None
                                query_ms = 0.0

                                if args.condition == "action_repeat" and action_repeat_left > 0:
                                    if held_action is None:
                                        raise RuntimeError("action-repeat state lost its held action")
                                    action = held_action.copy()
                                    action_repeat_left -= 1
                                    reused = True
                                elif args.condition == "guarded_action_reuse":
                                    if reuse_controller is None:
                                        raise RuntimeError("guarded reuse controller was not initialized")
                                    query_box: dict[str, Any] = {}

                                    def infer() -> np.ndarray:
                                        result = run_query(policy, image, instruction, False)
                                        query_box["feedback"] = result[1]
                                        query_box["query_ms"] = result[2]
                                        return result[0]

                                    action, reused = reuse_controller.step(image, infer)
                                    feedback = query_box.get("feedback")
                                    query_ms = float(query_box.get("query_ms", 0.0))
                                else:
                                    if fusion is not None:
                                        force_event_keyframe = False
                                        if (
                                            previous_fusion_image is not None
                                            and args.fusion_event_motion_threshold is not None
                                        ):
                                            event_motion = patch_motion(
                                                previous_fusion_image,
                                                image,
                                                grid_size=(16, 16),
                                            )
                                            force_event_keyframe = bool(
                                                np.max(event_motion)
                                                > args.fusion_event_motion_threshold
                                            )
                                        decision = fusion.prepare(
                                            image,
                                            task_relevance=previous_relevance,
                                            force_keyframe=force_event_keyframe,
                                        )
                                        if decision is None:
                                            fusion_keyframes += 1
                                        else:
                                            fusion_reusable_tokens.append(
                                                len(decision.reusable_patch_ids)
                                            )
                                    action, feedback, query_ms = run_query(
                                        policy,
                                        image,
                                        instruction,
                                        args.condition == "temporal_fusion"
                                        and args.fusion_collect_relevance,
                                    )
                                    if args.condition == "action_repeat":
                                        held_action = action.copy()
                                        action_repeat_left = args.action_repeat - 1
                                    if fusion is not None:
                                        previous_fusion_image = image.copy()

                                if fusion is not None and feedback is not None:
                                    coarse = feedback.get("coarse_relevance_map")
                                    if coarse is not None:
                                        previous_relevance = np.asarray(coarse, dtype=np.float32).reshape(-1)
                                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                                episode_inference_ms.append(inference_ms)
                                if query_ms > 0.0:
                                    episode_query_ms.append(query_ms)
                                    all_query_ms.append(query_ms)
                                    episode_calls += 1
                                if reused:
                                    episode_reuses += 1

                                obs, _reward, success, truncated, info = env.step(
                                    convert_maniskill(action)
                                )
                                episode_steps += 1
                                if success or truncated:
                                    break
                                new_instruction = env.get_language_instruction()
                                if new_instruction != instruction:
                                    instruction = new_instruction
                        except Exception as exc:  # retain episode-level failure evidence
                            error = repr(exc)
                            error_traceback = traceback.format_exc()

                        episode_elapsed_ms = (time.perf_counter() - episode_started) * 1000.0
                        record = {
                            "model_name": args.model_name,
                            "config_name": args.config_name or args.condition,
                            "condition": args.condition,
                            "task": task_name,
                            "task_index": task_index,
                            "episode_index": episode_index,
                            "seed": episode_seed,
                            "success": bool(success),
                            "truncated": bool(truncated),
                            "steps_executed": episode_steps,
                            "policy_calls": episode_calls,
                            "reuses": episode_reuses,
                            "episode_elapsed_ms": episode_elapsed_ms,
                            "query_latency_ms": quantiles(episode_query_ms),
                            "inference_cycle_ms": quantiles(episode_inference_ms),
                            "error": error,
                            "error_traceback": error_traceback,
                        }
                        output.write(json.dumps(jsonable(record)) + "\n")
                        output.flush()
                        records.append(record)
                        total_successes += int(bool(success))
                        total_steps += episode_steps
                        total_calls += episode_calls
                        total_reuses += episode_reuses
                        all_inference_ms.extend(episode_inference_ms)
                        print(
                            f"{task_name} episode={episode_index + 1}/{args.num_trials_per_task} "
                            f"seed={episode_seed} success={bool(success)} steps={episode_steps} "
                            f"calls={episode_calls} reuses={episode_reuses}",
                            flush=True,
                        )
                finally:
                    env.close()
    finally:
        if fusion is not None:
            fusion.detach()
        if pruner is not None:
            pruner.restore()

    per_task: dict[str, dict[str, Any]] = {}
    for task_name in TASKS:
        task_records = [record for record in records if record["task"] == task_name]
        task_steps = [record["steps_executed"] for record in task_records]
        task_elapsed = [record["episode_elapsed_ms"] for record in task_records]
        task_calls = [record["policy_calls"] for record in task_records]
        task_successes = sum(int(record["success"]) for record in task_records)
        per_task[task_name] = {
            "episodes": len(task_records),
            "successes": task_successes,
            "success_rate": task_successes / len(task_records) if task_records else 0.0,
            "average_steps": float(np.mean(task_steps)) if task_steps else 0.0,
            "average_policy_calls": float(np.mean(task_calls)) if task_calls else 0.0,
            "average_episode_ms": float(np.mean(task_elapsed)) if task_elapsed else 0.0,
        }

    summary = {
        "status": "completed",
        "model_name": args.model_name,
        "config_name": args.config_name or args.condition,
        "condition": args.condition,
        "checkpoint_manifest": manifest,
        "task_suite_name": "simpler_widowx",
        "tasks": list(TASKS),
        "episodes": len(records),
        "successes": total_successes,
        "success_rate": total_successes / len(records) if records else 0.0,
        "average_steps": total_steps / len(records) if records else 0.0,
        "average_policy_calls": total_calls / len(records) if records else 0.0,
        "average_reuses": total_reuses / len(records) if records else 0.0,
        "policy_query_latency_ms": quantiles(all_query_ms),
        "inference_cycle_latency_ms": quantiles(all_inference_ms),
        "average_episode_ms": float(np.mean([r["episode_elapsed_ms"] for r in records])) if records else 0.0,
        "average_control_frequency_hz": (
            float(total_steps / (sum(r["episode_elapsed_ms"] for r in records) / 1000.0))
            if records and sum(r["episode_elapsed_ms"] for r in records) > 0
            else 0.0
        ),
        "per_task": per_task,
        "depth_calibration": {
            "influence": depth_influence,
            "selected_layers": list(selected_depth),
        },
        "fusion": {
            "collect_relevance": bool(args.fusion_collect_relevance),
            "median_reusable_visual_tokens": float(np.median(fusion_reusable_tokens))
            if fusion_reusable_tokens
            else 0.0,
            "keyframes": fusion_keyframes,
        },
        "arguments": jsonable(vars(args)),
        "claim_scope": (
            "Task success and timing from SimplerEnv rollouts. Compare paired "
            "seeds across conditions; synthetic action agreement is not used."
        ),
    }
    summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    evaluate(parse_args())
