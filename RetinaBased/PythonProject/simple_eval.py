#!/usr/bin/env python3
"""Run one of the supported VLA policies on a SimplerEnv task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from typing import Optional

import numpy as np
from PIL import Image as PILImage

ROOT = os.path.abspath(os.path.dirname(__file__))
SIMPLER_ENV_ROOT = os.environ.get("SIMPLER_ENV_PATH", os.path.join(ROOT, "SimplerEnv"))
MANSKILL_ROOT = os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim")

for path in [ROOT, SIMPLER_ENV_ROOT, MANSKILL_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import transformers.models.paligemma.processing_paligemma as _paligemma_processing
import transformers.processing_utils as _pu
import transformers.utils.import_utils as _tui
import transformers as _transformers
from transformers.image_utils import is_valid_image

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: True

if not getattr(_pu.ProcessorMixin, "_check_patched", False):
    _pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    _pu.ProcessorMixin._check_patched = True

if not hasattr(_pu, "MultiModalData"):
    class _MultiModalData(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)

    _pu.MultiModalData = _MultiModalData

if "transformers.video_utils" not in sys.modules:
    _video_utils = types.ModuleType("transformers.video_utils")
    _video_utils.VideoInput = object
    sys.modules["transformers.video_utils"] = _video_utils

if not hasattr(_transformers, "AutoVideoProcessor"):
    class _AutoVideoProcessor:
        model_input_names = []

        def __call__(self, *call_args, **call_kwargs):
            return {}

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _transformers.AutoVideoProcessor = _AutoVideoProcessor

if "lightning" not in sys.modules:
    for name in ["lightning", "lightning.pytorch", "lightning.pytorch.trainer"]:
        sys.modules.setdefault(name, types.ModuleType(name))

    class _Trainer:
        pass

    sys.modules["lightning.pytorch.trainer"].Trainer = _Trainer

if not hasattr(_paligemma_processing, "make_batched_images"):
    def _make_batched_images(images):
        if images is None:
            return None
        if is_valid_image(images):
            return [[images]]
        if isinstance(images, (list, tuple)) and images:
            if is_valid_image(images[0]):
                return [[image] for image in images]
            if isinstance(images[0], (list, tuple)) and images[0] and is_valid_image(images[0][0]):
                return [list(batch) for batch in images]
        raise ValueError("images must be an image, list of images or list of list of images")

    _paligemma_processing.make_batched_images = _make_batched_images

from openvla_inference import (
    ActionRepeatOpenVLAInference,
    BlurFoveatedOpenVLAInference,
    FoveatedOpenVLAInference,
    OpenVLAInference,
    RetinotopicCachedOpenVLAInference,
)
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


TASK_CONFIGS = {
    "widowx_put_eggplant_in_basket": {
        "env_name": "PutEggplantInBasketScene-v0",
        "robot": "widowx_sink_camera_setup",
        "scene_name": "bridge_table_1_v2",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 120,
        "robot_init_x": 0.127,
        "robot_init_y": 0.06,
    },
    "widowx_carrot_on_plate": {
        "env_name": "PutCarrotOnPlateInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
    "widowx_stack_cube": {
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
    "widowx_spoon_on_towel": {
        "env_name": "PutSpoonOnTableClothInScene-v0",
        "robot": "widowx",
        "scene_name": "bridge_table_1_v1",
        "rgb_overlay_path": "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "rgb_overlay_cameras": ["3rd_view_camera"],
        "obj_episode_range": [0, 24],
        "obs_camera_name": "3rd_view_camera",
        "control_freq": 5,
        "sim_freq": 500,
        "max_episode_steps": 60,
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "openvla",
            "openvla_foveated",
            "openvla_foveated_blur",
            "openvla_chunk",
            "openvla_retina",
        ],
        required=True,
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="openvla_chunk only: execute each predicted action this many "
        "env steps (OpenVLA predicts one action per forward, so the "
        "chunk-exec analog is action-repeat; k=2 halves the forwards).",
    )
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS.keys()), required=True)
    parser.add_argument("--n-episodes", type=int, default=24)
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openvla-model-path", default="openvla/openvla-7b")
    parser.add_argument("--openvla-unnorm-key", default="bridge_orig")
    parser.add_argument("--foveated-keep-percent", type=float, default=20.0)
    parser.add_argument("--retina-mid-refresh", type=int, default=2)
    parser.add_argument("--retina-outer-refresh", type=int, default=4)
    parser.add_argument("--retina-max-action-reuse", type=int, default=2)
    parser.add_argument("--retina-fovea-fraction", type=float, default=0.22)
    parser.add_argument("--retina-mid-fraction", type=float, default=0.55)
    parser.add_argument("--retina-fovea-motion-threshold", type=float, default=0.015)
    parser.add_argument("--retina-mid-motion-threshold", type=float, default=0.025)
    parser.add_argument("--retina-outer-motion-threshold", type=float, default=0.040)
    parser.add_argument("--retina-full-refresh-motion-threshold", type=float, default=0.060)
    parser.add_argument("--retina-action-motion-threshold", type=float, default=0.018)
    parser.add_argument("--retina-max-reuse-action-norm", type=float, default=0.08)
    return parser.parse_args()


def save_gif(path: str, frames: list[np.ndarray]) -> None:
    images = [PILImage.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=100)


def get_video_frame(model, fallback_image: np.ndarray) -> np.ndarray:
    prepared_getter = getattr(model, "last_prepared_image", None)
    if callable(prepared_getter):
        prepared = prepared_getter()
        if prepared is not None:
            return np.asarray(prepared, dtype=np.uint8)
    return np.asarray(fallback_image, dtype=np.uint8).copy()


def build_env(task_cfg: dict, ep_id: int, policy_name: str):
    robot = task_cfg["robot"]
    env_kwargs = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=task_cfg["sim_freq"],
        control_mode=get_robot_control_mode(robot, policy_name),
        control_freq=task_cfg["control_freq"],
        max_episode_steps=task_cfg["max_episode_steps"],
        scene_name=task_cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
        renderer_kwargs={"offscreen_only": True},
    )

    for base in [SIMPLER_ENV_ROOT, os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim")]:
        overlay_path = os.path.join(base, task_cfg["rgb_overlay_path"])
        if os.path.exists(overlay_path):
            env_kwargs["rgb_overlay_path"] = overlay_path
            env_kwargs["rgb_overlay_cameras"] = task_cfg["rgb_overlay_cameras"]
            break

    env = build_maniskill2_env(task_cfg["env_name"], **env_kwargs)

    reset_options = {"obj_init_options": {"episode_id": ep_id}}
    if "robot_init_x" in task_cfg and "robot_init_y" in task_cfg:
        reset_options["robot_init_options"] = {
            "init_xy": np.array([task_cfg["robot_init_x"], task_cfg["robot_init_y"]]),
            "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }

    obs, _ = env.reset(options=reset_options)
    return env, obs


def get_image(env, obs, camera_name: str) -> np.ndarray:
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera_name)


def get_oracle_context(env, obs, camera_name: str) -> dict:
    unwrapped = env.unwrapped
    seg = obs["image"][camera_name].get("Segmentation")
    actor_seg = None if seg is None else np.asarray(seg[..., 1], dtype=np.int32)

    source_actor_ids = []
    target_actor_ids = []

    source_obj = getattr(unwrapped, "episode_source_obj", None)
    target_obj = getattr(unwrapped, "episode_target_obj", None)
    if source_obj is not None and hasattr(source_obj, "id"):
        source_actor_ids.append(int(source_obj.id))
    if target_obj is not None and hasattr(target_obj, "id"):
        target_actor_ids.append(int(target_obj.id))

    sink = getattr(unwrapped, "sink", None)
    if sink is not None and hasattr(sink, "id"):
        target_actor_ids.append(int(sink.id))

    return {
        "actor_seg": actor_seg,
        "source_actor_ids": sorted(set(source_actor_ids)),
        "target_actor_ids": sorted(set(target_actor_ids)),
        "source_name": getattr(source_obj, "name", ""),
        "target_name": getattr(target_obj, "name", ""),
    }


def get_bridge_proprio(env, obs) -> np.ndarray:
    del obs
    unwrapped = env.unwrapped
    tcp = unwrapped.tcp.pose
    gripper_open = 1.0 - float(unwrapped.agent.get_gripper_closedness())
    return np.concatenate(
        [
            np.asarray(tcp.p, dtype=np.float32),
            np.asarray(tcp.q, dtype=np.float32),
            np.array([gripper_open], dtype=np.float32),
        ]
    )


def build_model(args):
    retina_kwargs = {
        "keep_percent": args.foveated_keep_percent,
        "mid_refresh_interval": args.retina_mid_refresh,
        "outer_refresh_interval": args.retina_outer_refresh,
        "max_action_reuse": args.retina_max_action_reuse,
        "fovea_fraction": args.retina_fovea_fraction,
        "mid_fraction": args.retina_mid_fraction,
        "fovea_motion_threshold": args.retina_fovea_motion_threshold,
        "mid_motion_threshold": args.retina_mid_motion_threshold,
        "outer_motion_threshold": args.retina_outer_motion_threshold,
        "full_refresh_motion_threshold": args.retina_full_refresh_motion_threshold,
        "action_motion_threshold": args.retina_action_motion_threshold,
        "max_reuse_action_norm": args.retina_max_reuse_action_norm,
    }

    if args.model == "openvla":
        return OpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key,
            device=args.device,
        )
    if args.model == "openvla_foveated":
        return FoveatedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key,
            device=args.device,
            keep_percent=args.foveated_keep_percent,
        )
    if args.model == "openvla_foveated_blur":
        return BlurFoveatedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key,
            device=args.device,
            keep_percent=args.foveated_keep_percent,
        )
    if args.model == "openvla_chunk":
        return ActionRepeatOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key,
            device=args.device,
            repeat_k=args.action_repeat,
        )
    if args.model == "openvla_retina":
        return RetinotopicCachedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key,
            device=args.device,
            **retina_kwargs,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def normalize_env_actions(step_output):
    if isinstance(step_output, tuple) and len(step_output) == 2:
        _, env_actions = step_output
    else:
        env_actions = step_output

    if isinstance(env_actions, dict):
        return [env_actions]
    if isinstance(env_actions, list):
        return env_actions
    raise TypeError(f"Unsupported action output type: {type(env_actions)!r}")


def main():
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]
    camera_name = task_cfg["obs_camera_name"]

    output_dir = args.output_dir or os.path.join("results", args.model, args.task)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[load] model={args.model}", flush=True)
    model = build_model(args)
    print("[ok] model loaded", flush=True)

    if args.episode_ids.strip():
        episode_ids = [int(part.strip()) for part in args.episode_ids.split(",") if part.strip()]
    else:
        base_ids = list(range(*task_cfg["obj_episode_range"]))
        episode_ids = [base_ids[idx % len(base_ids)] for idx in range(args.n_episodes)]

    results = []

    for run_idx, episode_id in enumerate(episode_ids):
        print(f"\n-- episode {run_idx:02d} (env_id={episode_id}) --", flush=True)
        env, obs = build_env(task_cfg, episode_id, policy_name=args.model)
        instruction = env.get_language_instruction()
        image = get_image(env, obs, camera_name)
        print(f"instruction: {instruction}", flush=True)

        model.reset()
        start_episode = getattr(model, "start_episode", None)
        if callable(start_episode):
            start_episode()
        done = False
        truncated = False
        final_info = {}
        step_count = 0
        grasped = False
        frames = []
        start_time = time.time()

        while not (done or truncated) and step_count < task_cfg["max_episode_steps"]:
            phase_info = {
                "grasped": grasped,
                "step_count": step_count,
                "proprio": get_bridge_proprio(env, obs),
            }
            oracle_context = get_oracle_context(env, obs, camera_name)
            env_actions = normalize_env_actions(
                model.step(
                    image,
                    instruction,
                    phase_info=phase_info,
                    oracle_context=oracle_context,
                )
            )
            if args.save_video and step_count % 4 == 0:
                frames.append(get_video_frame(model, image))
            for env_action in env_actions:
                flat_action = np.concatenate(
                    [
                        env_action["world_vector"],
                        env_action["rot_axangle"],
                        np.atleast_1d(env_action["gripper"]),
                    ]
                )
                obs, _, done, truncated, info = env.step(flat_action)
                final_info = info
                grasped = grasped or bool(info.get("is_src_obj_grasped", False))
                image = get_image(env, obs, camera_name)

                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    model.reset()

                step_count += 1
                if done or truncated or step_count >= task_cfg["max_episode_steps"]:
                    break

        elapsed = time.time() - start_time
        success = bool(final_info.get("success", False))
        status = "SUCCESS" if success else "FAIL"
        print(f"result: {status} grasped={grasped} steps={step_count} time={elapsed:.1f}s", flush=True)

        if args.save_video and frames:
            gif_path = os.path.join(output_dir, f"ep{run_idx:02d}_{status.lower()}.gif")
            save_gif(gif_path, frames)
            print(f"gif: {gif_path}", flush=True)

        model_stats = (
            model.episode_stats()
            if callable(getattr(model, "episode_stats", None))
            else None
        )
        # Amortized inference cost per ENV step: with action-repeat k, each
        # forward covers k env steps, so this falls ~k x while ms/infer is flat.
        if model_stats and step_count and "model_ms_per_infer" in model_stats:
            model_stats["model_ms_per_env_step"] = (
                model_stats.get("model_calls_timed", 0.0)
                * model_stats["model_ms_per_infer"]
                / step_count
            )

        results.append(
            {
                "ep": run_idx,
                "ep_id": episode_id,
                "success": success,
                "grasped": grasped,
                "terminated": bool(done),
                "truncated": bool(truncated),
                "steps": step_count,
                "elapsed": elapsed,
                "final_info": final_info,
                "model_stats": model_stats,
            }
        )
        env.close()

    success_count = sum(item["success"] for item in results)
    grasp_count = sum(item["grasped"] for item in results)
    summary = {
        "model": args.model,
        "task": args.task,
        "success_rate": success_count / len(results),
        "grasp_rate": grasp_count / len(results),
        "avg_steps": float(np.mean([item["steps"] for item in results])),
        "avg_elapsed": float(np.mean([item["elapsed"] for item in results])),
        "episodes": results,
    }

    stats_rows = [item["model_stats"] for item in results if item.get("model_stats")]
    if stats_rows:
        aggregate_stats = {}
        for key in stats_rows[0]:
            # Some diagnostic fields (e.g. postgrasp_reuse_rate) are None on
            # episodes where that phase never occurred (never grasped) --
            # skip those rather than crashing on float(None).
            values = [float(row[key]) for row in stats_rows if row.get(key) is not None]
            if values:
                aggregate_stats[key] = float(np.mean(values))
        summary["model_stats"] = aggregate_stats

    print("\n==================================================", flush=True)
    print(f"task:         {args.task}", flush=True)
    print(f"model:        {args.model}", flush=True)
    print(f"success_rate: {success_count}/{len(results)} = {summary['success_rate']:.1%}", flush=True)
    print(f"grasp_rate:   {grasp_count}/{len(results)} = {summary['grasp_rate']:.1%}", flush=True)
    print(f"avg_steps:    {summary['avg_steps']:.0f}", flush=True)
    print(f"avg_elapsed:  {summary['avg_elapsed']:.1f}s", flush=True)
    model_ms = (summary.get("model_stats") or {}).get("model_ms_per_infer")
    if model_ms is not None:
        print(f"model_ms_per_infer: {model_ms:.1f}ms  (pure model forward pass, CUDA-synced)", flush=True)
    step_ms = (summary.get("model_stats") or {}).get("model_ms_per_env_step")
    if step_ms is not None:
        print(f"model_ms_per_env_step: {step_ms:.1f}ms  (amortized over executed actions)", flush=True)
    if args.model == "openvla_chunk":
        print(f"action-repeat: each prediction executed for {args.action_repeat} env steps", flush=True)
    if args.model == "openvla_retina":
        ms = summary.get("model_stats") or {}
        pre_rr, post_rr = ms.get("pregrasp_reuse_rate"), ms.get("postgrasp_reuse_rate")
        pre_fr, post_fr = ms.get("pregrasp_mean_refresh_ratio"), ms.get("postgrasp_mean_refresh_ratio")
        print(
            "reuse_rate  pre-grasp: "
            + (f"{pre_rr:.1%}" if pre_rr is not None else "n/a")
            + "  post-grasp: "
            + (f"{post_rr:.1%}" if post_rr is not None else "n/a"),
            flush=True,
        )
        print(
            "mean_refresh_ratio  pre-grasp: "
            + (f"{pre_fr:.1%}" if pre_fr is not None else "n/a")
            + "  post-grasp: "
            + (f"{post_fr:.1%}" if post_fr is not None else "n/a")
            + "  (lower refresh_ratio = more stale/cached image at decision time)",
            flush=True,
        )
    print("==================================================", flush=True)

    save_path = os.path.join(output_dir, f"results_{args.task}.json")
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"saved: {save_path}", flush=True)


if __name__ == "__main__":
    main()
