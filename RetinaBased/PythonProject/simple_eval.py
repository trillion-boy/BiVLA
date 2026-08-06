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
    DepthPrunedOpenVLAInference,
    FoveatedOpenVLAInference,
    OpenVLAInference,
    RetinotopicCachedOpenVLAInference,
)
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def _load_fractal_protocol():
    """Import the repo-root Google Robot protocol module.

    Imported, never copied. The SpatialVLA harness imports the same module, and
    the cross-backbone claim only means something if `--task X --episode-ids 7`
    lands on the SAME initial state in both. A local copy of the table here
    would drift and no result would show it.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(here, "simpler_fractal_protocol.py")):
            if here not in sys.path:
                sys.path.insert(0, here)
            import simpler_fractal_protocol as proto
            return proto
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise ImportError(
        "simpler_fractal_protocol.py not found in any parent of "
        f"{os.path.dirname(__file__)}. It lives at the BiVLA repo root and "
        "holds the Google Robot episode->initial-state mapping this harness "
        "shares with the SpatialVLA one."
    )


_proto = _load_fractal_protocol()
GOOGLE_ROBOT_TASKS = _proto.GOOGLE_ROBOT_TASKS
build_prepackaged_env = _proto.build_prepackaged_env
step_grasped = _proto.step_grasped
grasp_is_reported = _proto.grasp_is_reported
episode_grasped = _proto.episode_grasped


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

    # Google Robot / Fractal tasks come from the repo-root protocol module,
    # which the SpatialVLA harness imports too. One table, so an episode index
    # cannot come to mean two different initial states in two harnesses.
    **GOOGLE_ROBOT_TASKS,
}


def policy_setup_for(task: str) -> str:
    """-> the embodiment convention this task is scored under.

    Derived from the task, never passed in. The gripper convention and the
    action-unnormalization statistics both hinge on this, and both fail
    quietly: a Google Robot run under Bridge statistics does not crash, it
    produces a low-but-plausible success rate that reads as a real result.
    """
    return "google_robot" if task.startswith("google_robot") else "widowx_bridge"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "openvla",
            "openvla_foveated",
            "openvla_foveated_blur",
            "openvla_chunk",
            "openvla_depth",
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
    parser.add_argument(
        "--depth-prune",
        type=int,
        default=1,
        help="openvla_depth only: bypass this many of the most redundant "
        "decoder layers for the whole episode (fixed depth pruning). The "
        "ranking rule and its safeguards are shared with the other backbones "
        "via adaptive_sparse_vla/depth_prune.py -- change the COUNT per "
        "backbone if you must, never the rule.",
    )
    parser.add_argument(
        "--depth-min-layer",
        type=float,
        default=0.5,
        help="openvla_depth only: only the back (1 - this) fraction of the "
        "decoder stack is eligible for bypass. Early layers perform the "
        "foundational transforms everything downstream depends on.",
    )
    parser.add_argument(
        "--depth-min-gap",
        type=int,
        default=1,
        help="openvla_depth only: bypassed layers must be at least this far "
        "apart, so two adjacent layers are not both removed while cheaper "
        "candidates remain.",
    )
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=0,
        help="how many initial states to run. 0 = every state this task's "
        "protocol defines, which is what a reported number should be. A "
        "smaller count takes an ORDERED PREFIX, which is a biased sample "
        "wherever the ids are grouped (MoveNear's are grouped by object "
        "triplet, so the first 24 of 60 cover only two of five) -- fine for a "
        "quick check, not for a result.",
    )
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--openvla-model-path", default="openvla/openvla-7b")
    parser.add_argument(
        "--openvla-unnorm-key",
        default="",
        help="action un-normalization statistics. Empty = derive from the "
        "task's embodiment (bridge_orig for widowx_*, fractal20220817_data "
        "for google_robot_*), which is what the checkpoint was evaluated "
        "under. Override only to test a deliberate mismatch.",
    )
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


def build_env(task_cfg: dict, ep_id: int, policy_name: str, task_name: str = ""):
    # Google Robot / Fractal: build_prepackaged_env is shared with the
    # SpatialVLA harness, so both back ends hit the same env with the same
    # reset options for a given episode index.
    if task_cfg.get("prepackaged"):
        if not task_name:
            raise ValueError("prepackaged tasks need task_name to pick an env id")
        return build_prepackaged_env(task_cfg, ep_id, task_name, SIMPLER_ENV_ROOT)

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
    try:
        return _get_oracle_context(env, obs, camera_name)
    except (AttributeError, KeyError, TypeError):
        # Same reasoning as get_bridge_proprio: only the oracle-gaze
        # conditions read this, and the Google Robot envs name their actors
        # differently. Returning empty is honest -- an oracle condition then
        # has no oracle and will say so -- whereas crashing loses the whole run.
        return {"actor_seg": None, "source_actor_ids": [], "target_actor_ids": [],
                "source_name": "", "target_name": ""}


def _get_oracle_context(env, obs, camera_name: str) -> dict:
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


def get_bridge_proprio(env, obs) -> Optional[np.ndarray]:
    """TCP pose + gripper opening, or None if this env does not expose them.

    Only the oracle-gaze and retina conditions read this; the OpenVLA policy
    itself discards it. The Google Robot envs do not expose the same agent
    accessors, so this returns None there rather than aborting an episode over
    a field nothing in the condition being measured actually consumes.
    """
    del obs
    try:
        unwrapped = env.unwrapped
        tcp = unwrapped.tcp.pose
        gripper_open = 1.0 - float(unwrapped.agent.get_gripper_closedness())
    except (AttributeError, TypeError, ValueError):
        return None
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
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
            device=args.device,
        )
    if args.model == "openvla_foveated":
        return FoveatedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
            device=args.device,
            keep_percent=args.foveated_keep_percent,
        )
    if args.model == "openvla_foveated_blur":
        return BlurFoveatedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
            device=args.device,
            keep_percent=args.foveated_keep_percent,
        )
    if args.model == "openvla_chunk":
        return ActionRepeatOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
            device=args.device,
            repeat_k=args.action_repeat,
        )
    if args.model == "openvla_depth":
        return DepthPrunedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
            device=args.device,
            depth_prune=args.depth_prune,
            depth_min_layer=args.depth_min_layer,
            depth_min_gap=args.depth_min_gap,
        )
    if args.model == "openvla_retina":
        return RetinotopicCachedOpenVLAInference(
            model_path=args.openvla_model_path,
            unnorm_key=args.openvla_unnorm_key or None,
            policy_setup=policy_setup_for(args.task),
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

    setup = policy_setup_for(args.task)
    print(f"[load] model={args.model} policy_setup={setup} "
          f"unnorm_key={args.openvla_unnorm_key or '(derived)'}", flush=True)
    model = build_model(args)
    print(f"[ok] model loaded  unnorm_key={model.unnorm_key} "
          f"sticky_gripper={model.sticky_gripper_num_repeat}", flush=True)

    if args.episode_ids.strip():
        episode_ids = [int(part.strip()) for part in args.episode_ids.split(",") if part.strip()]
    else:
        base_ids = list(range(*task_cfg["obj_episode_range"]))
        # n_episodes 0 = the whole protocol. Anything smaller is an ordered
        # prefix; see the --n-episodes help.
        n_want = args.n_episodes or len(base_ids)
        episode_ids = [base_ids[idx % len(base_ids)] for idx in range(n_want)]
    if len(episode_ids) < len(range(*task_cfg["obj_episode_range"])):
        print(f"[warn] running {len(episode_ids)} of this task's "
              f"{len(range(*task_cfg['obj_episode_range']))} protocol episodes "
              f"-- an ordered prefix, not an unbiased subsample", flush=True)

    results = []
    grasp_seen = False

    for run_idx, episode_id in enumerate(episode_ids):
        print(f"\n-- episode {run_idx:02d} (env_id={episode_id}) --", flush=True)
        env, obs = build_env(task_cfg, episode_id, policy_name=args.model,
                             task_name=args.task)
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
                # Read every grasp key the env families use, not just Bridge's.
                # Reading only `is_src_obj_grasped` does not crash on Fractal --
                # it silently reports 0% grasp, and grasp-vs-success is the
                # split every failure diagnosis in this project turns on.
                grasped = grasped or step_grasped(info)
                grasp_seen = grasp_seen or grasp_is_reported(info)
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
        grasped = grasped or episode_grasped(final_info)
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
        # The embodiment conventions this run was scored under. Recorded, not
        # assumed: the same --model on the same checkpoint means a different
        # gripper convention and different action statistics per setup, and a
        # results file that does not say which one is not reproducible.
        "policy_setup": policy_setup_for(args.task),
        "unnorm_key": model.unnorm_key,
        "sticky_gripper_num_repeat": int(model.sticky_gripper_num_repeat),
        "n_episodes": len(results),
        "protocol_episodes": len(range(*task_cfg["obj_episode_range"])),
        "success_rate": success_count / len(results),
        # None, not 0.0: an env that never reports grasping has no grasp rate,
        # and writing 0% there reads as a total failure to grasp.
        "grasp_rate": (grasp_count / len(results)) if grasp_seen else None,
        "avg_steps": float(np.mean([item["steps"] for item in results])),
        "avg_elapsed": float(np.mean([item["elapsed"] for item in results])),
        # Which condition this run IS. paired_test.py reads these to label the
        # run and to refuse a directory whose per-task files were not all run
        # the same way, which is otherwise invisible once the logs scroll past.
        "action_repeat": (
            int(args.action_repeat) if args.model == "openvla_chunk" else 1
        ),
        # OpenVLA predicts exactly one action per forward, so the open-loop
        # horizon is action_repeat x 1. Recorded explicitly because the same
        # flag on a chunking backbone lands at a different horizon.
        "predict_action_frames": 1,
        "exec_chunk": 0,
        "foveate": {
            "enabled": args.model
            in ("openvla_foveated", "openvla_foveated_blur", "openvla_retina"),
            "mode": {
                "openvla_foveated": "logpolar",
                "openvla_foveated_blur": "blur",
                "openvla_retina": "logpolar+cache",
            }.get(args.model),
            "keep_percent": float(args.foveated_keep_percent),
        },
        "llm_prune_count": (
            int(args.depth_prune) if args.model == "openvla_depth" else 0
        ),
        "episodes": results,
    }
    if args.model == "openvla_depth":
        summary["depth"] = model.depth_summary()

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
    print(f"setup:        {summary['policy_setup']} (unnorm={summary['unnorm_key']}, "
          f"sticky={summary['sticky_gripper_num_repeat']})", flush=True)
    print(f"success_rate: {success_count}/{len(results)} = {summary['success_rate']:.1%}", flush=True)
    if summary["grasp_rate"] is None:
        print("grasp_rate:   n/a (this env does not report a grasp signal)", flush=True)
    else:
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
    if args.model == "openvla_depth":
        d = summary["depth"]
        print(
            f"depth: bypassed {d['n_bypassed']} of {d['n_layers']} layers "
            f"{d['bypassed_layers']}",
            flush=True,
        )
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
