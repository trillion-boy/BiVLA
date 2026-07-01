#!/usr/bin/env python3
"""
Evaluate frozen VLA policies on the local manipulation benchmark.

Active model types:
- `baseline`: fixed-resolution frozen VLA
- `adaptive_sparse`: adaptive sparse visual gate
- `task_aware_hybrid`: baseline-by-default task router with selective sparse carrot intervention
"""

import argparse
import json
import os
import sys
import time
import types

ROOT = os.environ.get(
    "UNIVLA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "UniVLA")),
)
EXP = os.path.dirname(__file__)
EMU3 = os.path.join(ROOT, "reference", "Emu3")
SIMPLER = os.environ.get(
    "SIMPLER_ENV_PATH",
    os.path.abspath(os.path.join(ROOT, "../SimplerEnv")),
)
MANSKILL = os.path.join(SIMPLER, "ManiSkill2_real2sim")

for p in [ROOT, EXP, EMU3, SIMPLER, MANSKILL]:
    if p not in sys.path:
        sys.path.insert(0, p)

import transformers.processing_utils as _pu
import transformers.utils.import_utils as _tui

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: True

if not getattr(_pu.ProcessorMixin, "_check_patched", False):
    _pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    _pu.ProcessorMixin._check_patched = True

if "lightning" not in sys.modules:
    for _n in ["lightning", "lightning.pytorch", "lightning.pytorch.trainer"]:
        sys.modules.setdefault(_n, types.ModuleType(_n))

    class _Trainer:
        pass

    sys.modules["lightning.pytorch.trainer"].Trainer = _Trainer

from emu3.mllm import Emu3Tokenizer

if not hasattr(Emu3Tokenizer, "mergeable_ranks"):
    Emu3Tokenizer.mergeable_ranks = {}

import numpy as np
from PIL import Image as _PIL

from inference import (
    AdaptiveSparseInference,
    EmuVLAInference,
    SharedCompactFocusRouterInference,
    TaskAwareEpisodeRouterInference,
    TaskAwareHybridInference,
    UniformControlSparseInference,
    UniformGeometrySparseInference,
    UniformRefinementInference,
)


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
    },
}


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    if left.shape != right.shape:
        raise ValueError(
            f"Expected matching image shapes, got {left.shape} and {right.shape}"
        )
    h, _, c = left.shape
    divider = np.full((h, 4, c), 255, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def save_gif(path: str, frames: list[np.ndarray]) -> None:
    pils = [_PIL.fromarray(frame) for frame in frames]
    pils[0].save(path, save_all=True, append_images=pils[1:], loop=0, duration=100)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emu-hub", required=True)
    p.add_argument("--vq-hub", required=True)
    p.add_argument("--fast-path", required=True)
    p.add_argument("--task", default="widowx_put_eggplant_in_basket")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=0,
                   help="if >0, profile encoding/prefill/decode for N steps and exit")
    p.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated environment episode ids to evaluate in order.",
    )
    p.add_argument("--output-dir", default="/content/bivla_eval")
    p.add_argument(
        "--model-type",
        choices=["baseline", "adaptive_sparse", "task_aware_hybrid", "shared_compact_focus", "uniform_refine", "uniform_geometry_sparse", "uniform_control_sparse"],
        default="baseline",
    )
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--min-pixels", type=int, default=6400)
    p.add_argument("--vision-device", default="cuda")
    p.add_argument("--device", default="cuda")
    p.add_argument("--disable-video-history", action="store_true")
    p.add_argument("--llm-prune-count", type=int, default=0)
    p.add_argument("--llm-prune-min-layer", type=float, default=0.5)
    p.add_argument("--llm-prune-min-gap", type=int, default=1)
    p.add_argument(
        "--llm-prune-layers",
        default="",
        help="Optional comma-separated explicit decoder layers to bypass instead of importance-based selection.",
    )
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--patch-grid-size", type=int, default=20)
    p.add_argument("--max-highres-patches", type=int, default=80)
    p.add_argument("--lowres-factor", type=float, default=0.1)
    p.add_argument(
        "--patch-grasp-steps",
        type=int,
        default=12,
        help="Use the sparse visual view for this many early acquisition steps before switching to the raw view.",
    )
    p.add_argument("--selector-hub", default="nvidia/AutoGaze")
    p.add_argument("--selector-target-patch-size", type=int, default=16)
    p.add_argument("--selector-task-loss-requirement", type=float, default=-1.0)
    p.add_argument("--selector-device", default="cpu")
    p.add_argument(
        "--selector-refresh-stride",
        type=int,
        default=1,
        help="Reuse the previous sparse patch mask for this many patch-mode steps before recomputing it.",
    )
    p.add_argument(
        "--sparse-gate-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="Control sparse visual gating for ablations.",
    )
    p.add_argument("--hybrid-carrot-dx-ratio", type=float, default=0.22)
    p.add_argument("--hybrid-carrot-dy-ratio", type=float, default=0.04)
    p.add_argument("--hybrid-carrot-area-min", type=float, default=0.010)
    p.add_argument("--hybrid-carrot-area-max", type=float, default=0.0135)
    p.add_argument("--hybrid-carrot-peak-max", type=float, default=0.75)
    p.add_argument("--uniform-gate-dx-ratio", type=float, default=0.22)
    p.add_argument("--uniform-gate-dy-ratio", type=float, default=0.04)
    p.add_argument("--uniform-gate-area-min", type=float, default=0.010)
    p.add_argument("--uniform-gate-area-max", type=float, default=0.0135)
    p.add_argument("--uniform-gate-peak-max", type=float, default=0.75)
    p.add_argument("--focus-gate-min-confidence", type=float, default=0.65)
    p.add_argument("--focus-gate-max-confidence", type=float, default=0.78)
    p.add_argument("--focus-gate-min-context-confidence", type=float, default=0.55)
    p.add_argument("--close-hold-steps", type=int, default=3)
    p.add_argument("--refine-candidates", type=int, default=2)
    p.add_argument("--refine-beams", type=int, default=2)
    p.add_argument("--refine-max-steps", type=int, default=6)
    p.add_argument("--uncertainty-margin-threshold", type=float, default=0.40)
    p.add_argument("--uncertainty-action-gap-threshold", type=float, default=0.020)
    p.add_argument("--rerank-temporal-weight", type=float, default=0.35)
    p.add_argument("--rerank-smoothness-weight", type=float, default=0.15)
    return p.parse_args()


def profile_latency(model, cfg, cam_name, n_steps):
    """Split one UniVLA control step into VQ-encoding / LLM-prefill / LLM-decode,
    to test whether the decode dominates (as it does on SpatialVLA). Wraps
    image_tokenizer.encode (encoding) and the Emu3 decoder forward (prefill when
    q_len>1, decode when q_len==1)."""
    import time
    import torch
    cuda = torch.cuda.is_available()
    def sync():
        if cuda:
            torch.cuda.synchronize()
    enc, prefill, decode = [], [], []

    tok = model.image_tokenizer
    orig_enc = tok.encode
    def timed_enc(*a, **k):
        sync(); t = time.time(); out = orig_enc(*a, **k); sync(); enc.append(time.time() - t)
        return out
    tok.encode = timed_enc

    emu = model.model.model  # Emu3Model (decoder stack)
    orig_fwd = emu.forward
    def timed_fwd(*a, **k):
        ii = k.get("input_ids", None); ie = k.get("inputs_embeds", None)
        q = (ie.shape[1] if ie is not None else (ii.shape[1] if ii is not None
             else (a[0].shape[1] if a and hasattr(a[0], "shape") else 1)))
        sync(); t = time.time(); out = orig_fwd(*a, **k); sync(); dt = time.time() - t
        (prefill if q > 1 else decode).append(dt)
        return out
    emu.forward = timed_fwd

    env, obs = build_env(cfg, 0)
    instruction = env.get_language_instruction()
    image = get_image(env, obs, cam_name)
    model.reset()
    print(f"profiling '{instruction}' for up to {n_steps} steps ...", flush=True)
    done = truncated = False
    step = 0
    step_times = []
    while not (done or truncated) and step < n_steps:
        sync(); ts = time.time()
        _, env_actions = model.step(image, instruction)
        sync(); step_times.append(time.time() - ts)
        for env_a in env_actions:
            obs, _, done, truncated, info = env.step(
                np.concatenate([env_a["world_vector"], env_a["rot_axangle"], env_a["gripper"]]))
            image = get_image(env, obs, cam_name)
            step += 1
            if done or truncated:
                break
    env.close()

    n = len(step_times)
    enc_ms = sum(enc) / n * 1000; pre_ms = sum(prefill) / n * 1000
    dec_ms = sum(decode) / n * 1000; step_ms = sum(step_times) / n * 1000
    other = step_ms - enc_ms - pre_ms - dec_ms
    dtoks = len(decode) / n
    print(f"\n{'='*58}")
    print(f"  UniVLA per-step latency breakdown (avg over {n} steps)")
    print(f"{'='*58}")
    print(f"  total step        : {step_ms:7.1f} ms  (100%)")
    print(f"  ├─ encoding (VQ)  : {enc_ms:7.1f} ms  ({enc_ms/step_ms:5.1%})")
    print(f"  ├─ LLM prefill    : {pre_ms:7.1f} ms  ({pre_ms/step_ms:5.1%})")
    print(f"  ├─ LLM decode     : {dec_ms:7.1f} ms  ({dec_ms/step_ms:5.1%})  [{dtoks:.0f} tok]")
    print(f"  └─ other          : {other:7.1f} ms  ({other/step_ms:5.1%})")
    print(f"{'='*58}")
    print(f"  PREFILL-SIDE ceiling (enc+prefill) = {(enc_ms+pre_ms)/step_ms:.1%}")
    print(f"  decode = {dec_ms/step_ms:.0%} (only LLM-depth shrinks this)")
    print(f"{'='*58}", flush=True)


def build_env(cfg, ep_id):
    from simpler_env.utils.env.env_builder import (
        build_maniskill2_env,
        get_robot_control_mode,
    )

    robot = cfg["robot"]
    kw = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=cfg["sim_freq"],
        control_mode=get_robot_control_mode(robot, "emu_vla"),
        control_freq=cfg["control_freq"],
        max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
        renderer_kwargs={"offscreen_only": True},
    )
    for base in [SIMPLER, os.path.join(SIMPLER, "ManiSkill2_real2sim")]:
        cand = os.path.join(base, cfg["rgb_overlay_path"])
        if os.path.exists(cand):
            kw["rgb_overlay_path"] = cand
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
            break
    env = build_maniskill2_env(cfg["env_name"], **kw)
    obs, _ = env.reset(options={"obj_init_options": {"episode_id": ep_id}})
    return env, obs


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def build_model(args):
    common_kwargs = dict(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vq_hub,
        device=args.device,
        vision_device=args.vision_device,
        policy_setup="widowx_bridge",
        fast_path=args.fast_path,
        image_size_override=(args.image_size, args.image_size),
        min_pixels_override=args.min_pixels,
        llm_prune_count=args.llm_prune_count,
        llm_prune_min_layer=args.llm_prune_min_layer,
        llm_prune_min_gap=args.llm_prune_min_gap,
        llm_prune_layers=tuple(
            int(x.strip()) for x in args.llm_prune_layers.split(",") if x.strip()
        ) if args.llm_prune_layers.strip() else None,
    )
    if args.model_type == "baseline":
        model = EmuVLAInference(**common_kwargs)
        model.video_mode = not args.disable_video_history
        return model
    shared_sparse_kwargs = dict(
        **common_kwargs,
        patch_grid_size=args.patch_grid_size,
        max_highres_patches=args.max_highres_patches,
        lowres_factor=args.lowres_factor,
        patch_grasp_steps=args.patch_grasp_steps,
        autogaze_hub=args.selector_hub,
        autogaze_target_patch_size=args.selector_target_patch_size,
        autogaze_task_loss_requirement=(
            None if args.selector_task_loss_requirement < 0 else args.selector_task_loss_requirement
        ),
        selector_device=args.selector_device or None,
        selector_refresh_stride=args.selector_refresh_stride,
        sparse_gate_mode=args.sparse_gate_mode,
    )
    if args.model_type == "task_aware_hybrid":
        model = TaskAwareEpisodeRouterInference(
            **shared_sparse_kwargs,
            carrot_dx_ratio=args.hybrid_carrot_dx_ratio,
            carrot_dy_ratio=args.hybrid_carrot_dy_ratio,
            carrot_area_min=args.hybrid_carrot_area_min,
            carrot_area_max=args.hybrid_carrot_area_max,
            carrot_peak_max=args.hybrid_carrot_peak_max,
        )
    elif args.model_type == "shared_compact_focus":
        model = SharedCompactFocusRouterInference(
            **shared_sparse_kwargs,
            gate_dx_ratio=args.uniform_gate_dx_ratio,
            gate_dy_ratio=args.uniform_gate_dy_ratio,
            gate_area_min=args.uniform_gate_area_min,
            gate_area_max=args.uniform_gate_area_max,
            gate_peak_max=args.focus_gate_max_confidence,
            gate_peak_min=args.focus_gate_min_confidence,
            gate_context_min=args.focus_gate_min_context_confidence,
        )
    elif args.model_type == "uniform_geometry_sparse":
        model = UniformGeometrySparseInference(
            **shared_sparse_kwargs,
            gate_dx_ratio=args.uniform_gate_dx_ratio,
            gate_dy_ratio=args.uniform_gate_dy_ratio,
            gate_area_min=args.uniform_gate_area_min,
            gate_area_max=args.uniform_gate_area_max,
            gate_peak_max=args.uniform_gate_peak_max,
        )
    elif args.model_type == "uniform_control_sparse":
        model = UniformControlSparseInference(
            **shared_sparse_kwargs,
            gate_dx_ratio=args.uniform_gate_dx_ratio,
            gate_dy_ratio=args.uniform_gate_dy_ratio,
            gate_area_min=args.uniform_gate_area_min,
            gate_area_max=args.uniform_gate_area_max,
            gate_peak_max=args.uniform_gate_peak_max,
            close_hold_steps=args.close_hold_steps,
        )
    elif args.model_type == "uniform_refine":
        model = UniformRefinementInference(
            **shared_sparse_kwargs,
            refine_candidates=args.refine_candidates,
            refine_beams=args.refine_beams,
            refine_max_steps=args.refine_max_steps,
            uncertainty_margin_threshold=args.uncertainty_margin_threshold,
            uncertainty_action_gap_threshold=args.uncertainty_action_gap_threshold,
            rerank_temporal_weight=args.rerank_temporal_weight,
            rerank_smoothness_weight=args.rerank_smoothness_weight,
        )
    else:
        model = AdaptiveSparseInference(
            **shared_sparse_kwargs,
        )
    model.video_mode = not args.disable_video_history
    return model


def main():
    args = parse_args()

    # Env overrides for quick latency sweeps without editing the run script.
    # LLM_PRUNE_COUNT bypasses N redundant decoder layers (cuts attention AND FFN,
    # the real latency lever on UniVLA's FFN-bound decode). LLM_PRUNE_MIN_LAYER /
    # LLM_PRUNE_MIN_GAP tune which layers are eligible.
    if os.environ.get("LLM_PRUNE_COUNT"):
        args.llm_prune_count = int(os.environ["LLM_PRUNE_COUNT"])
    if os.environ.get("LLM_PRUNE_MIN_LAYER"):
        args.llm_prune_min_layer = float(os.environ["LLM_PRUNE_MIN_LAYER"])
    if os.environ.get("LLM_PRUNE_MIN_GAP"):
        args.llm_prune_min_gap = int(os.environ["LLM_PRUNE_MIN_GAP"])

    os.makedirs(args.output_dir, exist_ok=True)

    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]

    print(f"[load] model_type={args.model_type} ...", flush=True)
    model = build_model(args)
    print(
        f"[OK] model loaded  image_size={args.image_size}  min_pixels={args.min_pixels}",
        flush=True,
    )

    if getattr(args, "profile_steps", 0) and args.profile_steps > 0:
        profile_latency(model, task_cfg, cam_name, args.profile_steps)
        return

    if args.episode_ids.strip():
        ep_ids = [int(x.strip()) for x in args.episode_ids.split(",") if x.strip()]
    else:
        base_ids = list(range(*task_cfg["obj_episode_range"]))
        ep_ids = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────", flush=True)
        env, obs = build_env(task_cfg, ep_id)
        instruction = env.get_language_instruction()
        image = get_image(env, obs, cam_name)
        print(f"   instruction: {instruction}", flush=True)

        if hasattr(model, "route_episode"):
            model.route_episode(image, instruction)
        model.reset()
        frames = []
        policy_frames = []
        comparison_frames = []
        done = truncated = False
        final_info = {}
        phase_info = {}
        step = 0
        t0 = time.time()
        model_time = 0.0
        model_calls = 0

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            raw_frame = image.copy()
            _t_model = time.time()
            _, env_actions = model.step(
                image,
                instruction,
                phase_info=phase_info,
                oracle_context={
                    "obs": obs,
                    "camera_name": cam_name,
                    "task": args.task,
                },
            )
            model_time += time.time() - _t_model
            model_calls += 1
            if args.save_video and step % 4 == 0:
                frames.append(raw_frame)
                prepared_frame = model.last_prepared_image()
                if prepared_frame is not None and args.model_type != "baseline":
                    policy_frames.append(prepared_frame)
                    comparison_frames.append(make_side_by_side(raw_frame, prepared_frame))
            for env_a in env_actions:
                obs, _, done, truncated, info = env.step(
                    np.concatenate(
                        [env_a["world_vector"], env_a["rot_axangle"], env_a["gripper"]]
                    )
                )
                final_info = info
                phase_info = info
                image = get_image(env, obs, cam_name)
                new_instr = env.get_language_instruction()
                if new_instr != instruction:
                    instruction = new_instr
                    if hasattr(model, "route_episode"):
                        model.route_episode(image, instruction)
                    model.reset()
                step += 1
                if done or truncated:
                    break
        final_success = bool(final_info.get("success", False))
        elapsed = time.time() - t0
        model_ms = (model_time / model_calls * 1000.0) if model_calls else 0.0
        status = "SUCCESS" if final_success else "FAIL"
        print(f"   → {status}  ({step} steps, {elapsed:.1f}s, "
              f"{model_ms:.0f} ms/infer)", flush=True)

        episode_result = {
            "ep": ep_count,
            "ep_id": ep_id,
            "success": final_success,
            "terminated": bool(done),
            "truncated": bool(truncated),
            "steps": step,
            "elapsed": elapsed,
            "model_ms_per_infer": model_ms,
            "final_info": final_info,
        }
        if args.model_type != "baseline":
            episode_result.update(model.selection_summary())
        if hasattr(model, "pruning_summary"):
            episode_result.update(model.pruning_summary())
        results.append(episode_result)
        env.close()

        if args.save_video and frames:
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            save_gif(vpath, frames)
            print(f"   GIF: {vpath}", flush=True)
            if args.model_type != "baseline" and policy_frames:
                ppath = os.path.join(
                    args.output_dir,
                    f"ep{ep_count:02d}_{status.lower()}_policy_view.gif",
                )
                cpath = os.path.join(
                    args.output_dir,
                    f"ep{ep_count:02d}_{status.lower()}_comparison.gif",
                )
                save_gif(ppath, policy_frames)
                save_gif(cpath, comparison_frames)
                print(f"   Policy GIF: {ppath}", flush=True)
                print(f"   Compare GIF: {cpath}", flush=True)

    n_ok = sum(r["success"] for r in results)
    sr = n_ok / len(results)
    summary = {
        "model_type": args.model_type,
        "task": args.task,
        "image_size": args.image_size,
        "min_pixels": args.min_pixels,
        "vision_device": args.vision_device,
        "llm_prune_count": args.llm_prune_count,
        "llm_prune_min_layer": args.llm_prune_min_layer,
        "llm_prune_min_gap": args.llm_prune_min_gap,
        "llm_prune_layers": [
            int(x.strip()) for x in args.llm_prune_layers.split(",") if x.strip()
        ],
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_elapsed": float(np.mean([r["elapsed"] for r in results])),
        "avg_model_ms_per_infer": float(
            np.mean([r["model_ms_per_infer"] for r in results])
        ),
        "fastv": {
            "enabled": os.environ.get("FASTV_ENABLE", "0") == "1",
            "k": int(os.environ.get("FASTV_K", "3")),
            "keep_ratio": float(os.environ.get("FASTV_KEEP_RATIO", "0.4")),
        },
        "depth_ctrl": {
            "enabled": os.environ.get("DEPTH_CTRL_ENABLE", "0") == "1",
            "deep": int(os.environ.get("DEPTH_CTRL_DEEP", "2")),
            "shallow": int(os.environ.get("DEPTH_CTRL_SHALLOW", "8")),
            "close_steps": int(os.environ.get("DEPTH_CTRL_CLOSE_STEPS", "2")),
            "archetypes": os.environ.get("DEPTH_CTRL_ARCHETYPES", "") or "all",
        },
        "episodes": results,
    }
    if args.model_type in {"adaptive_sparse", "task_aware_hybrid", "shared_compact_focus", "uniform_refine", "uniform_geometry_sparse", "uniform_control_sparse"}:
        summary["patch_selector"] = {
            "grid_size": args.patch_grid_size,
            "max_highres_patches": args.max_highres_patches,
            "lowres_factor": args.lowres_factor,
            "patch_grasp_steps": args.patch_grasp_steps,
            "selector_hub": args.selector_hub,
            "selector_target_patch_size": args.selector_target_patch_size,
            "selector_task_loss_requirement": (
                None if args.selector_task_loss_requirement < 0 else args.selector_task_loss_requirement
            ),
            "selector_refresh_stride": args.selector_refresh_stride,
            "sparse_gate_mode": args.sparse_gate_mode,
            "avg_selected_patch_count": float(
                np.mean([r["avg_selected_patch_count"] for r in results])
            ),
            "avg_selection_ratio": float(
                np.mean([r["avg_selection_ratio"] for r in results])
            ),
            "patch_usage_ratio": float(
                np.mean([r.get("patch_usage_ratio", 0.0) for r in results])
            ),
            "selector_refresh_ratio": float(
                np.mean([r.get("selector_refresh_ratio", 0.0) for r in results])
            ),
        }
        if args.model_type == "task_aware_hybrid":
            summary["task_router"] = {
                "carrot_dx_ratio": args.hybrid_carrot_dx_ratio,
                "carrot_dy_ratio": args.hybrid_carrot_dy_ratio,
                "carrot_area_min": args.hybrid_carrot_area_min,
                "carrot_area_max": args.hybrid_carrot_area_max,
                "carrot_peak_max": args.hybrid_carrot_peak_max,
                "router_profiles": sorted(
                    {r.get("router_profile", "unresolved") for r in results}
                ),
                "carrot_adaptive_episode_ratio": float(
                    np.mean(
                        [
                            1.0 if r.get("router_profile") == "carrot_adaptive" else 0.0
                            for r in results
                        ]
                    )
                ),
            }
        if args.model_type == "shared_compact_focus":
            summary["compact_focus_router"] = {
                "gate_min_confidence": args.focus_gate_min_confidence,
                "gate_max_confidence": args.focus_gate_max_confidence,
                "gate_min_context_confidence": args.focus_gate_min_context_confidence,
                "gate_dx_ratio": args.uniform_gate_dx_ratio,
                "gate_dy_ratio": args.uniform_gate_dy_ratio,
                "gate_area_min": args.uniform_gate_area_min,
                "gate_area_max": args.uniform_gate_area_max,
                "router_profiles": sorted(
                    {r.get("router_profile", "unresolved") for r in results}
                ),
                "adaptive_episode_ratio": float(
                    np.mean(
                        [
                            1.0 if r.get("router_profile") == "compact_focus_adaptive" else 0.0
                            for r in results
                        ]
                    )
                ),
                "focus_enabled_ratio": float(
                    np.mean([1.0 if r.get("focus_enabled") else 0.0 for r in results])
                ),
            }
        if args.model_type == "uniform_refine":
            summary["uniform_refine"] = {
                "refine_candidates": args.refine_candidates,
                "refine_beams": args.refine_beams,
                "refine_max_steps": args.refine_max_steps,
                "uncertainty_margin_threshold": args.uncertainty_margin_threshold,
                "uncertainty_action_gap_threshold": args.uncertainty_action_gap_threshold,
                "rerank_temporal_weight": args.rerank_temporal_weight,
                "rerank_smoothness_weight": args.rerank_smoothness_weight,
                "avg_score_margin": float(
                    np.mean([r.get("avg_score_margin", 0.0) for r in results])
                ),
                "avg_first_action_gap": float(
                    np.mean([r.get("avg_first_action_gap", 0.0) for r in results])
                ),
                "uncertainty_trigger_ratio": float(
                    np.mean([r.get("uncertainty_trigger_ratio", 0.0) for r in results])
                ),
                "refine_patch_ratio": float(
                    np.mean([r.get("refine_patch_ratio", 0.0) for r in results])
                ),
            }
        if args.model_type == "uniform_geometry_sparse":
            summary["uniform_geometry_sparse"] = {
                "gate_dx_ratio": args.uniform_gate_dx_ratio,
                "gate_dy_ratio": args.uniform_gate_dy_ratio,
                "gate_area_min": args.uniform_gate_area_min,
                "gate_area_max": args.uniform_gate_area_max,
                "gate_peak_max": args.uniform_gate_peak_max,
                "uniform_gate_ratio": float(
                    np.mean([r.get("uniform_gate_ratio", 0.0) for r in results])
                ),
                "avg_gate_dx_ratio": float(
                    np.mean([r.get("avg_gate_dx_ratio", 0.0) for r in results])
                ),
                "avg_gate_dy_ratio": float(
                    np.mean([r.get("avg_gate_dy_ratio", 0.0) for r in results])
                ),
                "avg_gate_source_area": float(
                    np.mean([r.get("avg_gate_source_area", 0.0) for r in results])
                ),
            }
        if args.model_type == "uniform_control_sparse":
            summary["uniform_control_sparse"] = {
                "close_hold_steps": args.close_hold_steps,
                "gate_dx_ratio": args.uniform_gate_dx_ratio,
                "gate_dy_ratio": args.uniform_gate_dy_ratio,
                "gate_area_min": args.uniform_gate_area_min,
                "gate_area_max": args.uniform_gate_area_max,
                "gate_peak_max": args.uniform_gate_peak_max,
                "uniform_gate_ratio": float(
                    np.mean([r.get("uniform_gate_ratio", 0.0) for r in results])
                ),
            }
    if any("llm_prune_count" in r for r in results):
        summary["llm_pruning"] = {
            "avg_active_prune_count": float(
                np.mean([r.get("llm_prune_count", 0) for r in results])
            ),
            "active_layers": results[0].get("llm_pruned_layers", []),
        }

    print(f"\n{'=' * 50}", flush=True)
    print(f"  task: {args.task}", flush=True)
    print(f"  model: {args.model_type}", flush=True)
    print(f"  success_rate: {n_ok}/{len(results)} = {sr:.1%}", flush=True)
    print(f"  avg_steps: {summary['avg_steps']:.0f}", flush=True)
    print(f"  avg_elapsed: {summary['avg_elapsed']:.1f}s", flush=True)
    print(f"  avg_model_ms_per_infer: {summary['avg_model_ms_per_infer']:.0f} ms  "
          f"(pure inference latency; step-count independent)", flush=True)
    if summary["fastv"]["enabled"]:
        print(f"  fastv: K={summary['fastv']['k']} keep={summary['fastv']['keep_ratio']}", flush=True)
    if summary["depth_ctrl"]["enabled"]:
        dc = summary["depth_ctrl"]
        print(f"  depth_ctrl: deep(prune {dc['deep']}) -> shallow(prune {dc['shallow']}) "
              f"after {dc['close_steps']} close steps", flush=True)
    if args.model_type in {"adaptive_sparse", "task_aware_hybrid", "shared_compact_focus", "uniform_refine", "uniform_geometry_sparse", "uniform_control_sparse"}:
        print(
            f"  avg_selected_patches: {summary['patch_selector']['avg_selected_patch_count']:.1f}",
            flush=True,
        )
        print(
            f"  avg_selection_ratio: {summary['patch_selector']['avg_selection_ratio']:.4f}",
            flush=True,
        )
        print(
            f"  patch_usage_ratio: {summary['patch_selector']['patch_usage_ratio']:.2f}",
            flush=True,
        )
        print(
            f"  selector_refresh_ratio: {summary['patch_selector']['selector_refresh_ratio']:.2f}",
            flush=True,
        )
    if args.model_type == "uniform_refine":
        print(
            f"  uncertainty_trigger_ratio: {summary['uniform_refine']['uncertainty_trigger_ratio']:.2f}",
            flush=True,
        )
        print(
            f"  refine_patch_ratio: {summary['uniform_refine']['refine_patch_ratio']:.2f}",
            flush=True,
        )
    if args.model_type == "uniform_geometry_sparse":
        print(
            f"  uniform_gate_ratio: {summary['uniform_geometry_sparse']['uniform_gate_ratio']:.2f}",
            flush=True,
        )
    if args.model_type == "uniform_control_sparse":
        print(
            f"  uniform_gate_ratio: {summary['uniform_control_sparse']['uniform_gate_ratio']:.2f}",
            flush=True,
        )
    if args.model_type == "task_aware_hybrid":
        print(
            f"  carrot_adaptive_episode_ratio: {summary['task_router']['carrot_adaptive_episode_ratio']:.2f}",
            flush=True,
        )
        print(
            f"  router_profiles: {summary['task_router']['router_profiles']}",
            flush=True,
        )
    if args.model_type == "shared_compact_focus":
        print(
            f"  compact_focus_episode_ratio: {summary['compact_focus_router']['adaptive_episode_ratio']:.2f}",
            flush=True,
        )
        print(
            f"  focus_enabled_ratio: {summary['compact_focus_router']['focus_enabled_ratio']:.2f}",
            flush=True,
        )
        print(
            f"  router_profiles: {summary['compact_focus_router']['router_profiles']}",
            flush=True,
        )
    if "llm_pruning" in summary:
        print(
            f"  llm_pruned_layers: {summary['llm_pruning']['active_layers']}",
            flush=True,
        )
    print(f"{'=' * 50}", flush=True)

    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results: {save_path}", flush=True)


if __name__ == "__main__":
    main()
