#!/usr/bin/env python3
"""
spatialvla_eval.py

SpatialVLA + Latent Saccade SimplerEnv 평가 스크립트.

LatentSaccadeSpatialVLAInference 가 공식 SpatialVLAInference 를 상속하므로
ActionEnsembler, image history, do_normalize=False, cv2 resize, raw prompt 등
공식 파이프라인이 완벽히 동일하게 유지됩니다.
--no-latent-mask 플래그로 ON / OFF 를 동일 코드에서 대조 실험합니다.

사용법:
  # Latent Saccade ON
  python experiments/latent_saccade/spatialvla_eval.py \\
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \\
    --task widowx_put_eggplant_in_basket --n-episodes 24

  # Baseline (OFF)
  python experiments/latent_saccade/spatialvla_eval.py \\
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \\
    --task widowx_put_eggplant_in_basket --n-episodes 24 --no-latent-mask
"""

import sys
import os
import argparse
import json
import time

import numpy as np
from PIL import Image as PIL_Image

# ── SimplerEnv 경로 (환경에 맞게 수정) ────────────────────────────────────
SIMPLER_ENV_ROOT = os.environ.get("SIMPLER_ENV_ROOT", "/content/SimplerEnv")
if os.path.exists(SIMPLER_ENV_ROOT):
    sys.path.insert(0, SIMPLER_ENV_ROOT)
    sys.path.insert(0, os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim"))


# ── Task configs ───────────────────────────────────────────────────────────
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
        # robot init: from scripts/bridge.sh (widowx_sink_camera_setup scene)
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
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
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
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
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
        # robot init: from scripts/bridge.sh (widowx bridge_table_1_v1 scene)
        "robot_init_x": 0.147,
        "robot_init_y": 0.028,
    },

    # ── Google Robot / Fractal ─────────────────────────────────────────────
    # `env_name`/`env_kwargs` are the authors' own ENVIRONMENT_MAP entries, and
    # `prepackaged_config=True` (set in build_env) leaves robot, control mode,
    # control/sim freq, scene and the visual-matching overlay to the simulator
    # rather than restating them here. Restating them is how a Fractal run
    # silently ends up on the wrong scene or without an overlay, which is
    # exactly the failure that cost us a UniVLA campaign on Bridge.
    #
    # Only `obs_camera_name`, `max_episode_steps` and how an episode index maps
    # to an initial state are ours to choose.
    #
    # `variation` says where that mapping lives, because the three Google Robot
    # env families do it three different ways and only one of them accepts the
    # `episode_id` the Bridge tasks use:
    #
    #   "xy_grid"     GraspSingle (coke can) has no episode_id at all. The
    #                 reference eval sweeps a 5x5 grid of object init xy over
    #                 [-0.35,-0.12] x [-0.02,0.42], so ep_id indexes that grid.
    #   "episode_id"  MoveNear does accept episode_id; the reference sweeps
    #                 0..59 (source/target object triplet x xy config).
    #   "seed_only"   The drawer envs pick their station (9 overlays, each with
    #                 its own robot pose) from the episode RNG, so the seed is
    #                 the whole variation.
    #
    # In every case the seed is set to ep_id explicitly. Without it the URDF
    # variant -- and, on the drawers, the station -- is drawn from a per-env
    # RNG that we reseed every episode by rebuilding the env, so two conditions
    # would silently be scored on different initial states. That is the failure
    # mode a paired test cannot detect and cannot survive.
    "google_robot_pick_horizontal_coke_can": {
        "prepackaged": True,
        "env_name": "GraspSingleOpenedCokeCanInScene-v0",
        "env_kwargs": {"lr_switch": True},
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "xy_grid",
        "obj_init_xy_grid": {"x": (-0.35, -0.12, 5), "y": (-0.02, 0.42, 5)},
        "obj_episode_range": [0, 25],
    },
    "google_robot_pick_vertical_coke_can": {
        "prepackaged": True,
        "env_name": "GraspSingleOpenedCokeCanInScene-v0",
        "env_kwargs": {"laid_vertically": True},
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "xy_grid",
        "obj_init_xy_grid": {"x": (-0.35, -0.12, 5), "y": (-0.02, 0.42, 5)},
        "obj_episode_range": [0, 25],
    },
    "google_robot_pick_standing_coke_can": {
        "prepackaged": True,
        "env_name": "GraspSingleOpenedCokeCanInScene-v0",
        "env_kwargs": {"upright": True},
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "xy_grid",
        "obj_init_xy_grid": {"x": (-0.35, -0.12, 5), "y": (-0.02, 0.42, 5)},
        "obj_episode_range": [0, 25],
    },
    "google_robot_move_near": {
        "prepackaged": True,
        "env_name": "MoveNearGoogleBakedTexInScene-v1",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "episode_id",
        # The reference sweeps all 60. --n-episodes takes a prefix of this
        # range, so 24 episodes means ids 0..23 -- a subset of the protocol,
        # not a different one, but not comparable to a published 60-episode
        # number either.
        "obj_episode_range": [0, 60],
    },
    # Drawer tasks render with the ray-tracing shader and swap the overlay per
    # station, so they cost several times a coke-can episode. Kept out of the
    # default four; enable deliberately, with the extra wall-clock budgeted.
    "google_robot_open_drawer": {
        "prepackaged": True,
        "env_name": "OpenDrawerCustomInScene-v0",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 113,
        "variation": "seed_only",
        "obj_episode_range": [0, 24],
    },
    "google_robot_place_in_closed_drawer": {
        "prepackaged": True,
        "env_name": "PlaceIntoClosedDrawerCustomInScene-v0",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 200,
        "variation": "seed_only",
        "obj_episode_range": [0, 24],
    },
}


def prepackaged_reset_options(cfg, ep_id):
    """-> (seed, options) turning an episode index into one fixed initial state.

    Kept next to TASK_CONFIGS rather than inside build_env because the three
    eval harnesses all need the same mapping, and an episode index that means
    something different in two of them is not a paired experiment.
    """
    mode = cfg.get("variation", "seed_only")
    ep_id = int(ep_id)
    if mode == "episode_id":
        return ep_id, {"obj_init_options": {"episode_id": ep_id}}
    if mode == "xy_grid":
        import numpy as np
        g = cfg["obj_init_xy_grid"]
        xs, ys = np.linspace(*g["x"]), np.linspace(*g["y"])
        n = len(xs) * len(ys)
        if ep_id >= n:
            raise ValueError(
                f"episode {ep_id} is outside this task's {len(xs)}x{len(ys)} "
                f"object-placement grid ({n} distinct initial states). Asking "
                f"for more episodes than the protocol defines would re-run the "
                f"same states and inflate n without adding information."
            )
        x, y = xs[ep_id // len(ys)], ys[ep_id % len(ys)]
        return ep_id, {"obj_init_options": {"init_xy": [float(x), float(y)]}}
    if mode == "seed_only":
        return ep_id, {}
    raise ValueError(f"unknown variation mode {mode!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="HF hub 경로 또는 로컬 체크포인트 디렉토리")
    p.add_argument("--unnorm-key", default="bridge_orig/1.0.0",
                   help="action un-normalization 통계 키")
    p.add_argument("--policy-setup", default="widowx_bridge",
                   choices=["widowx_bridge", "google_robot"])
    p.add_argument("--task", default="widowx_put_eggplant_in_basket",
                   choices=list(TASK_CONFIGS.keys()))
    p.add_argument("--n-episodes", type=int, default=24)
    p.add_argument("--output-dir", default="./latent_saccade_spatialvla_results")
    # Saccade weights (fovea-only boost, grasp/place 분리)
    p.add_argument("--bg-weight",        type=float, default=1.0,
                   help="배경 weight. 1.0=억제 안 함. 낮추면 공간 계획 파괴")
    p.add_argument("--place-src-weight", type=float, default=1.1)
    p.add_argument("--grasp-fovea-weight", type=float, default=1.15,
                   help="grasp 단계 target fovea weight (약하게 — 파지 방해 최소화)")
    p.add_argument("--place-fovea-weight", type=float, default=1.3,
                   help="place 단계 target fovea weight (강하게 — placement 정밀도)")
    p.add_argument("--fovea-weight",     type=float, default=None,
                   help="주면 grasp/place 둘 다 이 값으로 덮어씀 (하위호환)")
    # Saccade timing
    p.add_argument("--min-grasp-steps",  type=int, default=10)
    p.add_argument("--max-grasp-steps",  type=int, default=60,
                   help="Force grasp→place after this many steps (0=disabled)")
    p.add_argument("--consec-close",     type=int, default=3)
    p.add_argument("--min-place-steps",  type=int, default=8)
    # DINO
    p.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--dino-cache-steps", type=int, default=5)
    p.add_argument("--box-threshold",   type=float, default=0.15)
    p.add_argument("--text-threshold",  type=float, default=0.15)
    p.add_argument("--dino-debug-dir",  default=None)
    # Toggle foveation
    p.add_argument("--enable-latent-mask", action="store_true", default=True)
    p.add_argument("--no-latent-mask",     dest="enable_latent_mask", action="store_false")
    # grasp 단계 foveation (SpatialVLA 는 기본 OFF — place 단계만 foveate)
    p.add_argument("--foveate-grasp", action="store_true", default=False,
                   help="grasp 단계에도 foveation 적용 (실험용, 기본 OFF)")
    p.add_argument("--no-foveate-grasp", dest="foveate_grasp", action="store_false")
    # place 전환 후 foveation 지연 (lift 확보 → 파지 마무리 방해 방지)
    p.add_argument("--place-foveation-delay", type=int, default=2,
                   help="place 전환 후 N 스텝 foveation 보류 (물체 lift 확보용)")
    # bbox area 필터 (전체화면 오탐 차단, 기본 활성)
    p.add_argument("--enable-area-filter", action="store_true", default=True,
                   help="bbox area 필터 활성화 (전체화면 오탐 차단)")
    p.add_argument("--no-area-filter", dest="enable_area_filter", action="store_false")
    p.add_argument("--grasp-max-area-ratio", type=float, default=0.5,
                   help="grasp 단계 bbox 최대 면적 비율. eggplant(sink cam)=0.5, bridge_table_1_v1=0.95")
    p.add_argument("--place-max-area-ratio", type=float, default=0.6,
                   help="place 단계 bbox 최대 면적 비율. eggplant(sink cam)=0.6, bridge_table_1_v1=0.95")
    # Video / OOD
    p.add_argument("--save-video",  action="store_true")
    p.add_argument("--no-overlay",  action="store_true",
                   help="OOD: rgb_overlay 제거")
    p.add_argument("--overlay-path", default=None,
                   help="OOD: 다른 overlay 이미지 경로")
    p.add_argument("--brightness",   type=float, default=1.0,
                   help="OOD: 이미지 밝기 스케일 (1.0=정상)")

    # ── ToMe (training-free SigLIP token merging) ──────────────────────────
    p.add_argument("--tome", action="store_true", default=False,
                   help="frozen SigLIP ViT 내부에서 ToMe 토큰 병합 (학습 없음, 끝에서 원래 토큰수로 복원)")
    p.add_argument("--tome-r", type=int, default=8,
                   help="레이어당 병합 토큰 수 (클수록 빠르고 근사 ↑)")
    p.add_argument("--tome-layers", type=int, default=6,
                   help="앞쪽 몇 개 encoder 레이어에서 병합할지")
    p.add_argument("--tome-protect", default="none",
                   choices=["none", "center"],
                   help="중요 영역 보호 prior (center=중앙 정사각형 고해상도 유지)")
    p.add_argument("--tome-protect-ratio", type=float, default=0.25,
                   help="center 보호 시 고해상도 유지 패치 비율")
    return p.parse_args()


def build_env(cfg, ep_id, no_overlay=False, overlay_path=None, task_name=None):
    # Google Robot / Fractal: let SimplerEnv build it. `prepackaged_config`
    # inside the env sets robot, control mode, freqs, scene and the
    # visual-matching overlay; re-deriving those here is how a run silently
    # loses its overlay and reports a collapse that is really a setup bug.
    if cfg.get("prepackaged"):
        import gymnasium as gym
        import simpler_env  # noqa: F401  -- imports mani_skill2_real2sim.envs, which registers the ids
        if task_name is None:
            raise ValueError("prepackaged tasks need task_name to pick an env id")
        # gym.make directly rather than simpler_env.make: older installs of
        # simpler_env define make(task_name) with no **kwargs, so passing
        # obs_mode/max_episode_steps through it raises TypeError. Newer ones
        # also write `env_kwargs["obs_mode"] = "rgbd",` -- the trailing comma
        # makes it the tuple ("rgbd",), which the env rejects. Going straight
        # to gym.make sidesteps both, and `prepackaged_config=True` still
        # leaves robot, control mode, freqs, scene and overlay to the env.
        try:
            env = gym.make(
                cfg["env_name"],
                obs_mode="rgbd",
                prepackaged_config=True,
                max_episode_steps=cfg["max_episode_steps"],
                **cfg.get("env_kwargs", {}),
            )
        except TypeError as e:
            # An old ManiSkill2_real2sim predates prepackaged_config. Falling
            # back to hand-set robot/scene/overlay would run, and would be the
            # wrong experiment; say what to fix instead.
            raise RuntimeError(
                f"{task_name}: {cfg['env_name']} rejected the prepackaged "
                f"visual-matching config ({e}). The SimplerEnv checkout at "
                f"{SIMPLER_ENV_ROOT} is older than the Google Robot eval "
                f"protocol. Update it rather than restating robot/scene/overlay "
                f"here -- getting those wrong is silent, not loud."
            ) from e
        seed, options = prepackaged_reset_options(cfg, ep_id)
        obs, _ = env.reset(seed=seed, options=options)
        # The overlay is not optional on these -- the checkpoint was evaluated
        # against the visual-matching image, and without it we would be scoring
        # a distribution the policy has never seen.
        inner = env.unwrapped
        if not getattr(inner, "rgb_overlay_path", None):
            raise RuntimeError(
                f"{task_name}: SimplerEnv returned no rgb_overlay_path. The "
                f"real_inpainting assets are missing under "
                f"{SIMPLER_ENV_ROOT}/ManiSkill2_real2sim/data. Refusing to run "
                f"rather than evaluate on the raw sim render."
            )
        return env, obs

    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    robot = cfg["robot"]
    try:
        control_mode = get_robot_control_mode(robot, "spatialvla")
    except Exception:
        control_mode = get_robot_control_mode(robot, "openvla")
    kw = dict(
        obs_mode="rgbd",
        robot=robot,
        sim_freq=cfg["sim_freq"],
        control_mode=control_mode,
        control_freq=cfg["control_freq"],
        max_episode_steps=cfg["max_episode_steps"],
        scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
    )
    if not no_overlay:
        cand = None
        if overlay_path and os.path.exists(overlay_path):
            cand = overlay_path
        else:
            for base in [SIMPLER_ENV_ROOT, os.path.join(SIMPLER_ENV_ROOT, "ManiSkill2_real2sim")]:
                p = os.path.join(base, cfg["rgb_overlay_path"])
                if os.path.exists(p):
                    cand = p
                    break
        if cand:
            kw["rgb_overlay_path"] = cand
            kw["rgb_overlay_cameras"] = cfg["rgb_overlay_cameras"]
    env = build_maniskill2_env(cfg["env_name"], **kw)
    reset_options = {"obj_init_options": {"episode_id": ep_id}}
    if "robot_init_x" in cfg:
        reset_options["robot_init_options"] = {
            "init_xy": np.array([cfg["robot_init_x"], cfg["robot_init_y"]]),
            "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    obs, _ = env.reset(options=reset_options)
    return env, obs


def step_grasped(info) -> bool:
    """Did the gripper hold the target object on THIS step?

    Bridge (put_on_in_scene) calls it `is_src_obj_grasped`; the Google Robot
    grasp/move-near envs call it `is_grasped` / `is_src_obj_grasped` depending
    on the family. Reading only the Bridge key does not crash on Fractal -- it
    silently reports a 0% grasp rate, and grasp-vs-success is the split that
    every failure diagnosis in this project has turned on.
    """
    if not isinstance(info, dict):
        return False
    return bool(info.get("is_src_obj_grasped") or info.get("is_grasped"))


def episode_grasped(final_info) -> bool:
    """Did the gripper ever hold it, from the env's own episode-level record."""
    if not isinstance(final_info, dict):
        return False
    if final_info.get("ever_grasped_src"):
        return True
    stats = final_info.get("episode_stats") or {}
    return bool(stats.get("grasped") or stats.get("consec_grasp"))


def get_image(env, obs, cam_name):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    return get_image_from_maniskill2_obs_dict(env, obs, camera_name=cam_name)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    from PIL import ImageEnhance
    pil = PIL_Image.fromarray(image)
    return np.array(ImageEnhance.Brightness(pil).enhance(factor))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    task_cfg = TASK_CONFIGS[args.task]
    cam_name = task_cfg["obs_camera_name"]

    # ── SpatialVLA path for latent_saccade module ──────────────────────────
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from experiments.latent_saccade.latent_saccade_spatialvla import LatentSaccadeSpatialVLAInference

    # ── Instantiate model (model loading happens inside __init__) ──────────
    # All official SpatialVLAInference params are passed through so the
    # pipeline is exactly equivalent to running the vanilla policy.
    policy = LatentSaccadeSpatialVLAInference(
        saved_model_path=args.model_path,
        unnorm_key=args.unnorm_key,
        policy_setup=args.policy_setup,
        # Latent Saccade params
        dino_model=args.dino_model,
        dino_cache_steps=args.dino_cache_steps,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        bg_weight=args.bg_weight,
        place_src_weight=args.place_src_weight,
        grasp_fovea_weight=args.grasp_fovea_weight,
        place_fovea_weight=args.place_fovea_weight,
        fovea_weight=args.fovea_weight,
        min_grasp_steps=args.min_grasp_steps,
        max_grasp_steps=args.max_grasp_steps,
        consecutive_close_required=args.consec_close,
        min_place_steps=args.min_place_steps,
        enable_latent_mask=args.enable_latent_mask,
        foveate_grasp=args.foveate_grasp,
        place_foveation_delay=args.place_foveation_delay,
        enable_area_filter=args.enable_area_filter,
        grasp_max_area_ratio=args.grasp_max_area_ratio,
        place_max_area_ratio=args.place_max_area_ratio,
        dino_debug_dir=args.dino_debug_dir,
    )
    print(
        f"[OK] num_patches={policy.num_patches}  "
        f"enable_latent_mask={args.enable_latent_mask}",
        flush=True,
    )

    # ── Optional: ToMe token merging on the frozen SigLIP ViT ──────────────
    if args.tome:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../tome"))
        from tome_siglip import apply_tome_to_siglip, center_protect_provider
        base_model = getattr(policy, "vla", None) or getattr(policy, "model", None)
        vision_tower = getattr(base_model, "vision_tower", None)
        if vision_tower is None:
            raise AttributeError(
                "Could not locate SpatialVLA vision_tower (tried policy.vla / policy.model). "
                "ToMe needs the SigLIP tower."
            )
        protect = None
        if args.tome_protect == "center":
            protect = center_protect_provider(args.tome_protect_ratio)
        apply_tome_to_siglip(
            vision_tower,
            r=args.tome_r,
            num_merge_layers=args.tome_layers,
            protect_provider=protect,
        )

    # ── Episode loop ───────────────────────────────────────────────────────
    base_ids = list(range(*task_cfg["obj_episode_range"]))
    ep_ids   = [base_ids[i % len(base_ids)] for i in range(args.n_episodes)]
    results  = []

    for ep_count, ep_id in enumerate(ep_ids):
        print(f"\n── ep {ep_count:02d} (env_id={ep_id}) ──────────────────────────", flush=True)
        env, obs    = build_env(task_cfg, ep_id, no_overlay=args.no_overlay, overlay_path=args.overlay_path, task_name=args.task)
        instruction = env.get_language_instruction()
        image       = get_image(env, obs, cam_name)
        image       = apply_brightness(image, args.brightness)
        print(f"   instruction: {instruction}", flush=True)

        # reset() takes task_description (matches official SpatialVLAInference API)
        policy.reset(instruction)

        frames    = [image.copy()] if args.save_video else []
        done      = truncated = False
        grasped   = False
        step      = 0
        t0        = time.time()

        while not (done or truncated) and step < task_cfg["max_episode_steps"]:
            # step() returns (raw_action, action) — identical to official policy
            raw_action, action = policy.step(image, instruction)

            # official maniskill2_evaluator concatenates the dict into a flat array
            env_action = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                np.atleast_1d(action["gripper"]),
            ])
            obs, _, done, truncated, info = env.step(env_action)
            image = apply_brightness(get_image(env, obs, cam_name), args.brightness)

            if not grasped and step_grasped(info):
                grasped = True
                print(f"[Grasp] env-reported grasp at step={step}", flush=True)

            if args.save_video and step % 4 == 0:
                frames.append(image.copy())

            # If environment changes the instruction (multi-stage tasks), sync policy
            new_instr = env.get_language_instruction()
            if new_instr != instruction:
                instruction = new_instr
                policy.reset(instruction)

            step += 1

        elapsed   = time.time() - t0
        grasp_str = "G+" if grasped else "G-"
        status    = "SUCCESS" if done else "FAIL"
        print(f"   → {grasp_str} {status}  ({step} steps, {elapsed:.1f}s)", flush=True)
        env.close()

        if args.save_video and frames:
            vpath = os.path.join(args.output_dir, f"ep{ep_count:02d}_{status.lower()}.gif")
            pils  = [PIL_Image.fromarray(f) for f in frames]
            pils[0].save(vpath, save_all=True, append_images=pils[1:], loop=0, duration=100)
            print(f"   GIF: {vpath}", flush=True)

        results.append({
            "ep": ep_count, "ep_id": ep_id,
            "grasped": grasped, "success": bool(done),
            "steps": step, "elapsed": elapsed,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    n_grasp = sum(r["grasped"] for r in results)
    n_ok    = sum(r["success"] for r in results)
    gr      = n_grasp / len(results)
    sr      = n_ok    / len(results)
    tag     = "ON" if args.enable_latent_mask else "OFF (baseline)"
    tome_tag = (f"ToMe r={args.tome_r}x{args.tome_layers} protect={args.tome_protect}"
                if args.tome else "ToMe OFF")
    total_steps = sum(r["steps"] for r in results)
    total_time  = sum(r["elapsed"] for r in results)
    ms_per_step = (total_time / total_steps * 1000.0) if total_steps else 0.0
    print(f"\n{'='*50}", flush=True)
    print(f"  model:     SpatialVLA + LatentSaccade [{tag}] | {tome_tag}", flush=True)
    print(f"  task:      {args.task}", flush=True)
    print(f"  파지율:    {n_grasp}/{len(results)} = {gr:.1%}", flush=True)
    print(f"  성공률:    {n_ok}/{len(results)} = {sr:.1%}", flush=True)
    print(f"  평균 스텝: {np.mean([r['steps'] for r in results]):.0f}", flush=True)
    print(f"  ms/step:   {ms_per_step:.1f}  (latency; lower=faster)", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        g_mark = "G+" if r["grasped"] else "G-"
        s_mark = "OK" if r["success"] else "--"
        print(f"  {s_mark}{g_mark} ep{r['ep']:02d} (id={r['ep_id']}): {r['steps']} steps", flush=True)

    summary = {
        "model": f"SpatialVLA+LatentSaccade[{tag}]",
        "task": args.task,
        "enable_latent_mask": args.enable_latent_mask,
        "ood_no_overlay": args.no_overlay,
        "ood_overlay_path": args.overlay_path,
        "ood_brightness": args.brightness,
        "grasp_rate": gr,
        "success_rate": sr,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "ms_per_step": float(ms_per_step),
        "tome": {
            "enabled": args.tome,
            "r": args.tome_r,
            "layers": args.tome_layers,
            "protect": args.tome_protect,
            "protect_ratio": args.tome_protect_ratio,
        },
        "config": {
            "grasp_fovea_weight": args.grasp_fovea_weight,
            "place_fovea_weight": args.place_fovea_weight,
            "fovea_weight_override": args.fovea_weight,
            "bg_weight": args.bg_weight,
            "place_src_weight": args.place_src_weight,
            "foveate_grasp": args.foveate_grasp,
            "place_foveation_delay": args.place_foveation_delay,
            "min_grasp_steps": args.min_grasp_steps,
            "consec_close": args.consec_close,
            "dino_cache_steps": args.dino_cache_steps,
            "unnorm_key": args.unnorm_key,
            "policy_setup": args.policy_setup,
        },
        "episodes": results,
    }
    save_path = os.path.join(args.output_dir, f"results_{args.task}.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n결과 저장: {save_path}", flush=True)


if __name__ == "__main__":
    main()
