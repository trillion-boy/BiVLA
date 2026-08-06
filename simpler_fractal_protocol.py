"""The Google Robot / Fractal eval protocol, shared by every harness.

One copy, imported. The protocol is a mapping from an episode index to one
fixed initial state, and the whole point of the campaign is that two backbones
-- or two conditions on one backbone -- are scored on the SAME states. A second
copy of this table in another harness would drift by one `max_episode_steps` or
one episode range and produce numbers that look comparable and are not. That is
the failure a paired test cannot detect, so the table does not get copied.

Deliberately dependency-free at import time (no torch, no simpler_env): the
harnesses that import it live in different subrepos with different conda envs,
and a heavy import here would make one of them refuse to load.
"""

import os


# ── The task table ─────────────────────────────────────────────────────────
GOOGLE_ROBOT_TASKS = {
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
    # Two MoveNear variants exist and they are not cosmetic: v0's triplets are
    # the `baked_*` meshes (textures baked for real-to-sim visual matching), v1's
    # are the plain ones. simpler_env.make() maps the task name to v1, but the
    # authors' reference eval script runs v0. Measured here, v1 scores 86.7%
    # against a published 69.6%, so which one a number came from has to be
    # recorded next to it. Both are registered; do not mix them in one table.
    "google_robot_move_near": {
        "prepackaged": True,
        "env_name": "MoveNearGoogleBakedTexInScene-v1",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "episode_id",
        # Run all 60. The ids are ordered by object triplet (id // 12), so a
        # prefix is a biased sample, not a smaller unbiased one: ids 0..23 cover
        # only the first two of five triplets and miss both that contain the
        # coke can beside the redbull can -- the only episodes where the policy
        # has to tell two look-alike cans apart from the instruction. Measured
        # at n=24 this task scored 91.7% against a published 69.6%.
        "obj_episode_range": [0, 60],
    },
    "google_robot_move_near_v0": {
        "prepackaged": True,
        "env_name": "MoveNearGoogleBakedTexInScene-v0",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 80,
        "variation": "episode_id",
        "obj_episode_range": [0, 60],
    },
    # Drawer tasks render with the ray-tracing shader and swap the overlay per
    # station, so they cost several times a coke-can episode. Enable
    # deliberately, with the extra wall-clock budgeted.
    #
    # The published "Open/Close Drawer" column is ONE number covering both
    # directions, so reporting only `open_drawer` against it would compare our
    # half to their whole. Both are registered; run both or neither.
    #
    # These use "seed_only": the env draws its station (9 overlays, each with
    # its own robot pose), its drawer (top/middle/bottom) and its URDF variant
    # from the episode RNG. That samples the reference protocol rather than
    # enumerating it -- the reference sweeps 4 URDFs x 9 stations x 6 env ids =
    # 216 episodes, which we are not paying for. Our number is therefore a
    # Monte-Carlo estimate of theirs, comparable in expectation and noisier;
    # it is not the same measurement and should not be tabled as if it were.
    "google_robot_open_drawer": {
        "prepackaged": True,
        "env_name": "OpenDrawerCustomInScene-v0",
        "obs_camera_name": "overhead_camera",
        "max_episode_steps": 113,
        "variation": "seed_only",
        "obj_episode_range": [0, 24],
    },
    "google_robot_close_drawer": {
        "prepackaged": True,
        "env_name": "CloseDrawerCustomInScene-v0",
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


def build_prepackaged_env(cfg, ep_id, task_name, simpler_env_root=""):
    """-> (env, obs) for a Google Robot task, reset to episode `ep_id`.

    `prepackaged_config=True` inside the env sets robot, control mode, freqs,
    scene and the visual-matching overlay; re-deriving those in a harness is how
    a run silently loses its overlay and reports a collapse that is really a
    setup bug.
    """
    import gymnasium as gym
    import simpler_env  # noqa: F401 -- imports mani_skill2_real2sim.envs, which registers the ids

    # gym.make directly rather than simpler_env.make: older installs of
    # simpler_env define make(task_name) with no **kwargs, so passing
    # obs_mode/max_episode_steps through it raises TypeError. Newer ones also
    # write `env_kwargs["obs_mode"] = "rgbd",` -- the trailing comma makes it
    # the tuple ("rgbd",), which the env rejects. Going straight to gym.make
    # sidesteps both, and `prepackaged_config=True` still leaves robot, control
    # mode, freqs, scene and overlay to the env.
    try:
        env = gym.make(
            cfg["env_name"],
            obs_mode="rgbd",
            prepackaged_config=True,
            max_episode_steps=cfg["max_episode_steps"],
            **cfg.get("env_kwargs", {}),
        )
    except TypeError as e:
        # An old ManiSkill2_real2sim predates prepackaged_config. Falling back
        # to hand-set robot/scene/overlay would run, and would be the wrong
        # experiment; say what to fix instead.
        raise RuntimeError(
            f"{task_name}: {cfg['env_name']} rejected the prepackaged "
            f"visual-matching config ({e}). The SimplerEnv checkout at "
            f"{simpler_env_root or '<SIMPLER_ENV_ROOT>'} is older than the "
            f"Google Robot eval protocol. Update it rather than restating "
            f"robot/scene/overlay here -- getting those wrong is silent, not loud."
        ) from e

    seed, options = prepackaged_reset_options(cfg, ep_id)
    obs, _ = env.reset(seed=seed, options=options)

    # The overlay is not optional on these -- the checkpoint was evaluated
    # against the visual-matching image, and without it we would be scoring a
    # distribution the policy has never seen.
    if not getattr(env.unwrapped, "rgb_overlay_path", None):
        raise RuntimeError(
            f"{task_name}: SimplerEnv returned no rgb_overlay_path. The "
            f"real_inpainting assets are missing under "
            f"{simpler_env_root or '<SIMPLER_ENV_ROOT>'}/ManiSkill2_real2sim/data. "
            f"Refusing to run rather than evaluate on the raw sim render."
        )
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


# Keys that carry a grasp signal, in either the per-step or the episode record.
# MoveNear reports none of them -- its success is "did the object move near the
# target", and a gripper never enters the criterion -- so a 0% grasp rate there
# is the env declining to answer, not the policy failing to grasp.
_GRASP_KEYS = ("is_src_obj_grasped", "is_grasped", "ever_grasped_src")


def grasp_is_reported(info) -> bool:
    """Does this env report a grasp signal at all?"""
    if not isinstance(info, dict):
        return False
    if any(k in info for k in _GRASP_KEYS):
        return True
    return any(k in (info.get("episode_stats") or {})
               for k in ("grasped", "consec_grasp"))


def episode_grasped(final_info) -> bool:
    """Did the gripper ever hold it, from the env's own episode-level record."""
    if not isinstance(final_info, dict):
        return False
    if final_info.get("ever_grasped_src"):
        return True
    stats = final_info.get("episode_stats") or {}
    return bool(stats.get("grasped") or stats.get("consec_grasp"))
