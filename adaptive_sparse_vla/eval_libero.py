#!/usr/bin/env python3
"""
Evaluate the frozen UniVLA(-Emu3) LIBERO checkpoint on real LIBERO simulation
episodes, with the same foveation / chunk-exec test-time interventions used
for the Bridge/SimplerEnv backbone in eval.py.

This does NOT use RoboVLMs code or its training scaffolding. The only thing
borrowed from the UniVLA authors' own LIBERO eval script (baaivision/UniVLA:
reference/RoboVLMs/eval/libero/{evaluate_libero_emu.py,libero_utils.py}) is
the LIBERO env-building/observation convention -- 180-degree image rotation,
per-suite episode length, the num_steps_wait warm-up -- since that convention
belongs to how the checkpoint was trained, not to RoboVLMs itself.

Requires (see the setup script this ships with):
  pip install libero            # official LIBERO benchmark, PyPI package
  apt-get install libgl1-mesa-dri libegl1   # headless MuJoCo rendering
  huggingface_hub download of the LIBERO assets + UNIVLA_LIBERO_IMG_BS192_8K
  checkpoint (huggingface.co/Yuqi1997/UniVLA) -- both need real internet
  access, so this must run in an environment that can reach huggingface.co
  (e.g. Colab), not the sandboxed dev container this was authored in.
"""

import argparse
import faulthandler
import json
import os
import sys
import time

# The failure modes this harness kept hitting are native-level: hangs and
# silent deaths inside render/library initialization, with no Python
# traceback. enable() prints the Python stack on SIGSEGV/SIGABRT/SIGBUS at
# the moment of death. The stall watchdog below prints all thread stacks
# only when no progress has been made for a while -- it is re-armed on
# every model call, so a healthy run stays silent.
faulthandler.enable()

WATCHDOG_S = 300


def rearm_watchdog() -> None:
    faulthandler.dump_traceback_later(WATCHDOG_S, repeat=False)


rearm_watchdog()  # covers startup: env warmup, model load, first env build

# Keep TensorFlow out of this process entirely. transformers lazily does
# `import tensorflow` inside image_transforms when TF is installed (it is,
# on Colab), and TF's native module segfaults when loaded after Mesa's GL
# libraries (exit 139 with the crash stack pointing at TF's preload_check).
# USE_TF=0 makes transformers treat TF as absent; must be set before the
# first transformers import below.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

# Library deprecation chatter (HF resume_download, torch.jit, etc.) drowns
# the dozen lines of real progress output. Actionable warnings from our own
# code are unaffected.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import torch before any GL library gets loaded (the env warmup below pulls
# in Mesa, whose libLLVM can collide with torch's bundled LLVM when torch
# loads second). Importing torch does NOT initialize CUDA -- the GPU stays
# untouched until the model is moved there -- so this does not reintroduce
# the CUDA-vs-renderer ordering problem.
import torch  # noqa: E402,F401  (side effect: load torch's native libs first)

# MuJoCo needs an explicit headless backend; without one robosuite dies on
# `'NoneType' object has no attribute 'eglQueryString'` deep inside a render
# context. Setting it here means the eval works even when the notebook cell
# that exported it ran in a different process (or a restarted runtime).
#
# Prefer GLX whenever a display is available (i.e. Xvfb is running). EGL
# shares the GPU with CUDA and segfaults during env construction once a
# policy is resident there; an Xvfb + GLX setup was verified working with
# UniVLA on LIBERO while EGL was not, so honour a display when one exists.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glx" if os.environ.get("DISPLAY") else "egl"

EXP = os.path.dirname(__file__)
ROOT = os.environ.get(
    "UNIVLA_ROOT",
    os.path.abspath(os.path.join(EXP, "..", "UniVLA")),
)
EMU3 = os.path.join(ROOT, "reference", "Emu3")
for p in [ROOT, EXP, EMU3]:
    if p not in sys.path:
        sys.path.insert(0, p)

import transformers.processing_utils as _pu
import transformers.utils.import_utils as _tui

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: True
if not getattr(_pu.ProcessorMixin, "_check_patched", False):
    _pu.ProcessorMixin.check_argument_for_proper_class = lambda self, name, arg: None
    _pu.ProcessorMixin._check_patched = True

import numpy as np  # noqa: E402
from PIL import Image as _PIL  # noqa: E402

from foveation import MotionGaze, foveate_image_blur, foveate_image_logpolar  # noqa: E402
from libero_oracle_gaze import LiberoOracleGaze  # noqa: E402


def _patch_emu3_tokenizer() -> None:
    """Emu3-only setup, deferred so --backbone spatialvla never imports emu3.

    The vendored Emu3 stack pins itself to an older transformers generation;
    keeping it off the import path entirely means a SpatialVLA run is not
    hostage to that compatibility.
    """
    from emu3.mllm import Emu3Tokenizer

    if not hasattr(Emu3Tokenizer, "mergeable_ranks"):
        Emu3Tokenizer.mergeable_ranks = {}


# Longest training-demo episode length per suite, +margin -- matches UniVLA's
# own LIBERO eval convention (baaivision/UniVLA libero_utils.get_episode_length).
EPISODE_LENGTH = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10  # let dropped objects settle before the policy acts
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]  # no-op, gripper open


def save_gif(path: str, frames: list) -> None:
    pils = [_PIL.fromarray(frame) for frame in frames]
    pils[0].save(path, save_all=True, append_images=pils[1:], loop=0, duration=100)


def apply_foveation(image: np.ndarray, args, gaze) -> np.ndarray:
    # `gaze` is a MotionGaze or LiberoOracleGaze (same reset/update interface),
    # or None for the fixed image centre. A gaze that returns None -- an oracle
    # that could not resolve the scene -- also falls back to the centre.
    center = None
    if args.foveate_center in ("motion", "oracle") and gaze is not None:
        center = gaze.update(image)
    fov = foveate_image_logpolar if args.foveate_mode == "logpolar" else foveate_image_blur
    return fov(image, keep_ratio=args.foveate_keep_percent / 100.0, center=center)


def get_libero_image(obs: dict) -> np.ndarray:
    img = obs["agentview_image"]
    return img[::-1, ::-1]  # matches UniVLA's LIBERO training preprocessing


def get_libero_wrist_image(obs: dict) -> np.ndarray:
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1]


def build_env(task, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution
    )
    env.seed(0)
    return env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="univla",
                    choices=["univla", "spatialvla", "openvla"],
                    help="which VLA to evaluate; the LIBERO harness, foveation and "
                         "the efficiency interventions are identical for all")
    # --- univla backbone ---
    p.add_argument("--emu-hub", help="[univla] Emu3MoE LIBERO checkpoint dir")
    p.add_argument("--vq-hub",
                    help="[univla] base Emu3 dir holding the TEXT tokenizer "
                         "(BAAI/Emu3-Stage1), not the LIBERO checkpoint")
    p.add_argument("--vision-hub",
                    help="[univla] Emu3 vision tokenizer dir "
                         "(BAAI/Emu3-VisionTokenizer). Defaults to --vq-hub")
    p.add_argument("--fast-path",
                    help="[univla] FAST tokenizer dir bundled with the LIBERO "
                         "checkpoint (different from Bridge's fast_bridge_t5_s50)")
    # --- spatialvla / openvla backbones ---
    p.add_argument("--model-path",
                    help="[spatialvla|openvla] HF id or local dir of the checkpoint")
    p.add_argument("--unnorm-key",
                    help="[spatialvla|openvla] which dataset's action statistics to "
                         "de-normalize with. OpenVLA's LIBERO checkpoints ship "
                         "exactly one, which is picked automatically; pass a bogus "
                         "value to have the available keys printed")
    p.add_argument("--invert-gripper", action="store_true",
                    help="[spatialvla] flip the gripper sign if the checkpoint was "
                         "trained with the opposite open/close convention "
                         "(openvla already applies its documented inversion)")
    p.add_argument("--task-suite", default="libero_goal",
                    choices=list(EPISODE_LENGTH.keys()))
    p.add_argument("--task-ids", default="",
                    help="comma-separated task indices within the suite; empty = all tasks")
    p.add_argument("--n-trials-per-task", type=int, default=10)
    p.add_argument("--action-repeat", type=int, default=1,
                    help="execute each predicted action N times open-loop before "
                         "querying the model again. This is the chunk-exec analog "
                         "for single-step policies like OpenVLA, which have no "
                         "chunk to truncate: N=2 halves the forward passes")
    p.add_argument("--exec-chunk", type=int, default=0,
                    help="0 = execute the full predicted chunk (matches training); "
                         ">0 = execute only the first N of the predict_action_frames "
                         "predicted actions before calling the model again")
    p.add_argument("--foveate", action="store_true")
    p.add_argument("--foveate-mode", default="logpolar", choices=["logpolar", "blur"])
    p.add_argument("--foveate-center", default="image",
                    choices=["image", "motion", "oracle"],
                    help="where the fovea goes. 'image' = fixed centre (what "
                         "every result so far used). 'motion' = centroid of "
                         "frame differences, which tracks the arm rather than "
                         "the target and never reads the instruction. "
                         "'oracle' = the simulator's ground-truth pose of the "
                         "object the task moves -- privileged state, so a "
                         "diagnostic upper bound rather than a method: it "
                         "answers whether fovea PLACEMENT is why the same "
                         "foveation helps on SimplerEnv and destroys OpenVLA "
                         "here")
    p.add_argument(
        "--foveate-phase", default="always", choices=["always", "pregrasp"],
        help="pregrasp = only foveate while the policy's last commanded gripper "
             "action was OPEN; full resolution once it has commanded a close",
    )
    p.add_argument("--foveate-keep-percent", type=float, default=20.0)
    p.add_argument(
        "--foveate-views", default="agent", choices=["agent", "both"],
        help="which camera views foveation degrades. 'agent' (default) leaves "
             "the wrist view untouched, which hands a two-camera policy like "
             "UniVLA an unfoveated backup that a single-view policy like "
             "OpenVLA does not get; 'both' degrades every view the policy "
             "sees, which is the matched cross-architecture comparison",
    )
    p.add_argument("--depth-prune", type=int, default=0,
                    help="[univla] depth axis: bypass the N most redundant LLM "
                         "decoder layers (redundancy = 1 - cos(layer_in, "
                         "layer_out), calibrated once on a real prompt). Unlike "
                         "foveation this actually cuts wall-clock, because the "
                         "autoregressive action decode -- ~70%% of a step -- pays "
                         "for every layer on every token. Training-free")
    p.add_argument("--depth-ctrl", action="store_true",
                    help="[univla] depth axis, phase-adaptive: keep full depth "
                         "for the precise approach+grasp, then bypass more layers "
                         "once the policy commits to closing the gripper. "
                         "Overrides --depth-prune")
    p.add_argument("--depth-deep", type=int, default=2,
                    help="[univla] layers bypassed during approach+grasp")
    p.add_argument("--depth-shallow", type=int, default=8,
                    help="[univla] layers bypassed after the grasp")
    p.add_argument("--depth-close-steps", type=int, default=2,
                    help="[univla] consecutive close-gripper chunks before the "
                         "one-way deep->shallow switch (hysteresis)")
    p.add_argument("--depth-min-layer", type=float, default=0.5,
                    help="[univla] only layers past this fraction of the stack "
                         "are eligible; early layers carry too much to bypass")
    p.add_argument("--camera-resolution", type=int, default=256,
                    help="LIBERO renderer output size; the policy itself resizes "
                         "to 200x200 internally regardless of this value")
    p.add_argument("--mujoco-gl", default=None,
                    choices=["egl", "glx", "osmesa", "glfw"],
                    help="MuJoCo render backend. Default: respect $MUJOCO_GL, else "
                         "glx when $DISPLAY is set (Xvfb), else egl. Prefer glx "
                         "with Xvfb: egl shares the GPU with CUDA and segfaults "
                         "during env creation once a model is loaded. osmesa is "
                         "the CPU-rendering last resort")
    p.add_argument("--verbose", action="store_true",
                    help="chatty progress output: heartbeat every 5 model calls "
                         "and the [debug] action line on every trial (default: "
                         "heartbeat once a minute, [debug] once per run)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vision-device", default=None)
    p.add_argument("--min-pixels", type=int, default=6400)
    p.add_argument("--output-dir", default="/content/bivla_eval_libero")
    p.add_argument("--save-video", action="store_true")
    return p.parse_args()


_GL_CONTEXT = None  # module-level so the context is not garbage collected


def preinit_mujoco_gl() -> None:
    """Create the MuJoCo GL context *before* the policy touches CUDA.

    Building MuJoCo's EGL context after PyTorch has initialized CUDA on the
    same GPU segfaults: the process dies inside env construction with no
    Python traceback, which looks like a hang. Observed identically with
    both UniVLA and OpenVLA on an A100, while building the very same env in
    a fresh process (no model loaded) succeeded -- i.e. it is the ordering,
    not EGL itself, that is broken.

    Creating the context here, before any model is loaded or moved to the
    GPU, initializes EGL first and lets robosuite reuse the already-live
    display. Skipped for osmesa, which renders on the CPU and so has no
    contention to avoid. Failures are non-fatal: the run continues and the
    env build will report the real problem.
    """
    global _GL_CONTEXT
    backend = os.environ.get("MUJOCO_GL", "").lower()
    if backend in ("osmesa", "glx"):
        # osmesa renders on the CPU and glx goes through the X server
        # (Xvfb), so neither contends with CUDA for the GPU.
        print(f"[env] {backend} -- no GPU contention, skipping GL pre-init",
              flush=True)
        return
    try:
        import mujoco

        _GL_CONTEXT = mujoco.GLContext(64, 64)
        _GL_CONTEXT.make_current()
        print("[env] MuJoCo GL context pre-initialized before CUDA", flush=True)
    except Exception as e:
        print(f"[env] GL pre-init skipped ({type(e).__name__}: {e}); "
              f"if the run dies silently while building the env, "
              f"retry with --mujoco-gl osmesa", flush=True)


def ensure_libero_config() -> None:
    """Make sure ~/.libero/config.yaml exists and has every key LIBERO wants.

    LIBERO writes this file only when it is absent, and it does so from an
    interactive prompt that raises EOFError in a notebook -- so it normally
    gets pre-seeded by hand, and a half-written file then survives forever
    and fails later with `Key init_states not found in config file`. Rebuild
    any missing keys from the package layout. Deliberately avoids importing
    `libero.libero`, which is what triggers the prompt in the first place.
    """
    import importlib.util

    import yaml

    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    cfg_path = os.path.join(cfg_dir, "config.yaml")

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.origin:
        return  # libero not installed; the import in main() will say so
    root = os.path.join(os.path.dirname(spec.origin), "libero")
    defaults = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(root, "..", "datasets"),
        "assets": os.path.join(root, "assets"),
    }

    current = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                current = yaml.safe_load(f) or {}
        except Exception:
            current = {}  # unparseable (e.g. truncated mid-write) -> rebuild

    missing = [k for k in defaults if k not in current]
    if not missing:
        return
    merged = {**defaults, **current}
    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(merged, f)
    print(f"[libero] repaired {cfg_path}: added {missing}", flush=True)


def check_paths(args) -> None:
    """Fail fast, before the 16GB model load, if any checkpoint path is wrong.

    Without this, a missing local dir falls through to transformers trying to
    interpret the path string as a HuggingFace repo id, which dies much later
    with an opaque HFValidationError instead of saying "that folder isn't
    there".
    """
    if args.backbone in ("spatialvla", "openvla"):
        # A HF hub id is resolved by transformers, not by us; only validate
        # when it looks like a local path.
        if not args.model_path:
            sys.exit(f"[path error] --model-path is required for "
                     f"--backbone {args.backbone}")
        if os.sep in args.model_path and not os.path.isdir(args.model_path):
            sys.exit(f"[path error] --model-path directory does not exist: "
                     f"{args.model_path}")
        return

    missing_flags = [f for f, v in [("--emu-hub", args.emu_hub),
                                    ("--vq-hub", args.vq_hub),
                                    ("--fast-path", args.fast_path)] if not v]
    if missing_flags:
        sys.exit(f"[path error] --backbone univla requires {', '.join(missing_flags)}")

    checks = [
        ("--emu-hub", args.emu_hub, "config.json",
         "the UNIVLA_LIBERO_IMG_BS192_8K checkpoint (watch out for the nested "
         "subfolder snapshot_download creates)"),
        ("--vq-hub", args.vq_hub, "config.json",
         "the base Emu3 model holding the text tokenizer (snapshot_download "
         "of BAAI/Emu3-Stage1) -- NOT the LIBERO checkpoint folder"),
        ("--vision-hub", args.vision_hub or args.vq_hub, "preprocessor_config.json",
         "the frozen Emu3 vision tokenizer (snapshot_download of "
         "BAAI/Emu3-VisionTokenizer)"),
        ("--fast-path", args.fast_path, "processor_config.json",
         "the FAST action tokenizer dir (UniVLA/pretrain/fast_bridge_t5_s50 "
         "inside the BiVLA clone)"),
    ]
    problems = []
    for flag, path, marker, hint in checks:
        if not os.path.isdir(path):
            problems.append(f"  {flag}: directory does not exist: {path}\n"
                            f"      -> expected: {hint}")
        elif not os.path.exists(os.path.join(path, marker)):
            problems.append(f"  {flag}: {path} exists but has no {marker}\n"
                            f"      -> expected: {hint}\n"
                            f"      contents: {sorted(os.listdir(path))[:15]}")
    if problems:
        sys.exit("[path error] fix these before anything loads:\n" + "\n".join(problems))


def main():
    args = parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
        # mujoco.osmesa refuses to load unless PyOpenGL is routed to the
        # OSMesa platform. Colab pre-exports PYOPENGL_PLATFORM=egl, so this
        # must be a hard overwrite, not a setdefault.
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    print(f"[env] MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    check_paths(args)
    ensure_libero_config()
    preinit_mujoco_gl()  # must happen before the policy initializes CUDA
    os.makedirs(args.output_dir, exist_ok=True)

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    ep_len = EPISODE_LENGTH[args.task_suite]

    if args.task_ids.strip():
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = list(range(task_suite.n_tasks))

    # Build one throwaway env BEFORE the policy touches CUDA. Building this
    # exact env in a model-free process succeeds, while building it after
    # the model is resident hangs inside native render-context creation --
    # so run the whole GL/renderer initialization (dlopen, platform
    # selection, first context) now, while the GPU is still untouched.
    # Later env builds reuse the already-initialized platform.
    _t = time.time()
    print("[env] warmup: building first env before model load ...",
          end="", flush=True)
    warm_env = build_env(task_suite.get_task(task_ids[0]), resolution=128)
    warm_env.reset()
    warm_env.close()
    print(f" ok ({time.time() - _t:.1f}s)", flush=True)

    if args.backbone == "openvla":
        from inference_openvla_libero import OpenVLALiberoInference

        print(f"[load] LIBERO OpenVLA checkpoint from {args.model_path} ...",
              flush=True)
        model = OpenVLALiberoInference(
            model_path=args.model_path,
            unnorm_key=args.unnorm_key,
            device=args.device,
        )
        print(f"[openvla] unnorm_key={model.unnorm_key} "
              f"(single-step policy; use --action-repeat for the chunk-exec "
              f"analog)", flush=True)
    elif args.backbone == "spatialvla":
        from inference_spatialvla_libero import SpatialVLALiberoInference

        print(f"[load] LIBERO SpatialVLA checkpoint from {args.model_path} ...",
              flush=True)
        model = SpatialVLALiberoInference(
            model_path=args.model_path,
            unnorm_key=args.unnorm_key,
            device=args.device,
            invert_gripper=args.invert_gripper,
        )
        print(f"[spatialvla] unnorm_key={model.unnorm_key} "
              f"chunk={model.predict_action_frames}", flush=True)
    else:
        _patch_emu3_tokenizer()
        from inference_libero import EmuVLALiberoInference

        print(f"[load] LIBERO UniVLA checkpoint from {args.emu_hub} ...", flush=True)
        model = EmuVLALiberoInference(
            emu_hub=args.emu_hub,
            vq_hub=args.vq_hub,
            vision_hub=args.vision_hub or args.vq_hub,
            device=args.device,
            vision_device=args.vision_device,
            fast_path=args.fast_path,
            min_pixels_override=args.min_pixels,
            depth_prune=args.depth_prune,
            depth_ctrl=args.depth_ctrl,
            depth_deep=args.depth_deep,
            depth_shallow=args.depth_shallow,
            depth_close_steps=args.depth_close_steps,
            depth_min_layer=args.depth_min_layer,
        )
    print(
        f"[OK] model loaded  suite={args.task_suite}  tasks={task_ids}  "
        f"predict_action_frames={model.predict_action_frames}",
        flush=True,
    )
    if args.exec_chunk > 0:
        print(f"  exec-chunk: first {args.exec_chunk} of "
              f"{model.predict_action_frames} predicted actions", flush=True)
    if args.foveate:
        print(f"  foveate[{args.foveate_mode}/{args.foveate_center}/"
              f"{args.foveate_phase}] keep={args.foveate_keep_percent:.0f}% "
              f"views={args.foveate_views}", flush=True)
    if (args.depth_prune > 0 or args.depth_ctrl) and args.backbone != "univla":
        print(f"[warn] --depth-prune/--depth-ctrl are wired for the univla "
              f"backbone only; ignored for {args.backbone}", flush=True)

    results = []
    first_debug = True  # print the [debug] action-sanity line once per run
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)
        print(f"\n=== task {task_id}: {instruction} ===", flush=True)

        # One env per task, reset per trial -- same convention as OpenVLA's
        # official LIBERO eval. Rebuilding per trial multiplies the number
        # of native render-context creations (the fragile, hang-prone step)
        # by 10 for no benefit: set_init_state fully determines the episode.
        _te = time.time()
        print(f"   building env for task {task_id} ...", end="", flush=True)
        env = build_env(task, resolution=args.camera_resolution)
        print(f" ready ({time.time() - _te:.1f}s)", flush=True)

        for trial in range(args.n_trials_per_task):
            rearm_watchdog()
            env.reset()
            obs = env.set_init_state(init_states[trial % len(init_states)])
            model.reset()
            fov_gaze = None
            if args.foveate and args.foveate_center == "motion":
                fov_gaze = MotionGaze()
            elif args.foveate and args.foveate_center == "oracle":
                fov_gaze = LiberoOracleGaze(env, resolution=args.camera_resolution)
                if trial == 0:
                    # Log which object the oracle locked onto, once per task.
                    # libero_spatial puts several identical black bowls in the
                    # scene, so tracking the wrong one would look like a normal
                    # run while measuring nothing.
                    c = fov_gaze.update()
                    print(f"   [oracle-gaze] target={fov_gaze.target_name()} "
                          f"fovea={None if c is None else (round(c[0]), round(c[1]))} "
                          f"of {args.camera_resolution}px", flush=True)
            gripper_open = True
            frames = []
            done = False
            success = False
            step = 0
            t0 = time.time()
            last_beat = t0
            model_time = 0.0
            model_calls = 0

            try:
                while step < ep_len + NUM_STEPS_WAIT and not success:
                    if step < NUM_STEPS_WAIT:
                        # Settling steps: the task cannot be complete yet, so
                        # do not let a spurious done leak into the result.
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                        step += 1
                        continue

                    image = get_libero_image(obs)
                    wrist_image = get_libero_wrist_image(obs)
                    if args.save_video and step % 4 == 0:
                        frames.append(image)

                    policy_image = image
                    policy_wrist = wrist_image
                    if args.foveate and (args.foveate_phase == "always" or gripper_open):
                        policy_image = apply_foveation(image, args, fov_gaze)
                        if args.foveate_views == "both":
                            # Foveate the wrist view with its own centre (the
                            # gaze tracker follows the agent view, so reuse of
                            # its centre would be meaningless here).
                            policy_wrist = apply_foveation(wrist_image, args, None)

                    _t = time.time()
                    action_chunk = model.step(policy_image, instruction, wrist_image=policy_wrist)
                    model_time += time.time() - _t
                    model_calls += 1
                    rearm_watchdog()  # progress made; only a real stall dumps stacks
                    if args.verbose:
                        if model_calls == 1 or model_calls % 5 == 0:
                            print(f"      [heartbeat] call {model_calls}  env-step {step}  "
                                  f"last infer {time.time() - _t:.1f}s", flush=True)
                    elif time.time() - last_beat > 60:
                        last_beat = time.time()
                        print(f"      [heartbeat] call {model_calls}  env-step {step}  "
                              f"{model_time / model_calls * 1000:.0f} ms/infer avg",
                              flush=True)
                    if model_calls == 1 and (args.verbose or first_debug):
                        first_debug = False
                        print(f"      [debug] gen_len={getattr(model, 'last_generated_len', '?')} "
                              f"eoa={getattr(model, 'last_ended_with_eoa', 'n/a')} "
                              f"raw_ids={getattr(model, 'last_raw_ids', 'n/a')} "
                              f"chunk_shape={action_chunk.shape} "
                              f"dim_absmax={np.round(np.abs(action_chunk).max(axis=0), 3).tolist()} "
                              f"first_row={np.round(action_chunk[0], 3).tolist()}", flush=True)

                    if args.exec_chunk > 0:
                        action_chunk = action_chunk[: args.exec_chunk]
                    if args.action_repeat > 1:
                        action_chunk = np.repeat(action_chunk, args.action_repeat, axis=0)

                    for action_row in action_chunk:
                        gripper_open = float(action_row[-1]) <= 0.0
                        obs, _, done, _ = env.step(action_row.tolist())
                        step += 1
                        if done:
                            # Latch it: LIBERO's done means the goal predicate
                            # holds now, and the episode ends there. Without
                            # the latch a later step could flip it back and
                            # a solved episode would be scored as a failure.
                            success = True
                            break
                        if step >= ep_len + NUM_STEPS_WAIT:
                            break
            except Exception as e:
                print(f"   [exception during episode] {e}", flush=True)

            elapsed = time.time() - t0
            model_ms = (model_time / model_calls * 1000.0) if model_calls else 0.0
            status = "SUCCESS" if success else "FAIL"
            print(f"   trial {trial}: {status}  ({step} steps, {elapsed:.1f}s, "
                  f"{model_ms:.0f} ms/infer)", flush=True)

            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "trial": trial,
                "success": bool(success),
                "steps": step,
                "elapsed": elapsed,
                "model_ms_per_infer": model_ms,
            })
            if args.save_video and frames:
                vpath = os.path.join(
                    args.output_dir, f"task{task_id}_trial{trial}_{status.lower()}.gif"
                )
                save_gif(vpath, frames)

        env.close()

    faulthandler.cancel_dump_traceback_later()
    n_ok = sum(r["success"] for r in results)
    sr = n_ok / len(results) if results else 0.0
    summary = {
        "backbone": args.backbone,
        "checkpoint": (args.model_path if args.backbone in ("spatialvla", "openvla")
                       else args.emu_hub),
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "n_trials_per_task": args.n_trials_per_task,
        "exec_chunk": int(args.exec_chunk),
        "action_repeat": int(args.action_repeat),
        # UniVLA only: fraction of chunks the FAST tokenizer failed to decode.
        # A failure silently substitutes a fixed drift action, so a condition
        # that raises this rate is disturbing the policy in a way the success
        # rate alone does not show.
        "decode_failure_rate": (
            getattr(model, "decode_failures", 0) / getattr(model, "decode_calls", 0)
            if getattr(model, "decode_calls", 0) else None
        ),
        "predict_action_frames": int(model.predict_action_frames),
        "foveate": {
            "enabled": bool(args.foveate),
            "mode": args.foveate_mode,
            "center": args.foveate_center,
            "phase": args.foveate_phase,
            "keep_percent": float(args.foveate_keep_percent),
            "views": args.foveate_views,
        },
        # UniVLA only: which decoder layers ended up bypassed. Absent for
        # backbones the depth axis is not wired into.
        "depth": (model.depth_summary() if hasattr(model, "depth_summary") else None),
        "success_rate": sr,
        "n_episodes": len(results),
        "avg_steps": float(np.mean([r["steps"] for r in results])) if results else 0.0,
        "avg_model_ms_per_infer": (
            float(np.mean([r["model_ms_per_infer"] for r in results])) if results else 0.0
        ),
        "episodes": results,
    }
    out_path = os.path.join(
        args.output_dir, f"summary_{args.task_suite}_{int(time.time())}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if getattr(model, "decode_calls", 0):
        print(f"[decode] FAST failures: {model.decode_failures}/{model.decode_calls} "
              f"({model.decode_failures / model.decode_calls * 100:.1f}%)", flush=True)
    # Without this the controller is indistinguishable from --depth-prune
    # <depth_deep>: if the grasp signal never fires it silently never goes
    # shallow, and the success rate looks like a normal result.
    if args.depth_ctrl and summary["depth"]:
        d = summary["depth"]
        frac = d.get("shallow_fraction")
        print(f"[depth] episodes reaching shallow: "
              f"{d['episodes_reaching_shallow']}/{d['episodes']}"
              + (f" ({frac*100:.0f}%)" if frac is not None else ""), flush=True)
        if not d["episodes_reaching_shallow"]:
            print("[depth] WARNING: the controller never left the deep state, so "
                  f"this run is equivalent to --depth-prune {d['depth_deep']}",
                  flush=True)
    print(f"\n[SUMMARY] suite={args.task_suite}  success_rate={sr*100:.1f}% "
          f"({n_ok}/{len(results)})  avg_ms/infer={summary['avg_model_ms_per_infer']:.0f}",
          flush=True)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
