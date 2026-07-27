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
import json
import os
import sys
import time

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

from emu3.mllm import Emu3Tokenizer  # noqa: E402

if not hasattr(Emu3Tokenizer, "mergeable_ranks"):
    Emu3Tokenizer.mergeable_ranks = {}

import numpy as np  # noqa: E402
from PIL import Image as _PIL  # noqa: E402

from inference_libero import EmuVLALiberoInference  # noqa: E402
from foveation import MotionGaze, foveate_image_blur, foveate_image_logpolar  # noqa: E402


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


def apply_foveation(image: np.ndarray, args, gaze: "MotionGaze | None") -> np.ndarray:
    center = None
    if args.foveate_center == "motion" and gaze is not None:
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
    p.add_argument("--emu-hub", required=True)
    p.add_argument("--vq-hub", required=True)
    p.add_argument("--fast-path", required=True,
                    help="FAST tokenizer dir bundled with the LIBERO checkpoint "
                         "(different from Bridge's fast_bridge_t5_s50)")
    p.add_argument("--task-suite", default="libero_goal",
                    choices=list(EPISODE_LENGTH.keys()))
    p.add_argument("--task-ids", default="",
                    help="comma-separated task indices within the suite; empty = all tasks")
    p.add_argument("--n-trials-per-task", type=int, default=10)
    p.add_argument("--exec-chunk", type=int, default=0,
                    help="0 = execute the full predicted chunk (matches training); "
                         ">0 = execute only the first N of the predict_action_frames "
                         "predicted actions before calling the model again")
    p.add_argument("--foveate", action="store_true")
    p.add_argument("--foveate-mode", default="logpolar", choices=["logpolar", "blur"])
    p.add_argument("--foveate-center", default="image", choices=["image", "motion"])
    p.add_argument(
        "--foveate-phase", default="always", choices=["always", "pregrasp"],
        help="pregrasp = only foveate while the policy's last commanded gripper "
             "action was OPEN; full resolution once it has commanded a close",
    )
    p.add_argument("--foveate-keep-percent", type=float, default=20.0)
    p.add_argument("--camera-resolution", type=int, default=256,
                    help="LIBERO renderer output size; the policy itself resizes "
                         "to 200x200 internally regardless of this value")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vision-device", default=None)
    p.add_argument("--min-pixels", type=int, default=6400)
    p.add_argument("--output-dir", default="/content/bivla_eval_libero")
    p.add_argument("--save-video", action="store_true")
    return p.parse_args()


def check_paths(args) -> None:
    """Fail fast, before the 16GB model load, if any checkpoint path is wrong.

    Without this, a missing local dir falls through to transformers trying to
    interpret the path string as a HuggingFace repo id, which dies much later
    with an opaque HFValidationError instead of saying "that folder isn't
    there".
    """
    checks = [
        ("--emu-hub", args.emu_hub, "config.json",
         "the UNIVLA_LIBERO_IMG_BS192_8K checkpoint (watch out for the nested "
         "subfolder snapshot_download creates)"),
        ("--vq-hub", args.vq_hub, "preprocessor_config.json",
         "the frozen Emu3 vision tokenizer (snapshot_download of "
         "BAAI/Emu3-VisionTokenizer) -- NOT the LIBERO checkpoint folder"),
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
    check_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    ep_len = EPISODE_LENGTH[args.task_suite]

    if args.task_ids.strip():
        task_ids = [int(x.strip()) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = list(range(task_suite.n_tasks))

    print(f"[load] LIBERO UniVLA checkpoint from {args.emu_hub} ...", flush=True)
    model = EmuVLALiberoInference(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vq_hub,
        device=args.device,
        vision_device=args.vision_device,
        fast_path=args.fast_path,
        min_pixels_override=args.min_pixels,
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
              f"{args.foveate_phase}] keep={args.foveate_keep_percent:.0f}%", flush=True)

    results = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)
        print(f"\n=== task {task_id}: {instruction} ===", flush=True)

        for trial in range(args.n_trials_per_task):
            env = build_env(task, resolution=args.camera_resolution)
            env.reset()
            obs = env.set_init_state(init_states[trial % len(init_states)])
            model.reset()
            fov_gaze = (
                MotionGaze() if args.foveate and args.foveate_center == "motion" else None
            )
            gripper_open = True
            frames = []
            done = False
            step = 0
            t0 = time.time()
            model_time = 0.0
            model_calls = 0

            try:
                while step < ep_len + NUM_STEPS_WAIT:
                    if step < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        step += 1
                        continue

                    image = get_libero_image(obs)
                    wrist_image = get_libero_wrist_image(obs)
                    if args.save_video and step % 4 == 0:
                        frames.append(image)

                    policy_image = image
                    if args.foveate and (args.foveate_phase == "always" or gripper_open):
                        policy_image = apply_foveation(image, args, fov_gaze)

                    _t = time.time()
                    action_chunk = model.step(policy_image, instruction, wrist_image=wrist_image)
                    model_time += time.time() - _t
                    model_calls += 1
                    if model_calls == 1 or model_calls % 5 == 0:
                        print(f"      [heartbeat] call {model_calls}  env-step {step}  "
                              f"last infer {time.time() - _t:.1f}s", flush=True)

                    if args.exec_chunk > 0:
                        action_chunk = action_chunk[: args.exec_chunk]

                    for action_row in action_chunk:
                        gripper_open = float(action_row[-1]) <= 0.0
                        obs, _, done, _ = env.step(action_row.tolist())
                        step += 1
                        if done or step >= ep_len + NUM_STEPS_WAIT:
                            break
            except Exception as e:
                print(f"   [exception during episode] {e}", flush=True)

            elapsed = time.time() - t0
            model_ms = (model_time / model_calls * 1000.0) if model_calls else 0.0
            status = "SUCCESS" if done else "FAIL"
            print(f"   trial {trial}: {status}  ({step} steps, {elapsed:.1f}s, "
                  f"{model_ms:.0f} ms/infer)", flush=True)

            results.append({
                "task_id": task_id,
                "instruction": instruction,
                "trial": trial,
                "success": bool(done),
                "steps": step,
                "elapsed": elapsed,
                "model_ms_per_infer": model_ms,
            })
            env.close()

            if args.save_video and frames:
                vpath = os.path.join(
                    args.output_dir, f"task{task_id}_trial{trial}_{status.lower()}.gif"
                )
                save_gif(vpath, frames)

    n_ok = sum(r["success"] for r in results)
    sr = n_ok / len(results) if results else 0.0
    summary = {
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "n_trials_per_task": args.n_trials_per_task,
        "exec_chunk": int(args.exec_chunk),
        "predict_action_frames": int(model.predict_action_frames),
        "foveate": {
            "enabled": bool(args.foveate),
            "mode": args.foveate_mode,
            "center": args.foveate_center,
            "phase": args.foveate_phase,
            "keep_percent": float(args.foveate_keep_percent),
        },
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

    print(f"\n[SUMMARY] suite={args.task_suite}  success_rate={sr*100:.1f}% "
          f"({n_ok}/{len(results)})  avg_ms/infer={summary['avg_model_ms_per_infer']:.0f}",
          flush=True)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
