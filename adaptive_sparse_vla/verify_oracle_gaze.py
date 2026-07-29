"""Look at where the oracle gaze actually lands, before spending an hour on it.

`test_libero_oracle_gaze.py` proves the coordinate algebra is self-consistent,
but algebra cannot tell you whether the fovea ends up on the bowl — that needs
a real scene, and the answer is obvious to an eye and invisible to an
assertion. This renders one frame per task with a crosshair at the oracle
pixel, next to what the policy would actually receive.

No model, no GPU, about a minute:

    python verify_oracle_gaze.py --task-suite libero_spatial \\
        --out /content/oracle_gaze_check.png

Then open the PNG. Left column: the agent view with a crosshair on the
oracle's target. **The crosshair must sit on the bowl the instruction names.**
Right column: the same frame log-polar foveated there, i.e. what the policy
sees — the sharp region should cover that bowl.

If the crosshair is on the wrong bowl, on the table, or pinned to an image
edge, the oracle is measuring nothing and the eval would look normal while
being meaningless.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_spatial")
    p.add_argument("--task-ids", default="",
                   help="comma-separated; default = every task in the suite")
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--camera-resolution", type=int, default=256)
    p.add_argument("--foveate-keep-percent", type=float, default=20.0)
    p.add_argument("--foveate-mode", default="logpolar", choices=["logpolar", "blur"])
    p.add_argument("--mujoco-gl", default="osmesa",
                   choices=["egl", "glx", "osmesa", "glfw"])
    p.add_argument("--out", default="/content/oracle_gaze_check.png")
    return p.parse_args()


def draw_crosshair(img: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Magenta crosshair + ring. Drawn thick enough to survive downscaling in
    a notebook, and in a colour LIBERO's scenes never contain."""
    import cv2

    out = img.copy()
    c = (int(round(cx)), int(round(cy)))
    cv2.line(out, (c[0] - 14, c[1]), (c[0] + 14, c[1]), (255, 0, 255), 1)
    cv2.line(out, (c[0], c[1] - 14), (c[0], c[1] + 14), (255, 0, 255), 1)
    cv2.circle(out, c, 10, (255, 0, 255), 1)
    return out


def main() -> None:
    args = parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl
    # transformers' lazy `import tensorflow` segfaults once Mesa's GL is live;
    # eval_libero sets these too, but this script may import it first.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    import cv2

    # Import the harness rather than re-deriving anything: get_libero_image
    # carries the [::-1, ::-1] convention the whole oracle derivation depends
    # on, so a copy here could drift out of sync and hide the very bug this
    # script exists to catch.
    from eval_libero import build_env, ensure_libero_config, get_libero_image
    from foveation import foveate_image_blur, foveate_image_logpolar
    from libero_oracle_gaze import LiberoOracleGaze

    ensure_libero_config()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = (
        [int(x) for x in args.task_ids.split(",") if x.strip()]
        if args.task_ids.strip() else list(range(suite.n_tasks))
    )

    fov = foveate_image_logpolar if args.foveate_mode == "logpolar" else foveate_image_blur
    rows = []
    for tid in task_ids:
        task = suite.get_task(tid)
        env = build_env(task, resolution=args.camera_resolution)
        env.reset()
        init_states = suite.get_task_init_states(tid)
        obs = env.set_init_state(init_states[args.init_state % len(init_states)])

        gaze = LiberoOracleGaze(env, resolution=args.camera_resolution)
        image = get_libero_image(obs)
        center = gaze.update(image)

        if center is None:
            print(f"task {tid}: ORACLE FAILED -- {task.language}", flush=True)
            marked = image.copy()
            foveated = fov(image, args.foveate_keep_percent / 100.0, None)
        else:
            edge = (
                center[0] <= 1 or center[0] >= args.camera_resolution - 2
                or center[1] <= 1 or center[1] >= args.camera_resolution - 2
            )
            print(f"task {tid}: target={gaze.target_name()} "
                  f"fovea=({center[0]:.0f}, {center[1]:.0f})"
                  + ("  <-- PINNED TO EDGE, projection is out of view" if edge else "")
                  + f"  | {task.language}", flush=True)
            marked = draw_crosshair(image, *center)
            foveated = fov(image, args.foveate_keep_percent / 100.0, center)

        label = np.zeros((18, marked.shape[1] * 2, 3), dtype=np.uint8)
        cv2.putText(label, f"task {tid}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([label, np.hstack([marked, foveated])]))
        env.close()

    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\n[saved] {args.out}", flush=True)
    print("Left = agent view with the oracle crosshair; right = what the policy "
          "would see.\nThe crosshair must be ON the bowl the instruction names.",
          flush=True)


if __name__ == "__main__":
    main()
