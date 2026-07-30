"""Slice the oracle-gaze verification grid into report-sized figures.

`verify_oracle_gaze.py` stacks one row per task -- an 18px label strip above
`[agent view with crosshair | what the policy sees]` -- into a single tall PNG.
That is the right shape for checking all 10 tasks at once and the wrong shape
for a document: at 10 rows it renders as a thin ribbon nobody can read.

    python split_oracle_gaze.py oracle_gaze_check.png \\
        --groups 0-2,3-5,6-9 --out-dir ../experiments/figures

Needs only Pillow -- no GPU, no simulator, no LIBERO install. It re-slices a
PNG that already exists.

Row height is derived from the image rather than hardcoded, because the label
strip and the render resolution have both changed before. The script refuses
to run if the height does not divide evenly by the task count, since a
fractional row means the assumption is wrong and slicing anyway would emit
figures cut through the middle of a task.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image


def parse_groups(spec: str, n_tasks: int) -> list[list[int]]:
    groups = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = (int(x) for x in chunk.split("-", 1))
            ids = list(range(lo, hi + 1))
        else:
            ids = [int(chunk)]
        for i in ids:
            if not 0 <= i < n_tasks:
                raise SystemExit(f"task {i} is outside 0..{n_tasks - 1}")
        groups.append(ids)

    seen = [i for g in groups for i in g]
    if len(seen) != len(set(seen)):
        raise SystemExit("a task appears in more than one group")
    missing = sorted(set(range(n_tasks)) - set(seen))
    if missing:
        print(f"[note] tasks not included in any group: {missing}")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--groups", default="0-2,3-5,6-9")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--prefix", default="oracle_gaze")
    args = ap.parse_args()

    img = Image.open(args.image)
    width, height = img.size
    if height % args.n_tasks:
        raise SystemExit(
            f"{height}px does not divide into {args.n_tasks} equal rows "
            f"({height / args.n_tasks:.2f}px each). Either the grid holds a "
            f"different number of tasks or its layout changed -- pass "
            f"--n-tasks rather than slicing on a wrong assumption."
        )
    row = height // args.n_tasks
    os.makedirs(args.out_dir, exist_ok=True)

    for ids in parse_groups(args.groups, args.n_tasks):
        lo, hi = min(ids), max(ids)
        if ids != list(range(lo, hi + 1)):
            raise SystemExit(f"group {ids} is not contiguous; rows are stacked "
                             f"in task order and cannot be reordered by cropping")
        crop = img.crop((0, lo * row, width, (hi + 1) * row))
        name = f"{args.prefix}_tasks{lo}-{hi}.png"
        path = os.path.join(args.out_dir, name)
        crop.save(path)
        print(f"[saved] {path}  ({width}x{crop.size[1]}, tasks {lo}-{hi})")


if __name__ == "__main__":
    main()
