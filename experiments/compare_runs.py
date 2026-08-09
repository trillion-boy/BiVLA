"""Compare two runs of the same condition episode by episode.

Written because comparing success *rates* is not a reproducibility check. The
UniVLA/Bridge log-polar cell reproduced its old pooled rate to the decimal
(86.5%, 83/96) while four of its per-task rates had moved by up to 8.3 points --
the aggregate agreed by coincidence. Only the per-episode outcome vector
distinguishes "the run reproduced" from "the run happened to land on the same
average".

    python experiments/compare_runs.py <reference-dir> <new-dir>

Each directory is a condition directory holding <task>/results_<task>.json.
Tasks present in only one of the two are reported and skipped.

The check is exact-match, not a statistical comparison, because that is what
determinism means: a single disagreeing episode falsifies it, and agreement
across a whole task is not something that happens by chance. One re-run
therefore settles the question -- there is no need to average over repeats.

Exit status is 0 when every shared episode agrees, 1 otherwise, so this can gate
a campaign script.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load(cond_dir: str) -> dict:
    """-> {task: {ep_id: (success, steps)}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(cond_dir, "*", "results_*.json"))):
        with open(path) as fh:
            summary = json.load(fh)
        task = summary.get("task") or os.path.basename(os.path.dirname(path))
        out[task] = {
            int(e["ep_id"]): (bool(e["success"]), e.get("steps"))
            for e in summary["episodes"]
        }
    return out


def main(ref_dir: str, new_dir: str) -> int:
    ref, new = load(ref_dir), load(new_dir)
    if not ref or not new:
        print(f"no results under {ref_dir if not ref else new_dir}")
        return 1

    for task in sorted(set(ref) ^ set(new)):
        print(f"  {task}: present in only one run -- skipped")

    all_agree = True
    for task in sorted(set(ref) & set(new)):
        ids = sorted(set(ref[task]) & set(new[task]))
        flips = [i for i in ids if ref[task][i][0] != new[task][i][0]]
        # Step counts can differ while the outcome does not; that still means
        # the trajectories diverged, so it is reported separately rather than
        # folded into the verdict.
        step_diffs = [i for i in ids
                      if ref[task][i][0] == new[task][i][0]
                      and ref[task][i][1] != new[task][i][1]]
        r = sum(ref[task][i][0] for i in ids)
        n = sum(new[task][i][0] for i in ids)
        verdict = "identical" if not flips else f"{len(flips)} episode(s) flipped"
        print(f"  {task:32s} n={len(ids):3d}  ref {r}/{len(ids)}  new {n}/{len(ids)}"
              f"  -> {verdict}")
        if flips:
            all_agree = False
            for i in flips:
                print(f"      ep {i:3d}: {ref[task][i][0]} -> {new[task][i][0]}")
        if step_diffs:
            print(f"      same outcome but different step count on "
                  f"{len(step_diffs)} episode(s): {step_diffs[:8]}")

    print("\nVERDICT:", "reproduced exactly" if all_agree
          else "NOT reproduced -- this run is not deterministic under these conditions")
    return 0 if all_agree else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
