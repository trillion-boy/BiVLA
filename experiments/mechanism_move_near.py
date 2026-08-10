"""Classify move_near failures by *what went wrong*, not just whether it worked.

Why this exists
---------------
Section 6 of the report argues that depth pruning damages referential grounding
rather than motor control. The evidence so far is a *between-task* comparison:
`google_robot_move_near_v0` collapses under pruning while the three coke-can
tasks barely move. That comparison is confounded -- the tasks differ in
instruction complexity, object count, episode length, and success criterion all
at once, so "grounding" is only one of several stories that fit.

`move_near_in_scene.py` already computes, at every step, enough state to tell
those stories apart *inside a single task*:

    moved_correct_obj = source moved >3cm and moved more than every other object
    moved_wrong_obj   = some other object moved >3cm and moved more than source
    near_tgt_obj      = source ended within a bbox-scaled radius of the target
    is_closest_to_tgt = source is nearer the target than any other object
    all_obj_keep_height = nothing fell off the table

and returns them under `info["episode_stats"]`. The OpenVLA harness stores the
whole terminal `info`, so every episode we have already run on Fractal carries
this. No new simulation is required to read it.

The taxonomy
------------
Each non-success episode lands in exactly one bucket, checked in this order:

  dropped        not all_obj_keep_height -- something ended up off the table
  wrong_object   moved_wrong_obj -- the arm worked, on the object we did not name
  misplaced      moved_correct_obj but not (near_tgt_obj and is_closest_to_tgt)
                 -- right object picked up, wrong destination
  no_contact     nothing moved more than 3cm -- never got a useful grip at all

The two that matter are `wrong_object` and `no_contact`. They make opposite
predictions:

  * If pruning degrades *referential grounding*, failures should shift toward
    `wrong_object`: the policy still reaches and grasps competently, it just
    acts on the wrong referent.
  * If pruning degrades *motor control*, failures should shift toward
    `no_contact`: the policy knows what to reach for and cannot execute.

Because both are measured within `move_near`, the task-difference confound that
weakens the between-task argument does not apply here.

What this script does *not* establish
-------------------------------------
The buckets are terminal-state descriptions, not causal readouts. `wrong_object`
is consistent with a grounding failure; it does not prove the language pathway
is where the damage sits. And `episode_stats` in this environment is overwritten
each step, so these are the values at the final step, not "ever during the
episode" -- an episode that briefly moved the wrong object and then recovered
reads as whatever was true at the end.

The sensitivity of `wrong_object` is also bounded by how the environment defines
it: some other object must move more than 3cm *and* move further than the source
did. A policy that reaches toward the wrong object and fails to shift it lands in
`no_contact`, not `wrong_object`. So a low `wrong_object` count rules out
"grounded on the wrong object and successfully manipulated it"; it does not rule
out mis-grounding on its own.

Usage
-----
    python experiments/mechanism_move_near.py                # all OpenVLA/Fractal conditions
    python experiments/mechanism_move_near.py <dir> [<dir>]  # explicit run dirs

Baseline is whichever directory is listed first; every later one is compared to
it on the shared episode ids.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

BUCKETS = ("success", "dropped", "wrong_object", "misplaced", "no_contact")

# Conditions in the order they should appear, with the baseline first. Names are
# the directory suffixes under results/.
DEFAULT_CAMPAIGN = "results/openvla_fractal_0806_{}"
DEFAULT_CONDITIONS = [
    "baseline",
    "depth_prune1",
    "depth_prune2",
    "depth_prune4",
    "depth_prune4_early",
    "foveate",
    "foveate_blur",
    "action_repeat2",
    "action_repeat4",
]


def truthy(v) -> bool:
    """The environment stringifies some of these booleans and not others.

    `all_obj_keep_height`, `moved_correct_obj` and `near_tgt_obj` arrive as
    "True"/"False" strings while `moved_wrong_obj` and `is_closest_to_tgt`
    arrive as real bools, because they are written at different points in the
    episode. bool("False") is True, so this cannot be left to the caller.
    """
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def outcome_stats(rec: dict) -> dict:
    """The two harnesses record the same flags in two different shapes.

    OpenVLA and UniVLA store the terminal `info` wholesale, so the flags sit
    under rec["final_info"]["episode_stats"]. SpatialVLA stores a flat record
    and writes the flags as `env_<key>` at the top level. Accept both, so a
    SpatialVLA run does not get silently skipped as "no outcome detail".
    """
    info = rec.get("final_info") or {}
    stats = info.get("episode_stats") or info
    if "moved_correct_obj" in stats:
        return stats
    flat = {k[len("env_"):]: v for k, v in rec.items() if k.startswith("env_")}
    return flat


def classify(rec: dict) -> str | None:
    """-> bucket name, or None when the episode carries no outcome detail."""
    stats = outcome_stats(rec)
    if "moved_correct_obj" not in stats:
        return None

    if rec.get("success"):
        return "success"

    keep_height = truthy(stats.get("all_obj_keep_height", True))
    correct = truthy(stats.get("moved_correct_obj"))
    wrong = truthy(stats.get("moved_wrong_obj"))
    near = truthy(stats.get("near_tgt_obj"))
    closest = truthy(stats.get("is_closest_to_tgt"))

    if not keep_height:
        return "dropped"
    if wrong:
        return "wrong_object"
    if correct:
        # Success is keep_height and correct and near and closest, so reaching
        # here with `correct` true means the destination test is what failed.
        assert not (near and closest), "success criteria disagree with success flag"
        return "misplaced"
    return "no_contact"


def load(run_dir: str) -> dict:
    """-> {ep_id: bucket} for the move_near task under run_dir."""
    hits = sorted(glob.glob(os.path.join(run_dir, "**", "results_*move_near*.json"),
                            recursive=True))
    if not hits:
        return {}
    with open(hits[0]) as fh:
        summary = json.load(fh)
    out = {}
    for rec in summary["episodes"]:
        bucket = classify(rec)
        if bucket is not None:
            out[int(rec["ep_id"])] = bucket
    return out


COL = 14  # wide enough for the longest bucket name


def fmt_cells(counts: Counter) -> str:
    return "".join(f"{counts[b]:>{COL}d}" for b in BUCKETS)


def main(dirs: list[str]) -> int:
    runs = [(os.path.basename(d.rstrip('/')), load(d)) for d in dirs]
    runs = [(name.replace("openvla_fractal_0806_", ""), eps)
            for name, eps in runs if eps]
    if not runs:
        print("no episodes with outcome detail found -- "
              "the harness for these runs did not store final_info")
        return 1

    header = "".join(f"{b:>{COL}s}" for b in BUCKETS)
    print("move_near failure taxonomy (terminal state, one bucket per episode)\n")
    print(f"  {'condition':22s} {'n':>5s}{header}")
    for name, eps in runs:
        print(f"  {name:22s} {len(eps):5d}{fmt_cells(Counter(eps.values()))}")

    base_name, base = runs[0]
    print(f"\nAmong episodes that {base_name} solved and the condition lost "
          f"(paired, same ep ids):\n")
    print(f"  {'condition':22s} {'lost':>5s}{header}")
    for name, eps in runs[1:]:
        ids = [i for i in sorted(set(base) & set(eps))
               if base[i] == "success" and eps[i] != "success"]
        if not ids:
            print(f"  {name:22s} {0:5d}   -- nothing lost")
            continue
        print(f"  {name:22s} {len(ids):5d}{fmt_cells(Counter(eps[i] for i in ids))}")

    print("\n`wrong_object` = the arm moved an object we did not name; "
          "`no_contact` = nothing moved at all.")
    print("A shift toward the first is consistent with grounding damage, "
          "toward the second with motor damage.")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [DEFAULT_CAMPAIGN.format(c) for c in DEFAULT_CONDITIONS]
        args = [d for d in args if os.path.isdir(d)]
    sys.exit(main(args))
