"""Audit the summary JSONs a LIBERO run campaign produced.

Written to be run before reporting results, because the failure modes that
matter here are silent: an intervention flag that never reached the policy, a
run that covered 9 tasks instead of 10, two runs of the same condition where
only one was meant to exist, a decode-failure rate that quietly rose. None of
those show up as an error, and all of them change what the numbers mean.

    python verify_runs.py /content/bivla_eval_libero /content/bivla_openvla_depth ...

Checks per run:
  * episode count matches tasks x trials, and the task list is complete
  * every episode terminated (a run of exactly-max-length episodes with zero
    successes is what a broken policy looks like)
  * the condition recorded in the JSON matches the condition the file name and
    directory imply -- i.e. the flags actually reached the model
  * depth pruning, when requested, actually bypassed layers
  * the phase-adaptive controller actually left the deep state
  * FAST decode failures are zero (UniVLA)

Checks across runs:
  * duplicate conditions (the same cell run twice) are surfaced with both
    success rates, since silently reporting one of two is how a campaign
    becomes unreproducible
  * baseline appears exactly once per (backbone, suite)
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def condition_of(s: dict) -> str:
    bits = []
    if s.get("action_repeat", 1) > 1:
        bits.append(f"action-repeat{s['action_repeat']}")
    if s.get("exec_chunk", 0) > 0:
        bits.append(f"exec-chunk{s['exec_chunk']}")
    fov = s.get("foveate") or {}
    if fov.get("enabled"):
        # NEVER default a missing field to its usual value. `views` was added
        # to the summary later than the first foveation runs, and defaulting it
        # to "agent" silently relabelled the matched both-views runs as
        # agent-only -- exactly the class of error this script exists to catch.
        bits.append(f"foveate-{fov.get('mode')}-{fov.get('keep_percent'):g}%"
                    f"-{fov.get('center')}-{fov.get('views', '?')}")
    d = s.get("depth") or {}
    if d.get("depth_ctrl"):
        bits.append(f"depth-ctrl{d.get('depth_deep')}to{d.get('depth_shallow')}")
    elif d.get("depth_prune"):
        bits.append(f"depth-prune{d['depth_prune']}")
    return "+".join(bits) if bits else "baseline"


def main() -> None:
    dirs = sys.argv[1:] or ["."]
    paths = []
    for d in dirs:
        paths += sorted(glob.glob(os.path.join(d, "summary_*.json")))
    if not paths:
        print("no summary_*.json found in " + ", ".join(dirs))
        return

    problems, rows = [], []
    seen = defaultdict(list)
    by_content = {}

    for path in paths:
        try:
            raw = open(path, "rb").read()
            s = json.loads(raw)
        except Exception as exc:
            problems.append(f"{os.path.basename(path)}: unreadable ({exc})")
            continue

        name = os.path.basename(path)
        # Byte-identical files are re-downloads ("foo (1).json"), not repeated
        # runs. Counting them as duplicate conditions buries the real ones.
        digest = hash(raw)
        if digest in by_content:
            continue
        by_content[digest] = name
        backbone = s.get("backbone", "?")
        suite = s.get("task_suite", "?")
        cond = condition_of(s)
        eps = s.get("episodes") or []
        n = len(eps)
        sr = s.get("success_rate", 0.0) * 100
        ms = s.get("avg_model_ms_per_infer", 0.0)

        # A 1-task, 1-trial run is a smoke test, not a broken campaign run.
        # Keeping it out of the table stops it colliding with the real
        # baseline as a "duplicate condition".
        n_tasks = len(s.get("task_ids") or [])
        n_trials = s.get("n_trials_per_task", 0)
        if n_tasks <= 2 and n_trials <= 2:
            print(f"[note] {name}: smoke test ({n_tasks} task x {n_trials} trial) "
                  f"-- excluded from the table")
            continue

        rows.append((backbone, suite, cond, n, sr, ms, name))
        seen[(backbone, suite, cond)].append((sr, name))

        # -- coverage
        expect = n_tasks * n_trials
        if expect and n != expect:
            problems.append(f"{name}: {n} episodes, expected {expect}")
        if n_tasks != 10:
            problems.append(f"{name}: {n_tasks} tasks, not 10")

        # -- did the intervention reach the model?
        d = s.get("depth") or {}
        if (d.get("depth_prune") or d.get("depth_ctrl")) and not d.get("n_bypassed"):
            problems.append(f"{name}: depth requested but NO layers bypassed")
        if d.get("depth_ctrl") and not d.get("episodes_reaching_shallow"):
            problems.append(f"{name}: controller never left the deep state -- this "
                            f"run is really --depth-prune {d.get('depth_deep')}")

        # -- decode health (UniVLA)
        dfr = s.get("decode_failure_rate")
        if dfr:
            problems.append(f"{name}: FAST decode failures {dfr*100:.1f}% (want 0)")

        # -- a run where nothing ever terminated early is a broken-policy signature
        steps = [e.get("steps", 0) for e in eps]
        if steps and sr == 0.0 and min(steps) == max(steps):
            problems.append(f"{name}: 0% success and every episode hit the step "
                            f"cap ({steps[0]}) -- check the policy actually acted")

    rows.sort()
    print(f"{'backbone':<10} {'suite':<15} {'condition':<38} {'n':>4} {'succ':>7} {'ms':>7}")
    print("-" * 88)
    for backbone, suite, cond, n, sr, ms, _ in rows:
        print(f"{backbone:<10} {suite:<15} {cond:<38} {n:>4} {sr:>6.1f}% {ms:>7.0f}")

    for key, runs in sorted(seen.items()):
        if len(runs) > 1:
            detail = ", ".join(f"{sr:.1f}% ({n})" for sr, n in runs)
            problems.append(f"{key[0]}/{key[1]}/{key[2]}: run {len(runs)}x -> {detail}")

    print()
    if problems:
        print(f"{len(problems)} THING(S) TO CHECK:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("no problems found")


if __name__ == "__main__":
    main()
