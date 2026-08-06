"""Compare two runs as a paired experiment, which is what they are.

Every condition in this campaign replays the SAME initial states through the
same deterministic loop, so a run is N matched pairs, not N independent samples
from each of two populations. Treating them as independent -- which is what a
two-proportion z-test does -- throws away the pairing and badly understates the
evidence: at n=50 an unpaired test cannot resolve anything smaller than roughly
18 points, so real 8-16 point effects get reported as "noise" when the paired
data may say otherwise.

McNemar's test uses only the episodes where the two runs DISAGREE. If an
episode succeeds under both conditions, or fails under both, it carries no
information about which condition is better and only inflates the variance.

Both harnesses are supported, and each is detected from the episode records
rather than from a flag:

    LIBERO      one summary JSON per run, episodes keyed by (task_id, trial)
    SimplerEnv  one summary JSON per TASK, episodes keyed by ep_id

    python paired_test.py baseline.json foveate_blur.json         # LIBERO
    python paired_test.py runs/baseline runs/action_repeat2       # SimplerEnv

For SimplerEnv, pass the directory holding the per-task subdirectories; every
`results_*.json` beneath it is merged and keyed by (task, ep_id). The per-task
files are required to agree on the condition they were run under -- a directory
that mixes, say, an action-repeat run with a baseline run is a mistake, not a
condition, so it is refused rather than averaged.

Prints the 2x2 agreement table, the exact binomial p-value on the discordant
pairs, and the per-task breakdown -- the last because a change that costs 5
points spread evenly is a different phenomenon from one that destroys a single
task, and the aggregate number cannot tell them apart.

Refuses to run on mismatched episode sets rather than silently intersecting
them, because a partial overlap ("I only re-ran tasks 8 and 9") produces a
p-value that looks fine and answers a question nobody asked.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from math import comb

# Fields that define WHICH condition a SimplerEnv run is. Per-task files inside
# one directory must agree on all of them, or the directory is not one run.
_CONDITION_FIELDS = (
    "model_type", "model", "exec_chunk", "action_repeat", "llm_prune_count",
    "foveate", "fastv", "depth_ctrl",
)


def _episodes_of(summary: dict, path: str) -> tuple[str, dict]:
    """-> (schema, {key: success}) for one summary, schema detected from a record."""
    records = summary.get("episodes") or []
    if not records:
        raise SystemExit(f"{path}: no episodes in this summary")
    first = records[0]
    if "task_id" in first and "trial" in first:
        return "libero", {
            (e["task_id"], e["trial"]): bool(e["success"]) for e in records
        }
    if "ep_id" in first:
        task = summary.get("task")
        if not task:
            raise SystemExit(f"{path}: SimplerEnv summary has no 'task' field")
        return "simpler_env", {
            (task, e["ep_id"]): bool(e["success"]) for e in records
        }
    raise SystemExit(
        f"{path}: cannot tell which harness wrote this -- episode records have "
        f"neither (task_id, trial) nor ep_id, only {sorted(first)}"
    )


def load(path: str) -> tuple[dict, dict]:
    """Load one run. `path` is a summary JSON, or a SimplerEnv run directory."""
    if not os.path.isdir(path):
        with open(path) as fh:
            summary = json.load(fh)
        schema, eps = _episodes_of(summary, path)
        summary.setdefault("task_suite", schema)
        return summary, eps

    files = sorted(
        glob.glob(os.path.join(path, "results_*.json"))
        + glob.glob(os.path.join(path, "*", "results_*.json"))
    )
    if not files:
        raise SystemExit(f"{path}: no results_*.json under this directory")

    merged: dict = {}
    per_file: list[tuple[str, dict]] = []
    for f in files:
        with open(f) as fh:
            s = json.load(fh)
        schema, eps = _episodes_of(s, f)
        if schema != "simpler_env":
            raise SystemExit(
                f"{f}: a directory of runs is the SimplerEnv layout, but this "
                f"file is {schema}. Pass the LIBERO summary JSON directly."
            )
        clash = merged.keys() & eps.keys()
        if clash:
            raise SystemExit(
                f"{f}: episodes {sorted(clash)[:4]} already came from another "
                f"file in {path} -- the same task appears twice"
            )
        merged.update(eps)
        per_file.append((f, s))

    # One directory is supposed to be ONE condition. If the per-task files
    # disagree about the condition, merging them produces a number that
    # describes no experiment that was actually run.
    ref_path, ref = per_file[0]
    for f, s in per_file[1:]:
        for field in _CONDITION_FIELDS:
            if s.get(field) != ref.get(field):
                raise SystemExit(
                    f"{path}: these per-task files were not run under the same "
                    f"condition -- '{field}' is {ref.get(field)!r} in "
                    f"{os.path.basename(ref_path)} but {s.get(field)!r} in "
                    f"{os.path.basename(f)}"
                )

    summary = dict(ref)
    summary["task"] = f"{len(per_file)} tasks"
    summary["task_suite"] = "simpler_env"
    return summary, merged


def exact_two_sided(b: int, c: int) -> float:
    """Exact binomial p for b successes out of n=b+c under p=0.5.

    The chi-square form of McNemar is only valid when b+c is large; these runs
    routinely produce 5-15 discordant pairs, where it is not. The exact test
    costs nothing here.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _exec_chunk(summary: dict) -> int:
    """Executed chunk length k, 0 meaning "the whole predicted chunk".

    Three harnesses write this field three ways -- an int (UniVLA, OpenVLA),
    `{"enabled": False}` or `{"enabled": True, "k": n}` (SpatialVLA) -- so it
    has to be read rather than assumed. Comparing a dict to an int raises,
    which at least fails loudly; silently reading it as 0 would not.
    """
    v = summary.get("exec_chunk")
    if isinstance(v, dict):
        return int(v.get("k", 0)) if v.get("enabled") else 0
    return int(v or 0)


def _prune_count(summary: dict) -> int:
    """Number of bypassed decoder layers, across the same three schemas."""
    d = summary.get("depth_prune")
    if isinstance(d, dict):
        return int(d.get("count", 0))
    if d is not None:
        return int(d or 0)
    nested = summary.get("depth") or {}
    return int(nested.get("depth_prune") or summary.get("llm_prune_count") or 0)


def horizon(summary: dict) -> int | None:
    """Env steps executed per model call, or None if the run did not record it.

    This is the quantity the temporal conditions actually change, and it is not
    comparable across backbones without being stated: a chunking policy at
    action-repeat 2 sits at 2x its chunk length, not at 2.
    """
    if "action_repeat" not in summary:
        return None
    repeat = int(summary.get("action_repeat", 1) or 1)
    chunk = _exec_chunk(summary)
    if chunk <= 0:  # 0 means "execute the whole predicted chunk"
        chunk = int(summary.get("predict_action_frames", 1) or 1)
    return chunk * repeat


def label(summary: dict, path: str) -> str:
    bits = [summary.get("backbone") or summary.get("model_type")
            or summary.get("model") or "?"]
    fov = summary.get("foveate") or {}
    if fov.get("enabled"):
        bits.append(f"fov-{fov.get('mode')}-{fov.get('keep_percent'):g}%")
    # LIBERO nests the depth-controller settings under "depth"; the SimplerEnv
    # harnesses write depth_ctrl / llm_prune_count / depth_prune at the top.
    d = summary.get("depth") or {}
    ctrl = summary.get("depth_ctrl") or {}
    if d.get("depth_ctrl") or ctrl.get("enabled"):
        deep = d.get("depth_deep", ctrl.get("deep"))
        shallow = d.get("depth_shallow", ctrl.get("shallow"))
        bits.append(f"depth-ctrl{deep}to{shallow}")
    elif _prune_count(summary):
        bits.append(f"depth-prune{_prune_count(summary)}")
    if _exec_chunk(summary) > 0:
        bits.append(f"chunk{_exec_chunk(summary)}")
    if int(summary.get("action_repeat", 1) or 1) > 1:
        bits.append(f"rep{summary['action_repeat']}")
    if len(bits) == 1:
        bits.append("baseline")
    h = horizon(summary)
    tail = "" if h is None else f", {h} env step{'s' if h != 1 else ''}/call"
    return f"{'/'.join(bits)}  [{os.path.basename(path.rstrip('/'))}{tail}]"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python paired_test.py <run_a> <run_b>\n"
            "  each argument is a LIBERO summary JSON, or a SimplerEnv run "
            "directory containing per-task results_*.json"
        )
    path_a, path_b = sys.argv[1], sys.argv[2]
    sum_a, a = load(path_a)
    sum_b, b_eps = load(path_b)

    # Suite first: if these are different benchmarks the episode-set diff below
    # would just dump every key from both, which buries the actual problem.
    if sum_a.get("task_suite") != sum_b.get("task_suite"):
        raise SystemExit(
            f"different suites ({sum_a.get('task_suite')} vs "
            f"{sum_b.get('task_suite')}) -- nothing to pair"
        )
    if set(a) != set(b_eps):
        only_a, only_b = sorted(set(a) - set(b_eps)), sorted(set(b_eps) - set(a))
        raise SystemExit(
            "these two runs do not cover the same episodes, so they are not "
            "paired.\n"
            f"  only in A ({len(only_a)}): {only_a[:8]}{' ...' if len(only_a) > 8 else ''}\n"
            f"  only in B ({len(only_b)}): {only_b[:8]}{' ...' if len(only_b) > 8 else ''}"
        )

    keys = sorted(a)
    both = sum(1 for k in keys if a[k] and b_eps[k])
    a_only = sum(1 for k in keys if a[k] and not b_eps[k])
    b_only = sum(1 for k in keys if not a[k] and b_eps[k])
    neither = sum(1 for k in keys if not a[k] and not b_eps[k])
    n = len(keys)

    sr_a, sr_b = (both + a_only) / n * 100, (both + b_only) / n * 100
    p = exact_two_sided(a_only, b_only)

    print(f"A = {label(sum_a, path_a)}")
    print(f"B = {label(sum_b, path_b)}")
    print(f"\n{n} paired episodes\n")
    print(f"{'':>14} {'B success':>10} {'B fail':>8}")
    print(f"{'A success':>14} {both:>10} {a_only:>8}")
    print(f"{'A fail':>14} {b_only:>10} {neither:>8}")
    print(f"\n  A {sr_a:.1f}%   B {sr_b:.1f}%   difference {sr_b - sr_a:+.1f} points")
    ha, hb = horizon(sum_a), horizon(sum_b)
    if ha is not None and hb is not None and ha != hb:
        print(f"  NOTE: different open-loop horizons ({ha} vs {hb} env steps "
              f"per model call) -- this p-value says the two runs differ, not "
              f"that the intervention would differ at a matched horizon")
    print(f"  discordant pairs: {a_only + b_only} "
          f"({a_only} A-only, {b_only} B-only)")
    # A large, lopsided set of discordant pairs drives p below what %.4f can
    # show, and printing "0.0000" reads as a formatting bug rather than a result.
    print(f"  McNemar exact two-sided p = {p:.4f}" if p >= 1e-4
          else f"  McNemar exact two-sided p = {p:.2e}")

    if a_only + b_only < 6:
        print("  -- too few disagreements to conclude anything either way")
    elif p < 0.05:
        print("  -- the difference is larger than chance disagreement explains")
    else:
        print("  -- NOT distinguishable from chance; this is 'not detected', "
              "not 'no effect'")

    print("\nper task (A -> B):")
    tasks = sorted({k[0] for k in keys})
    for t in tasks:
        ks = [k for k in keys if k[0] == t]
        na = sum(a[k] for k in ks)
        nb = sum(b_eps[k] for k in ks)
        flip = "".join("." if a[k] == b_eps[k] else ("-" if a[k] else "+") for k in ks)
        print(f"  task {t}: {na}/{len(ks)} -> {nb}/{len(ks)}   {flip}")
    print("  (+ = B fixed an A failure, - = B broke an A success, . = same)")


if __name__ == "__main__":
    main()
