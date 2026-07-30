"""Compare two LIBERO runs as a paired experiment, which is what they are.

Every condition in this campaign replays the SAME (task_id, trial) initial
states through the same deterministic loop, so a run is 50 matched pairs, not
50 independent samples from each of two populations. Treating them as
independent -- which is what a two-proportion z-test does -- throws away the
pairing and badly understates the evidence: at n=50 an unpaired test cannot
resolve anything smaller than roughly 18 points, so real 8-16 point effects get
reported as "noise" when the paired data may say otherwise.

McNemar's test uses only the episodes where the two runs DISAGREE. If an
episode succeeds under both conditions, or fails under both, it carries no
information about which condition is better and only inflates the variance.

    python paired_test.py baseline.json foveate_blur.json

Prints the 2x2 agreement table, the exact binomial p-value on the discordant
pairs, and the per-task breakdown -- the last because a change that costs 5
points spread evenly is a different phenomenon from one that destroys a single
task, and the aggregate number cannot tell them apart.

Refuses to run on mismatched episode sets rather than silently intersecting
them, because a partial overlap ("I only re-ran tasks 8 and 9") produces a
p-value that looks fine and answers a question nobody asked.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb


def load(path: str) -> tuple[dict, dict]:
    with open(path) as fh:
        s = json.load(fh)
    eps = {}
    for e in s.get("episodes") or []:
        eps[(e["task_id"], e["trial"])] = bool(e["success"])
    if not eps:
        raise SystemExit(f"{path}: no episodes in this summary")
    return s, eps


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


def label(summary: dict, path: str) -> str:
    bits = [summary.get("backbone", "?")]
    fov = summary.get("foveate") or {}
    if fov.get("enabled"):
        bits.append(f"fov-{fov.get('mode')}-{fov.get('keep_percent'):g}%")
    d = summary.get("depth") or {}
    if d.get("depth_ctrl"):
        bits.append(f"depth-ctrl{d.get('depth_deep')}to{d.get('depth_shallow')}")
    elif d.get("depth_prune"):
        bits.append(f"depth-prune{d['depth_prune']}")
    if summary.get("action_repeat", 1) > 1:
        bits.append(f"rep{summary['action_repeat']}")
    if len(bits) == 1:
        bits.append("baseline")
    return f"{'/'.join(bits)}  [{os.path.basename(path)}]"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python paired_test.py <run_a.json> <run_b.json>")
    path_a, path_b = sys.argv[1], sys.argv[2]
    sum_a, a = load(path_a)
    sum_b, b_eps = load(path_b)

    if set(a) != set(b_eps):
        only_a, only_b = sorted(set(a) - set(b_eps)), sorted(set(b_eps) - set(a))
        raise SystemExit(
            "these two runs do not cover the same episodes, so they are not "
            "paired.\n"
            f"  only in A ({len(only_a)}): {only_a[:8]}{' ...' if len(only_a) > 8 else ''}\n"
            f"  only in B ({len(only_b)}): {only_b[:8]}{' ...' if len(only_b) > 8 else ''}"
        )
    if sum_a.get("task_suite") != sum_b.get("task_suite"):
        raise SystemExit(
            f"different suites ({sum_a.get('task_suite')} vs "
            f"{sum_b.get('task_suite')}) -- nothing to pair"
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
    print(f"  discordant pairs: {a_only + b_only} "
          f"({a_only} A-only, {b_only} B-only)")
    print(f"  McNemar exact two-sided p = {p:.4f}")

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
