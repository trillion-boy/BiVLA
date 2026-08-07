#!/usr/bin/env python3
"""Build the cross-backbone x cross-benchmark grid from the committed records.

Every number in the report is computed here from `results/**/results_*.json`,
never transcribed. Transcription is how a table and its data drift apart, and
this campaign's whole claim is a comparison between cells -- a single mistyped
cell would be a claim about nothing.

Cells that have not been run print as "--" and cells that are only partly run
print with the episode count, so an incomplete campaign reads as incomplete
rather than as a smaller result.

    python experiments/build_grid_report.py            # markdown to stdout
    python experiments/build_grid_report.py --json     # machine-readable

Re-run it whenever a condition finishes; nothing else needs editing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RESULTS = os.path.join(_ROOT, "results")


def exact_two_sided(b: int, c: int) -> float:
    """Exact binomial p for the discordant split, same rule as paired_test.py.

    The chi-square McNemar is invalid at the 5-20 discordant pairs these runs
    produce. Kept identical to `adaptive_sparse_vla/paired_test.py` so a number
    here and a number there cannot disagree.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


# Which campaign directory belongs to which (backbone, benchmark), and how the
# condition is spelled inside it. The two harnesses lay their output out
# differently -- SpatialVLA nests <condition>/<task>, the OpenVLA Fractal driver
# suffixes the campaign directory and inserts a <model> level -- so the mapping
# is declared rather than guessed from the path shape.
CAMPAIGNS = [
    ("OpenVLA",    "Bridge",  "openvla_bridge_0805",      "nested"),
    ("SpatialVLA", "Bridge",  "spatialvla_bridge_0805",   "nested"),
    ("UniVLA",     "Bridge",  "univla_bridge_0805",       "nested"),
    ("SpatialVLA", "Fractal", "spatialvla_fractal_0806",  "nested"),
    ("OpenVLA",    "Fractal", "openvla_fractal_0806",     "suffixed"),
]

# Display order and display names. Anything found but not listed still appears,
# appended, so a new condition is never silently dropped from the report.
CONDITION_ORDER = [
    ("baseline",         "original policy"),
    ("action_repeat2",   "action repeat 2"),
    ("action_repeat4",   "action repeat 4"),
    ("foveate_logpolar", "foveation log-polar 20%"),
    ("foveate",          "foveation log-polar 20%"),
    ("foveate_blur",     "foveation blur 20%"),
    ("depth_prune1",     "depth prune 1"),
    ("depth_prune4",     "depth prune 4"),
    ("depth_prune8",     "depth prune 8"),
]
DISPLAY = dict(CONDITION_ORDER)
# foveate / foveate_logpolar are the same condition under two harness spellings.
CANON = {"foveate": "foveate_logpolar"}


def benchmark_of(task: str) -> str:
    return "Fractal" if task.startswith("google_robot") else "Bridge"


def discover() -> dict:
    """-> {(backbone, benchmark, condition): {task: {ep_id: success}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for backbone, bench, campaign, layout in CAMPAIGNS:
        base = os.path.join(RESULTS, campaign)
        if layout == "nested":
            if not os.path.isdir(base):
                continue
            conditions = [(c, os.path.join(base, c)) for c in sorted(os.listdir(base))
                          if os.path.isdir(os.path.join(base, c))]
        else:
            # <campaign>_<condition>/<model>/<task>/
            parent = os.path.dirname(base)
            prefix = os.path.basename(base) + "_"
            conditions = []
            for d in sorted(os.listdir(parent)) if os.path.isdir(parent) else []:
                if not d.startswith(prefix):
                    continue
                cond_dir = os.path.join(parent, d)
                models = [m for m in sorted(os.listdir(cond_dir))
                          if os.path.isdir(os.path.join(cond_dir, m))]
                for m in models:
                    conditions.append((d[len(prefix):], os.path.join(cond_dir, m)))

        for cond, cdir in conditions:
            cond = CANON.get(cond, cond)
            for task in sorted(os.listdir(cdir)):
                tdir = os.path.join(cdir, task)
                path = os.path.join(tdir, f"results_{task}.json")
                if not os.path.isfile(path):
                    continue
                with open(path) as fh:
                    summary = json.load(fh)
                # Trust the record's own task name over the directory, and the
                # task name over the campaign, so a file filed in the wrong
                # place lands in the right cell instead of corrupting one.
                task = summary.get("task", task)
                key = (backbone, benchmark_of(task), cond)
                out[key][task] = {int(e["ep_id"]): bool(e["success"])
                                  for e in summary["episodes"]}
    return out


def cell(data: dict, backbone: str, bench: str, cond: str):
    """-> (n_success, n_total) pooled over tasks, or None if the cell is empty."""
    tasks = data.get((backbone, bench, cond))
    if not tasks:
        return None
    ok = sum(v for t in tasks.values() for v in t.values())
    n = sum(len(t) for t in tasks.values())
    return ok, n


def paired(data: dict, backbone: str, bench: str, cond: str):
    """-> (delta_points, fixed, broke, p) against this cell's own baseline.

    Only episodes present in BOTH runs are paired. A condition that has run
    three of four tasks is therefore compared against the same three tasks of
    the baseline, not against the baseline's full protocol -- otherwise a
    partial run would read as a collapse.
    """
    base = data.get((backbone, bench, "baseline"))
    cur = data.get((backbone, bench, cond))
    if not base or not cur or cond == "baseline":
        return None
    fixed = broke = same = 0
    for task, eps in cur.items():
        b = base.get(task)
        if not b:
            continue
        for ep, ok in eps.items():
            if ep not in b:
                continue
            if b[ep] and not ok:
                broke += 1
            elif ok and not b[ep]:
                fixed += 1
            else:
                same += 1
    n = fixed + broke + same
    if n == 0:
        return None
    delta = 100.0 * (fixed - broke) / n
    return delta, fixed, broke, exact_two_sided(fixed, broke), n


def fmt_cell(c, pr) -> str:
    if c is None:
        return "--"
    ok, n = c
    s = f"{100.0*ok/n:.1f}%"
    if pr is None:
        return f"**{s}** (n={n})"
    d, fixed, broke, p, npair = pr
    star = "***" if p < 0.003 else "**" if p < 0.05 else ""
    return f"{s}  {d:+.1f}{star}"


def conditions_present(data) -> list:
    seen = {k[2] for k in data}
    ordered = [c for c, _ in CONDITION_ORDER if c in seen]
    return ordered + sorted(seen - set(ordered))


def markdown(data) -> str:
    L = []
    cols = [("OpenVLA", "Bridge"), ("OpenVLA", "Fractal"),
            ("SpatialVLA", "Bridge"), ("SpatialVLA", "Fractal"),
            ("UniVLA", "Bridge")]
    cols = [c for c in cols if any(k[0] == c[0] and k[1] == c[1] for k in data)]

    L.append("## The grid\n")
    L.append("Success rate, and the paired delta against that column's own "
             "baseline. `**` clears p<0.05, `***` clears the Bonferroni "
             "threshold for this campaign (a~0.003). Deltas use only episodes "
             "present in both runs.\n")
    L.append("| condition | " + " | ".join(f"{b}<br>{k}" for b, k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for cond in conditions_present(data):
        row = [DISPLAY.get(cond, cond)]
        for b, k in cols:
            row.append(fmt_cell(cell(data, b, k, cond), paired(data, b, k, cond)))
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## Paired detail\n")
    L.append("| backbone | benchmark | condition | n | delta | fixed | broke | p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for b, k in cols:
        for cond in conditions_present(data):
            pr = paired(data, b, k, cond)
            if pr is None:
                continue
            d, fixed, broke, p, n = pr
            L.append(f"| {b} | {k} | {DISPLAY.get(cond, cond)} | {n} | "
                     f"{d:+.1f} | {fixed} | {broke} | {p:.4f} |")

    L.append("\n## Per task\n")
    for b, k in cols:
        tasks = sorted({t for key, v in data.items()
                        if key[0] == b and key[1] == k for t in v})
        if not tasks:
            continue
        L.append(f"\n**{b} / {k}**\n")
        L.append("| condition | " + " | ".join(t.replace("google_robot_", "")
                                               .replace("widowx_", "") for t in tasks) + " |")
        L.append("|---|" + "---|" * len(tasks))
        for cond in conditions_present(data):
            got = data.get((b, k, cond))
            if not got:
                continue
            row = [DISPLAY.get(cond, cond)]
            for t in tasks:
                eps = got.get(t)
                row.append("--" if not eps else
                           f"{100.0*sum(eps.values())/len(eps):.1f}%")
            L.append("| " + " | ".join(row) + " |")

    missing = []
    for b, k in cols:
        for cond in conditions_present(data):
            if not data.get((b, k, cond)):
                missing.append(f"{b}/{k}: {DISPLAY.get(cond, cond)}")
    if missing:
        L.append("\n## Not run\n")
        L.extend(f"- {m}" for m in missing)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    data = discover()
    if not data:
        sys.exit(f"no result records found under {RESULTS}")
    if args.json:
        out = {}
        for (b, k, c) in sorted(data):
            cl, pr = cell(data, b, k, c), paired(data, b, k, c)
            out[f"{b}/{k}/{c}"] = {
                "success": cl[0], "n": cl[1], "rate": cl[0] / cl[1],
                **({} if pr is None else
                   {"delta": pr[0], "fixed": pr[1], "broke": pr[2],
                    "p": pr[3], "n_paired": pr[4]}),
            }
        print(json.dumps(out, indent=2))
    else:
        print(markdown(data))


if __name__ == "__main__":
    main()
