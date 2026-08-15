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

# Per-episode records that live outside results/ because they predate it.
# Each is (backbone, benchmark, condition, records_dir, verify_baseline_dir):
# the last field is that campaign's OWN baseline, which must be episode-for-
# episode identical to the current one or the borrowed condition cannot be
# paired against it. Checked at load time, not assumed -- borrowing a condition
# from a campaign whose baseline drifted would silently mix two experiments.
IMPORTED = [
    ("OpenVLA", "Bridge", "foveate_logpolar",
     "RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated",
     "RetinaBased/GoogleColab/results_reproduction_eager/openvla"),
]

# Cells that were measured but whose per-episode records were not kept. They
# are reported because leaving them blank invites the reader to assume they
# were never run -- but they are UNPAIRED, and their delta is against their own
# campaign's baseline, which is not the baseline in this table's top row. That
# is why they carry their own baseline here and are typeset differently: you
# cannot subtract them from this column's baseline and get the same number.
#
# (rate, delta, own_baseline, source_document)
# Empty since 2026-08-10: every cell in the grid now has per-episode records.
# The table and its footnote stay in the code because the campaign will grow,
# and a cell that gets reported without records has to be visibly different
# from one that was paired -- but nothing is in that state today.
#
# What was retired, and what re-measuring cost each claim:
#   SpatialVLA/Bridge foveate_logpolar -- was 25.0% against a 32.3% baseline;
#     paired against this column's own 30.2% it is 21.9%, delta -8.3 at
#     p = 0.20. The direction survives, the claim does not.
#   SpatialVLA/Bridge foveate_blur -- was -2.1; paired it is exactly 0.0,
#     13 broken against 13 fixed, p = 1.0000.
#   SpatialVLA/Bridge depth_prune1 -- superseded by the paired -10.4.
#   UniVLA/Bridge foveation (both variants) -- see
#     LabMeeting_Bridge_Fractal_0806*.md; the log-polar cell reproduced its old
#     pooled rate to the decimal while three of four task rates had moved.
LEGACY = {
    # The two UniVLA/Bridge foveation cells used to live here. They were
    # re-measured with per-episode records on 2026-08-09 and now come out of
    # `results/`, so they are gone from this table. What the old numbers are
    # still good for is written up in LabMeeting_Bridge_Fractal_0806*.md: the
    # log-polar cell landed on the identical pooled rate (86.5%) while three of
    # its four task rates had moved, and the blur cell missed its old pooled
    # rate by 3.1 points. Keep them out of the grid; keep the comparison.
}

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
    ("depth_prune2",     "depth prune 2"),
    ("depth_prune4",     "depth prune 4"),
    ("depth_prune8",     "depth prune 8"),
    # Combinations go last and are named for their parts, so a reader can find
    # both single-axis rows above and check the sum for themselves.
    ("prune2_repeat2",   "depth prune 2 + action repeat 2"),
]
# Runs that are controls for a specific question rather than cells of the grid.
# `baseline_rerun` is the determinism check; listing it as a condition would
# report "no change against baseline" as though that were a finding.
# `depth_prune4_early`, `depth_prune4_mid`, `depth_prune4_back` and
# `depth_prune1_back` all remove the same number of layers as the condition they
# are named after, from a different region. Putting them in one column under
# adjacent names is the exact confusion that produced 3c-bis in the first place,
# and the region contrast is the whole point of them. They are analysed in the
# report where their question lives (§3.5, §3.6).
EXCLUDE = {"baseline_rerun", "depth_prune4_early", "depth_prune4_mid",
           "depth_prune4_back", "depth_prune1_back"}
DISPLAY = dict(CONDITION_ORDER)
# foveate / foveate_logpolar are the same condition under two harness spellings.
CANON = {"foveate": "foveate_logpolar"}

# UniVLA/Bridge was measured on two different GPUs. `baseline` and
# `action_repeat2` came off the July card; everything else came off an L4, and a
# re-run of one condition on the same L4 reproduced 24/24 while the July-vs-L4
# baselines differ by 3.1 points. Subtracting across that boundary would put a
# hardware component inside every delta, so the column is rebuilt around the L4
# pair and the July pair drops out of the grid. It is not deleted -- the two
# pairs measured under identical settings on different cards are the evidence
# for the hardware caveat in the report, which is where they are discussed.
#
# Keyed by (backbone, benchmark): {raw condition -> canonical, or None to drop}.
RENAME = {
    ("UniVLA", "Bridge"): {
        "baseline": None,
        "action_repeat2": None,
        "baseline_l4": "baseline",
        "action_repeat2_l4": "action_repeat2",
    },
}


def benchmark_of(task: str) -> str:
    return "Fractal" if task.startswith("google_robot") else "Bridge"


def _walk_campaigns():
    """Yield (backbone, condition, task, summary) for every committed record.

    Shared by the success grid and the cost table so the two can never end up
    reading different files and disagreeing about which cells exist.
    """
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
            renamed = RENAME.get((backbone, bench), {})
            if cond in renamed:
                if renamed[cond] is None:
                    continue
                cond = renamed[cond]
            for task in sorted(os.listdir(cdir)):
                path = os.path.join(cdir, task, f"results_{task}.json")
                if not os.path.isfile(path):
                    continue
                with open(path) as fh:
                    summary = json.load(fh)
                # Trust the record's own task name over the directory, and the
                # task name over the campaign, so a file filed in the wrong
                # place lands in the right cell instead of corrupting one.
                yield backbone, cond, summary.get("task", task), summary


def discover() -> dict:
    """-> {(backbone, benchmark, condition): {task: {ep_id: success}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for backbone, cond, task, summary in _walk_campaigns():
        out[(backbone, benchmark_of(task), cond)][task] = {
            int(e["ep_id"]): bool(e["success"]) for e in summary["episodes"]}

    for backbone, bench, cond, rec_dir, base_dir in IMPORTED:
        rec = _load_dir(os.path.join(_ROOT, rec_dir))
        if not rec:
            continue
        theirs = _load_dir(os.path.join(_ROOT, base_dir))
        ours = out.get((backbone, bench, "baseline"))
        if not _same_episodes(theirs, ours):
            print(f"# skipped {backbone}/{bench} {cond}: its campaign baseline "
                  f"is not episode-identical to the current one, so it cannot "
                  f"be paired against it", file=sys.stderr)
            continue
        out[(backbone, bench, cond)] = rec
    return out


def _load_dir(base: str) -> dict:
    """-> {task: {ep_id: success}} for a <dir>/<task>/results_<task>.json tree."""
    got: dict = {}
    if not os.path.isdir(base):
        return got
    for task in sorted(os.listdir(base)):
        path = os.path.join(base, task, f"results_{task}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            summary = json.load(fh)
        got[summary.get("task", task)] = {int(e["ep_id"]): bool(e["success"])
                                          for e in summary["episodes"]}
    return got


def _same_episodes(a: dict | None, b: dict | None) -> bool:
    """Do two runs agree on every episode they share, and cover the same set?"""
    if not a or not b or set(a) != set(b):
        return False
    for task in a:
        if a[task] != b[task]:
            return False
    return True


def cell(data: dict, backbone: str, bench: str, cond: str):
    """-> (n_success, n_total) pooled over tasks, or None if the cell is empty."""
    tasks = data.get((backbone, bench, cond))
    if not tasks:
        return None
    ok = sum(v for t in tasks.values() for v in t.values())
    n = sum(len(t) for t in tasks.values())
    return ok, n


def paired(data: dict, backbone: str, bench: str, cond: str, task_filter=None):
    """-> (delta_points, fixed, broke, p) against this cell's own baseline.

    Only episodes present in BOTH runs are paired. A condition that has run
    three of four tasks is therefore compared against the same three tasks of
    the baseline, not against the baseline's full protocol -- otherwise a
    partial run would read as a collapse.

    task_filter restricts the pairing to a subset of tasks. The whole point of
    the task-family split is that a cell's aggregate can be the average of two
    subsets moving in opposite directions, so that split has to be computed
    from the records here rather than transcribed into the report by hand.
    """
    base = data.get((backbone, bench, "baseline"))
    cur = data.get((backbone, bench, cond))
    if not base or not cur or cond == "baseline":
        return None
    fixed = broke = same = 0
    for task, eps in cur.items():
        b = base.get(task)
        if not b or (task_filter is not None and not task_filter(task)):
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


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]].

    Hand-rolled because scipy is not a dependency of this repo and this is the
    only test here that McNemar cannot answer.
    """
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return 1.0
    def p(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 + 1e-12))


def interaction(data, cond: str, left: tuple, right: tuple):
    """Does this condition act differently on `left` than on `right`?

    McNemar answers "did this condition change anything" within one column. It
    cannot answer "is the effect different over there", which is the actual
    claim: the two runs share no episodes, so there is nothing to pair. What
    generalises is the DISCORDANT SPLIT -- how the intervention divides the
    episodes it moved -- and whether that split differs is a 2x2 Fisher test.

    This matters because both cells can individually fail to reach p<0.05 while
    the difference between them clears it. Reporting only the per-cell tests
    would understate exactly the effect the campaign is about.
    """
    a, b = paired(data, left[0], left[1], cond), paired(data, right[0], right[1], cond)
    if a is None or b is None:
        return None
    # (fixed, broke) for each side
    return (a[1], a[2], b[1], b[2],
            fisher_exact_2x2(a[1], b[1], a[2], b[2]))


def grid_family(data, cols) -> int:
    """How many paired tests the grid actually runs -- the Bonferroni denominator.

    Counted, never written down: the number is 5 cells x 7 conditions plus the
    conditions that only some cells ran (`depth prune 8` in two, `depth prune 2
    + action repeat 2` in one), and it moves whenever a condition is added. It
    was hardcoded three times in the prose and was wrong all three times, so the
    only honest source is this count.
    """
    return sum(1 for b, k in cols for cond in conditions_present(data)
               if paired(data, b, k, cond) is not None)


def fmt_cell(c, pr, alpha, legacy=None) -> str:
    if c is None:
        if legacy is None:
            return "--"
        # Italic + dagger, never bold: a legacy cell must not read like a
        # measured-and-paired one at a glance.
        rate, delta, base, _src = legacy
        return f"*{rate:.1f}%  {delta:+.1f}†*"
    ok, n = c
    s = f"{100.0*ok/n:.1f}%"
    if pr is None:
        return f"**{s}** (n={n})"
    d, fixed, broke, p, npair = pr
    star = "***" if p < alpha else "**" if p < 0.05 else ""
    return f"{s}  {d:+.1f}{star}"


# Fractal splits cleanly into one task that requires resolving *which* of
# several named objects to act on, and three that name a single target in a
# fixed instruction. Two unrelated interventions (log-polar foveation on
# OpenVLA, depth pruning on SpatialVLA) both spare the second group and damage
# the first, which is the campaign's only mechanism -- so the split is reported
# rather than left for a reader to reconstruct from the per-task table.
TASK_FAMILIES = {
    "Fractal": [
        ("referential (move_near)", lambda t: t.startswith("google_robot_move_near")),
        ("single-target (pick_coke_can)", lambda t: "coke_can" in t),
    ],
}


def task_family_section(data, cols) -> list:
    L = []
    body = []
    # This is its own multiple-comparison family: every cell is split into two
    # task families and each half gets its own paired test, so the denominator
    # is neither the grid's nor the Fisher table's. Counted in one pass first so
    # the stars below use a threshold this section actually earned.
    n_fam = 0
    for b, k in cols:
        fams = TASK_FAMILIES.get(k)
        if not fams:
            continue
        for cond in conditions_present(data):
            if cond == "baseline" or not data.get((b, k, cond)):
                continue
            cells = [paired(data, b, k, cond, f) for _, f in fams]
            if not any(c is None for c in cells):
                n_fam += len(cells)
    fam_alpha = 0.05 / max(1, n_fam)
    for b, k in cols:
        fams = TASK_FAMILIES.get(k)
        if not fams:
            continue
        rows = []
        for cond in conditions_present(data):
            if cond == "baseline" or not data.get((b, k, cond)):
                continue
            cells = [paired(data, b, k, cond, f) for _, f in fams]
            if any(c is None for c in cells):
                continue
            rows.append((cond, cells))
        # One row cannot show a divergence, so there is nothing to report.
        if len(rows) < 2:
            continue
        body.append(f"\n**{b} / {k}**\n")
        body.append("| condition | " + " | ".join(
            f"{name} (n={cells[i][4]})" for i, (name, _) in enumerate(fams)
            for cond, cells in [rows[0]]) + " |")
        body.append("|---|" + "---|" * len(fams))
        for cond, cells in rows:
            cs = []
            for d, fixed, broke, p, _n in cells:
                star = "***" if p < fam_alpha else "**" if p < 0.05 else ""
                cs.append(f"{d:+.1f}{star} ({fixed} fixed / {broke} broke, p={p:.4f})")
            body.append(f"| {DISPLAY.get(cond, cond)} | " + " | ".join(cs) + " |")
    if not body:
        return L
    L.append("\n## By task family\n")
    L.append("A cell's aggregate delta is an average over its tasks, and on "
             "Fractal those tasks do not agree. `move_near` is the only task "
             "requiring you to resolve *which* of three named objects to act "
             "on; the three `pick_coke_can` variants share one instruction and "
             "one target. Deltas below are paired within each family against "
             "the same family of the baseline.\n")
    L.append(f"Own multiple-comparison family: {n_fam} paired tests, so `***` "
             f"is p < 0.05/{n_fam} ~ {fam_alpha:.4f} and `**` is p<0.05. This is "
             f"a different denominator from the grid's and from the Fisher "
             f"table's -- quoting one threshold for all three is how a family "
             f"size ends up written down wrong.\n")
    L.extend(body)
    return L


def discover_cost() -> dict:
    """-> {(backbone, benchmark, condition): {"infer": [...], "step": [...], "sec": [...]}}

    Three different quantities, because picking the wrong one reverses the
    conclusion for a whole axis:

    * `infer`  -- ms per model call. Action repeat does not change this at all;
      it changes how OFTEN the model is called. Reading the cost of action
      repeat off this column says it is free, which is wrong in the direction
      that flatters it.
    * `step`   -- ms of model time per environment step, i.e. `infer` amortised
      over the calls actually made. This is the compute number. Harnesses that
      do not record it get it derived as `infer / action_repeat`, which is
      exact: repeat k queries the policy once every k steps.
    * `sec`    -- wall-clock seconds per episode. NOT a clean cost metric: a
      condition that fails more runs to the step cap more often, so this column
      mixes speed with success. Reported because it is what a user feels, and
      flagged wherever it disagrees with `step`.
    """
    out: dict = defaultdict(lambda: {"infer": [], "step": [], "sec": []})
    for backbone, cond, task, summary in _walk_campaigns():
        key = (backbone, benchmark_of(task), cond)
        repeat = summary.get("action_repeat") or 1
        for e in summary["episodes"]:
            stats = e.get("model_stats") or {}
            inf = e.get("model_ms_per_infer") or stats.get("model_ms_per_infer")
            if inf:
                out[key]["infer"].append(inf)
                step = (e.get("model_ms_per_env_step")
                        or stats.get("model_ms_per_env_step") or inf / repeat)
                out[key]["step"].append(step)
            if e.get("elapsed"):
                out[key]["sec"].append(e["elapsed"])
    return out


def _mean(v):
    return sum(v) / len(v) if v else None


def cost_section(cost, cols) -> list:
    """Compute saved by each condition, beside what the grid says it costs."""
    L: list[str] = []
    rows = []
    for c in conditions_present({k: 1 for k in cost}):
        if c == "baseline":
            continue
        cells = []
        for b, k in cols:
            base, this = cost.get((b, k, "baseline")), cost.get((b, k, c))
            if not base or not this or not base["step"] or not this["step"]:
                cells.append("--")
                continue
            d = 100.0 * (_mean(this["step"]) - _mean(base["step"])) / _mean(base["step"])
            cells.append(f"{d:+.1f}%")
        if any(x != "--" for x in cells):
            rows.append((DISPLAY.get(c, c), cells))
    if not rows:
        return L

    L.append("\n## What the interventions cost, and what they buy\n")
    L.append("Change in **model time per environment step** against the same "
             "column's baseline. Negative is faster. This is the quantity the "
             "whole premise rests on -- an intervention that does not move it "
             "is not an efficiency intervention, whatever it does to success.\n")
    L.append("| condition | " + " | ".join(f"{b}<br>{k}" for b, k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for name, cells in rows:
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    L.append("\nBaselines, for scale:\n")
    L.append("| | " + " | ".join(f"{b}<br>{k}" for b, k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    base_ms, base_sec = [], []
    for b, k in cols:
        d = cost.get((b, k, "baseline"))
        base_ms.append(f"{_mean(d['step']):.0f} ms" if d and d["step"] else "--")
        base_sec.append(f"{_mean(d['sec']):.1f} s" if d and d["sec"] else "--")
    L.append("| ms / env step | " + " | ".join(base_ms) + " |")
    L.append("| sec / episode | " + " | ".join(base_sec) + " |")

    # Wall-clock, where the harness recorded it. Kept separate from the table
    # above because it is not a cost number: it moves with success too.
    wall = []
    for b, k in cols:
        base = cost.get((b, k, "baseline"))
        if not base or not base["sec"]:
            continue
        for c in conditions_present({key: 1 for key in cost}):
            this = cost.get((b, k, c))
            if c == "baseline" or not this or not this["sec"]:
                continue
            d = 100.0 * (_mean(this["sec"]) - _mean(base["sec"])) / _mean(base["sec"])
            s = 100.0 * (_mean(this["step"]) - _mean(base["step"])) / _mean(base["step"])
            wall.append((b, k, DISPLAY.get(c, c), s, d))
    if wall:
        L.append("\nWall-clock per episode, where the harness recorded it. It "
                 "tracks the compute column except where a condition fails more "
                 "often -- a failed episode runs to the step cap, so its "
                 "seconds go up while its compute per step goes down.\n")
        L.append("| backbone / benchmark | condition | ms/env-step | sec/episode |")
        L.append("|---|---|---|---|")
        for b, k, name, s, d in wall:
            L.append(f"| {b} / {k} | {name} | {s:+.1f}% | {d:+.1f}% |")
    return L


def conditions_present(data) -> list:
    seen = ({k[2] for k in data} | {k[2] for k in LEGACY}) - EXCLUDE
    ordered = [c for c, _ in CONDITION_ORDER if c in seen]
    return ordered + sorted(seen - set(ordered))


def markdown(data) -> str:
    L = []
    cols = [("OpenVLA", "Bridge"), ("OpenVLA", "Fractal"),
            ("SpatialVLA", "Bridge"), ("SpatialVLA", "Fractal"),
            ("UniVLA", "Bridge")]
    cols = [c for c in cols if any(k[0] == c[0] and k[1] == c[1] for k in data)]

    # Derived, not hardcoded: this family is the paired tests this table
    # actually runs, and it grows as conditions are added.
    n_grid = grid_family(data, cols)
    grid_alpha = 0.05 / max(1, n_grid)

    L.append("## The grid\n")
    L.append(f"Success rate, and the paired delta against that column's own "
             f"baseline. `**` clears p<0.05, `***` clears the Bonferroni "
             f"threshold for this family: {n_grid} paired tests, so "
             f"a = 0.05/{n_grid} ~ {grid_alpha:.4f}. Deltas use only episodes "
             f"present in both runs.\n")
    # The legend only appears when there is something to explain. Printing it
    # over a table with no daggers tells the reader to look for a caveat that
    # is not there, which is its own kind of wrong.
    if LEGACY:
        L.append("*Italic with `†`* = measured in an earlier campaign that kept "
                 "no per-episode records. Unpaired, and its delta is against "
                 "that campaign's own baseline, not the one in this table's "
                 "first row -- so it cannot be recomputed from this column and "
                 "cannot carry a claim that turns on its sign. Listed under the "
                 "table.\n")
    else:
        L.append("**Every cell is paired**: measured against this column's own "
                 "baseline on the same episode ids, on the same hardware. No "
                 "cell is carried over unpaired from an earlier campaign.\n")
    L.append("| condition | " + " | ".join(f"{b}<br>{k}" for b, k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for cond in conditions_present(data):
        row = [DISPLAY.get(cond, cond)]
        for b, k in cols:
            row.append(fmt_cell(cell(data, b, k, cond), paired(data, b, k, cond),
                                grid_alpha, LEGACY.get((b, k, cond))))
        L.append("| " + " | ".join(row) + " |")

    if LEGACY:
        L.append("\n† legacy cells, with the baseline each was measured against:\n")
        L.append("| backbone / benchmark | condition | rate | delta | own baseline | source |")
        L.append("|---|---|---|---|---|---|")
        for (b, k, cond), (rate, d, base, src) in sorted(LEGACY.items()):
            if cell(data, b, k, cond):      # superseded by a real run
                continue
            L.append(f"| {b} / {k} | {DISPLAY.get(cond, cond)} | {rate:.1f}% | "
                     f"{d:+.1f} | {base:.1f}% | `{src}` |")

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

    L.extend(cost_section(discover_cost(), cols))

    L.append("\n## Does the effect depend on where you measure it?\n")
    L.append("The per-cell tests above ask whether a condition changed anything "
             "*within* one column. The campaign's claim is different -- that the "
             "same condition acts differently *elsewhere* -- and the two runs "
             "share no episodes, so nothing can be paired. What compares is the "
             "discordant split, and whether it differs is a 2x2 Fisher exact "
             "test. Both cells can individually miss p<0.05 while the difference "
             "between them clears it.\n")
    pairs = []
    backbones = sorted({b for b, _ in cols})
    for bb in backbones:                       # same backbone, two benchmarks
        if (bb, "Bridge") in cols and (bb, "Fractal") in cols:
            pairs.append((f"{bb}: Bridge vs Fractal", (bb, "Bridge"), (bb, "Fractal")))
    # Same benchmark, two backbones. Both benchmarks, not just Bridge: an
    # earlier version hardcoded "Bridge" here, which silently dropped the one
    # backbone pair Fractal has (OpenVLA vs SpatialVLA) and so understated this
    # family by seven tests. The Bonferroni denominator below is len(rows), so
    # a dropped comparison also loosened the threshold.
    for benchmark in ("Bridge", "Fractal"):
        for i, b1 in enumerate(backbones):
            for b2 in backbones[i + 1:]:
                if (b1, benchmark) in cols and (b2, benchmark) in cols:
                    pairs.append((f"{benchmark}: {b1} vs {b2}",
                                  (b1, benchmark), (b2, benchmark)))
    rows = []
    for title, left, right in pairs:
        for cond in conditions_present(data):
            r = interaction(data, cond, left, right)
            if r is not None:
                rows.append((title, cond, r))
    # This family is its own multiple-comparison family, so the threshold is
    # 0.05 / (number of tests actually in this table) -- which grows as the
    # grid fills. Deriving it here is what stops a prose document from
    # quoting a threshold the table no longer uses.
    alpha = 0.05 / max(1, len(rows))
    L.append(f"{len(rows)} tests, so Bonferroni for this family is "
             f"a = 0.05/{len(rows)} ~ {alpha:.4f}; `***` marks the rows that "
             f"clear it, `**` marks p<0.05.\n")
    L.append("| comparison | condition | left (fixed/broke) | right (fixed/broke) | p |")
    L.append("|---|---|---|---|---|")
    for title, cond, (af, ab, bf, bb_, p) in rows:
        star = "***" if p < alpha else "**" if p < 0.05 else ""
        L.append(f"| {title} | {DISPLAY.get(cond, cond)} | {af}/{ab} | "
                 f"{bf}/{bb_} | {p:.4f}{star} |")

    L.extend(task_family_section(data, cols))

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
            # A legacy cell was run; it just has no records. Listing it as
            # "not run" would say the opposite of what is true.
            if not data.get((b, k, cond)) and (b, k, cond) not in LEGACY:
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
