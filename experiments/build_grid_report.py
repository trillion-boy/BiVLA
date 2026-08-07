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
LEGACY = {
    ("SpatialVLA", "Bridge", "foveate_logpolar"):
        (25.0, -7.3, 32.3, "SpatialVLA_Bridge_Grid.md"),
    ("SpatialVLA", "Bridge", "foveate_blur"):
        (30.2, -2.1, 32.3, "SpatialVLA_Bridge_Grid.md"),
    ("SpatialVLA", "Bridge", "depth_prune1"):
        (22.9, -9.4, 32.3, "SpatialVLA_Bridge_Grid.md (1 of 26 layers)"),
    ("UniVLA", "Bridge", "foveate_logpolar"):
        (86.5, +8.3, 78.1, "ChunkExecFoveation_univla.md"),
    ("UniVLA", "Bridge", "foveate_blur"):
        (76.0, -2.1, 78.1, "ChunkExecFoveation_univla.md"),
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


def fmt_cell(c, pr, legacy=None) -> str:
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
    star = "***" if p < 0.003 else "**" if p < 0.05 else ""
    return f"{s}  {d:+.1f}{star}"


def conditions_present(data) -> list:
    seen = {k[2] for k in data} | {k[2] for k in LEGACY}
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
    L.append("*Italic with `†`* = measured in an earlier campaign that kept no "
             "per-episode records. Unpaired, and its delta is against that "
             "campaign's own baseline, not the one in this table's first row -- "
             "so it cannot be recomputed from this column and cannot carry a "
             "claim that turns on its sign. Listed under the table.\n")
    L.append("| condition | " + " | ".join(f"{b}<br>{k}" for b, k in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for cond in conditions_present(data):
        row = [DISPLAY.get(cond, cond)]
        for b, k in cols:
            row.append(fmt_cell(cell(data, b, k, cond), paired(data, b, k, cond),
                                LEGACY.get((b, k, cond))))
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
    for i, b1 in enumerate(backbones):         # same benchmark, two backbones
        for b2 in backbones[i + 1:]:
            if (b1, "Bridge") in cols and (b2, "Bridge") in cols:
                pairs.append((f"Bridge: {b1} vs {b2}", (b1, "Bridge"), (b2, "Bridge")))
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
