#!/usr/bin/env python3
"""Table II (LIBERO) in the Overleaf layout: one row group per task suite
(Long, Goal, Object, Spatial), Policy rows, and one column group per
backbone (UniVLA, OpenVLA, SmolVLA) with Success, Latency, Avg. Steps and the
change from Original in a small coloured parenthesis.

Reads the per-suite CSVs under artifacts/results/mentor_2026-09-11/
summarization/libero/ and selects, for every backbone, suite and family, the
variant with the highest success rate and, on a tie, the fewer average steps,
the rule the mentor fixed on 2026-09-11. The mentor's libero/summary.csv
breaks eight of its ties differently (mostly by lower latency), so it is not
used for the selection; the script prints every cell where the two differ.
Writes paper/tablelibero.tex.

Latency is 1000 * avg_episode_time_s / avg_steps, wall-clock per environment
step, the definition the mentor fixed on 2026-09-08. The Overleaf Table II
of 2026-09-07 carried cycle_median_latency_ms and is replaced by this file.
Values are rounded half-up from the CSV strings; the parenthesis is the raw
change from the suite's Original row rounded half-up; green when the change
is in the arrow's direction or zero, red otherwise. Parameter counts are the
ones the original works report (mentor, 2026-09-11): UniVLA 8.5B, OpenVLA 7B,
SmolVLA 0.45B (450M).

Known gap: openvla_libero temporal_fusion_conservative_adaptive has no
libero_10 (Long) run, so the OpenVLA Long fusion row is whichever of the
other two fusion settings summary.csv selected.
"""
import csv
import os
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_DIR = os.path.join(ROOT, "artifacts", "results", "mentor_2026-09-11",
                       "summarization", "libero")
SUMMARY = os.path.join(CSV_DIR, "summary.csv")
OUT = os.path.join(HERE, "paper", "tablelibero.tex")

# (environment name in summary.csv, row label, per-suite CSV suffix)
SUITES = [
    ("Libero Long", "Long", "libero_10"),
    ("LIBERO-Goal", "Goal", "libero_goal"),
    ("LIBERO-Object", "Object", "libero_object"),
    ("LIBERO-Spatial", "Spatial", "libero_spatial"),
]
BACKBONES = [
    ("UniVLA", "8.5B", "univla"),
    ("OpenVLA", "7B", "openvla"),
    ("SmolVLA", "0.45B", "smolvla"),
]
FAMILIES = [
    ("original", "Original"),
    ("fixed_foveation", "Foveation"),
    ("action_repeat", "Action repeat"),
    ("depth_pruning", "Depth pruning"),
    ("guarded_reuse", "Guarded reuse"),
    ("temporal_fusion", "Temporal fusion"),
]
GOOD = "green!50!black"
BAD = "red"
CENT = Decimal("0.01")


def q(x):
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def load():
    """{(backbone key, environment, family): selected row}, highest success
    then fewer steps, from the per-suite CSVs."""
    out = {}
    for env, _, suffix in SUITES:
        for _, _, bk in BACKBONES:
            path = os.path.join(CSV_DIR, f"{bk}_{suffix}.csv")
            for row in csv.DictReader(open(path)):
                fam = next(p for p, _ in FAMILIES
                           if row["configuration"].startswith(p))
                key = (bk, env, fam)
                rank = (Decimal(row["success_rate_pct"]),
                        -Decimal(row["avg_steps"]))
                if key not in out or rank > out[key][0]:
                    out[key] = (rank, row)
    sel = {k: r for k, (_, r) in out.items()}
    if os.path.exists(SUMMARY):
        for row in csv.DictReader(open(SUMMARY)):
            bk = next(k for _, _, k in BACKBONES if k in row["model_name"])
            fam = next(p for p, _ in FAMILIES
                       if row["configuration"].startswith(p))
            mine = sel.get((bk, row["environment"], fam))
            if mine is not None and mine["configuration"] != row["configuration"]:
                print(f"note: {bk} {row['environment']} {fam}: rule picks "
                      f"{mine['configuration']}, summary.csv has "
                      f"{row['configuration']}")
    return sel


def metrics(row):
    steps = Decimal(row["avg_steps"])
    return (Decimal(row["success_rate_pct"]),
            Decimal(row["avg_episode_time_s"]) * 1000 / steps, steps)


def cell(value, base, higher_is_better):
    if base is None:
        return f"{q(value):.2f}"
    d = q(value - base)
    good = d >= 0 if higher_is_better else d <= 0
    colour = GOOD if good else BAD
    return f"{q(value):.2f} {{\\scriptsize\\textcolor{{{colour}}}{{({d:+.2f})}}}}"


def main():
    data = load()
    blocks = []
    for env, label, _ in SUITES:
        lines = []
        for fi, (fam, fname) in enumerate(FAMILIES):
            head = (f"\\multirow[c]{{{len(FAMILIES)}}}{{*}}"
                    f"{{\\rotatebox[origin=c]{{90}}{{{label}}}}}" if fi == 0 else "")
            cells = []
            for _, _, bk in BACKBONES:
                row = data.get((bk, env, fam))
                base = data.get((bk, env, "original"))
                if row is None or base is None:
                    cells.append("-- & -- & --")
                    continue
                s, l, n = metrics(row)
                bs, bl, bn = metrics(base) if fam != "original" else (None,) * 3
                cells.append(" & ".join([cell(s, bs, True), cell(l, bl, False),
                                         cell(n, bn, False)]))
            lines.append(f"{head} & {fname} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(lines))
    body = "\n\\midrule\n".join(blocks)
    heads = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{n} ({p})}}" for n, p, _ in BACKBONES)
    sub = " & ".join(["Success $\\uparrow$ (\\%) & Latency $\\downarrow$ (ms) & "
                      "Avg. Steps $\\downarrow$"] * len(BACKBONES))
    tex = f"""%% ---------------------------------------------------------------------------
%% Generated by make_libero_table.py on {date.today().isoformat()}. Do not
%% edit; regenerate. Input: the per-suite CSVs under artifacts/results/
%% mentor_2026-09-11/summarization/libero/, one variant per family and suite
%% by highest success then fewer average steps (the mentor's rule).
%% Latency is avg_episode_time_s * 1000 / avg_steps (per environment step),
%% replacing the cycle_median_latency_ms values of the earlier Overleaf table.
%% Parameter counts as reported by the original works. Needs booktabs,
%% multirow, graphicx, xcolor. Spans both columns.
%% ---------------------------------------------------------------------------
\\begin{{table*}}[t]
\\centering
\\caption{{LIBERO results across four task suites. Values in parentheses indicate the change relative to the original policy within the same task suite. Parameter counts are as reported by the original works.}}
\\label{{tab:libero-results}}
\\small
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{cl ccc ccc ccc}}
\\toprule
\\multirow{{2}}{{*}}{{Task}} & \\multirow{{2}}{{*}}{{Policy}} & {heads} \\\\
\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}} \\cmidrule(lr){{9-11}}
& & {sub} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
