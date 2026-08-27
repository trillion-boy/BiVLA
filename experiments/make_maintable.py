#!/usr/bin/env python3
"""Table I -- the big descriptive table, restructured after the 2026-08-26 lab
meeting to follow the LLMind layout: the largest table first, absolute numbers
rather than paired changes, and the backbones listed DOWN the page so the
five-model expansion adds rows instead of columns.

Rows are nested Benchmark > Backbone > Condition. Columns are the three metrics
the meeting asked for, all absolute:

  Success (%)      episode-weighted success rate, pooled over the cell's tasks
  Latency (s)      mean model time per policy call. Recovered per episode where
                   the harness records it, else from the run-level average
                   (OpenVLA logs it once per run, not per episode).
  Avg. steps       mean environment steps per episode

The paired changes and their McNemar/Bonferroni significance -- the argument of
the paper -- are NOT here any more. They move to the small tables in Results,
one per result, where the contrast that carries each claim can be shown on its
own. This table is the reference the reader checks a number against.

Everything is read through build_grid_report._walk_campaigns(), the same walk
the success grid and the cost table use, so no two tables can disagree about
which runs exist or how UniVLA/Bridge's two-card baseline is resolved.

Output: paper/tablemain.tex
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_grid_report import (  # noqa: E402
    _ROOT, _walk_campaigns, benchmark_of, DISPLAY, IMPORTED)

OUT = os.path.join(HERE, "paper")

# The eight conditions every filled cell runs, in reading order. The extra
# depth variants (prune8, prune2_repeat2) some cells also ran stay out of this
# table for the same reason they stayed out of the old one: they exist in two
# cells, not five, and a ragged block reads as missing data.
CONDITIONS = ["baseline", "action_repeat2", "action_repeat4",
              "foveate_logpolar", "foveate_blur",
              "depth_prune1", "depth_prune2", "depth_prune4"]

BACKBONE_ORDER = ["OpenVLA", "SpatialVLA", "UniVLA"]
BENCH_ORDER = ["Bridge", "Fractal"]


def _add(agg, key, summary):
    rec = agg.setdefault(key, {"ok": 0, "n": 0, "steps": [], "lat": []})
    # A run-level latency, used as the per-episode fallback for harnesses
    # (OpenVLA) that record model time once per run instead of per episode.
    run_lat = summary.get("avg_model_ms_per_infer")
    if run_lat is None:
        stats = summary.get("model_stats") or {}
        run_lat = stats.get("model_ms_per_infer")
    for e in summary["episodes"]:
        rec["n"] += 1
        rec["ok"] += 1 if e["success"] else 0
        st = e.get("steps") or e.get("num_steps")
        if st is not None:
            rec["steps"].append(st)
        stats = e.get("model_stats") or {}
        lat = (e.get("model_ms_per_infer")
               or stats.get("model_ms_per_infer") or run_lat)
        if lat is not None:
            rec["lat"].append(lat)


def collect():
    """-> {(bench, backbone, cond): {ok, n, steps:[...], lat:[...]}}."""
    agg = {}
    for backbone, cond, task, summary in _walk_campaigns():
        _add(agg, (benchmark_of(task), backbone, cond), summary)
    # The OpenVLA/Bridge foveation cell lives outside results/ and _walk_
    # campaigns does not see it. discover() splices it into the success grid;
    # we splice it here too, so the descriptive table is not missing a row the
    # paired table has.
    for backbone, bench, cond, rec_dir, _base in IMPORTED:
        for path in glob.glob(os.path.join(_ROOT, rec_dir, "**",
                                           "results_*.json"), recursive=True):
            _add(agg, (bench, backbone, cond), json.load(open(path)))
    return agg


def mean(v):
    return sum(v) / len(v) if v else None


def fmt(rec):
    if not rec or not rec["n"]:
        return None
    succ = "$%.1f$" % (100.0 * rec["ok"] / rec["n"])
    lat = mean(rec["lat"])
    steps = mean(rec["steps"])
    lat_s = "$%.2f$" % (lat / 1000.0) if lat is not None else r"\textemdash"
    step_s = "$%.1f$" % steps if steps is not None else r"\textemdash"
    return succ, lat_s, step_s


def main():
    agg = collect()

    body = []
    for bi, bench in enumerate(BENCH_ORDER):
        backbones = [b for b in BACKBONE_ORDER
                     if any((bench, b, c) in agg for c in CONDITIONS)]
        bench_span = sum(1 for b in backbones
                         for c in CONDITIONS if (bench, b, c) in agg)
        first_bench_row = True
        for b in backbones:
            conds = [c for c in CONDITIONS if (bench, b, c) in agg]
            first_back_row = True
            for c in conds:
                cells = fmt(agg[(bench, b, c)])
                lead = ""
                if first_bench_row:
                    lead += r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{%s}} " % (
                        bench_span, bench)
                    first_bench_row = False
                lead += "& "
                if first_back_row:
                    lead += r"\multirow{%d}{*}{%s} " % (len(conds), b)
                    first_back_row = False
                lead += "& "
                # "%" is a LaTeX comment, so "20%" would eat the rest of the
                # row. Escape it, and capitalise the condition for the column.
                label = DISPLAY[c].replace("%", r"\%")
                label = label[0].upper() + label[1:]
                row = f"{lead}{label} & " + " & ".join(cells) + r" \\"
                body.append(row)
            body.append(r"\cmidrule(l){2-6}")
        if bench != BENCH_ORDER[-1]:
            body[-1] = r"\midrule"

    # A rule right before \bottomrule doubles it; drop the last group separator.
    while body and body[-1].startswith(r"\cmidrule"):
        body.pop()

    for line in body:
        print(line.replace(r"\\", "").replace("&", "|"))

    tex = r"""%% ---------------------------------------------------------------------------
%% Table I -- the big descriptive table. Generated by make_maintable.py.
%% Restructured 2026-08-26 (lab meeting): LLMind layout, absolute metrics, the
%% backbones down the page so more models add rows. The paired changes and
%% their significance moved to the small Results tables. Regenerate; do not edit.
%%
%% Needs \usepackage{booktabs}, \usepackage{multirow}, \usepackage{graphicx}
%% (for \rotatebox). Spans both columns, so table* and not table.
%%
%% Latency is model time PER CALL, so the action-repeat rows read the same as
%% baseline: repeat changes how OFTEN the model is called, not the cost of one
%% call. If the meeting wants the repeat speedup visible in this table, add a
%% "model time per episode" column; that is a design choice, not a data gap.
%% ---------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{Absolute success rate, model latency per call, and mean episode length
for every backbone, benchmark, and condition in the grid.}
\label{tab:grid}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{ll l rrr}
\toprule
& Backbone & Condition & Success (\%) & Latency (s) & Avg.\ steps \\
\midrule
__BODY__
\bottomrule
\end{tabular}
\end{table*}
""".replace("__BODY__", "\n".join(body))

    with open(os.path.join(OUT, "tablemain.tex"), "w") as fh:
        fh.write(tex)
    print("\nwrote", os.path.join(OUT, "tablemain.tex"))


if __name__ == "__main__":
    main()
