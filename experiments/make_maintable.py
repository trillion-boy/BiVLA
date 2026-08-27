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

# Two condition sets.
#
# MAIN is the big Table I after the 2026-08-26 meeting: baseline plus the
# STRONGEST setting of each intervention, one row per family, so the table stays
# four rows per cell as the model list grows. "Strongest" is the largest-effect
# setting of each sweep (repeat 4, four-layer pruning, the log-polar warp), NOT
# the best-scoring one -- this is a paper about configuration moving results, so
# picking the highest number would be the cherry-pick we criticise. Using ONE
# fixed setting across every cell is also what keeps the sign reversal readable:
# depth prune 4 gains on OpenVLA/Fractal and loses on SpatialVLA/Fractal in the
# same row. The per-family sweeps are the evidence for Results 1 and 2 and live
# in the ablation tables, not here.
#
# FULL keeps all eight for the appendix, so no absolute number is lost.
CONDITIONS_MAIN = ["baseline", "action_repeat4",
                   "foveate_logpolar", "depth_prune4"]
CONDITIONS_FULL = ["baseline", "action_repeat2", "action_repeat4",
                   "foveate_logpolar", "foveate_blur",
                   "depth_prune1", "depth_prune2", "depth_prune4"]

BACKBONE_ORDER = ["OpenVLA", "SpatialVLA", "UniVLA"]
BENCH_ORDER = ["Bridge", "Fractal"]


def _add(agg, key, summary):
    rec = agg.setdefault(key, {"ok": 0, "n": 0, "steps": [], "lat": [],
                              "mtep": []})
    repeat = summary.get("action_repeat") or 1
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
        # Model time for the whole episode = (model ms per environment step) x
        # (environment steps). The per-env-step figure is the recorded one when
        # present -- which is the only correct source for UniVLA, whose 5-action
        # chunk means one call covers five steps -- and otherwise infer/repeat,
        # which is exact for the one-action backbones. This is what makes the
        # action-repeat speedup visible: repeat k queries the policy once every
        # k steps, so the per-episode model time falls with k.
        step_ms = (e.get("model_ms_per_env_step")
                   or stats.get("model_ms_per_env_step")
                   or (lat / repeat if lat is not None else None))
        if step_ms is not None and st is not None:
            rec["mtep"].append(step_ms * st)


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
    mtep = mean(rec["mtep"])
    steps = mean(rec["steps"])
    lat_s = "$%.2f$" % (lat / 1000.0) if lat is not None else r"\textemdash"
    mtep_s = "$%.1f$" % (mtep / 1000.0) if mtep is not None else r"\textemdash"
    step_s = "$%.1f$" % steps if steps is not None else r"\textemdash"
    return succ, lat_s, mtep_s, step_s


HEADER = r"""%% ---------------------------------------------------------------------------
%% Generated by make_maintable.py, 2026-08-26 restructure. Do not edit;
%% regenerate. LLMind layout: backbones down the page so more models add rows.
%% All numbers ABSOLUTE; the paired changes and their significance live in the
%% Section IV-B tables.
%%
%% Needs \usepackage{booktabs}, \usepackage{multirow}, \usepackage{graphicx}
%% (for \rotatebox). Spans both columns, so table* and not table.
%%
%% Two time columns answer different questions. "Latency" is model time PER
%% CALL, so it drops with depth pruning but not with action repeat (repeat
%% changes how OFTEN the model is called, not the cost of one call). "Model
%% s/ep" is model time over a whole EPISODE, so it drops with action repeat too
%% and rises when a broken intervention makes episodes run longer. True
%% wall-clock per episode is unusable here: SpatialVLA never recorded it, so two
%% of five cells would be blank.
"""


def emit(agg, conditions, out_name, caption, label="tab:grid", note=""):
    body = []
    for bench in BENCH_ORDER:
        backbones = [b for b in BACKBONE_ORDER
                     if any((bench, b, c) in agg for c in conditions)]
        bench_span = sum(1 for b in backbones
                         for c in conditions if (bench, b, c) in agg)
        first_bench_row = True
        for b in backbones:
            conds = [c for c in conditions if (bench, b, c) in agg]
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
                lead += "& "  # condition column follows
                # "%" is a LaTeX comment, so "20%" would eat the rest of the
                # row. Escape it, and capitalise the condition for the column.
                cond_label = DISPLAY[c].replace("%", r"\%")
                cond_label = cond_label[0].upper() + cond_label[1:]
                body.append(f"{lead}{cond_label} & " + " & ".join(cells) + r" \\")
            body.append(r"\cmidrule(l){2-7}")
        if bench != BENCH_ORDER[-1]:
            body[-1] = r"\midrule"
    while body and body[-1].startswith(r"\cmidrule"):
        body.pop()

    note_tex = ("\n\\vspace{2pt}\n\\parbox{\\textwidth}{\\footnotesize %s}"
                % note if note else "")
    tex = (HEADER + r"""%% ---------------------------------------------------------------------------
\begin{table*}[t]
\centering
\caption{__CAPTION__}
\label{__LABEL__}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{ll l rrrr}
\toprule
& Backbone & Condition & Success (\%) & Latency (s) & Model s/ep & Avg.\ steps \\
\midrule
__BODY__
\bottomrule
\end{tabular}__NOTE__
\end{table*}
""".replace("__BODY__", "\n".join(body))
        .replace("__CAPTION__", caption)
        .replace("__LABEL__", label)
        .replace("__NOTE__", note_tex))

    path = os.path.join(OUT, out_name)
    with open(path, "w") as fh:
        fh.write(tex)
    print("wrote", path, "(%d conditions/cell)" % len(conditions))
    return body


def main():
    agg = collect()

    main_note = (r"Each intervention is shown at its strongest setting "
                 r"(action repeat 4, log-polar foveation, four-layer depth "
                 r"pruning). The per-setting sweeps are in the ablation tables "
                 r"of Section~IV.")
    emit(agg, CONDITIONS_MAIN, "tablemain.tex",
         "Absolute success rate, model latency per call, model time per "
         "episode, and mean episode length, at the strongest setting of each "
         "intervention.", note=main_note)

    # The full sweep, kept for the appendix so no absolute number is lost.
    emit(agg, CONDITIONS_FULL, "tablemain_full.tex",
         "Absolute metrics for every condition in the grid (appendix). The "
         "main text shows the strongest setting of each intervention in "
         "Table~\\ref{tab:grid}.", label="tab:grid-full")


if __name__ == "__main__":
    main()
