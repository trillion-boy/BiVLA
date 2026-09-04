#!/usr/bin/env python3
"""Table I for the six-family (bag of tricks) paper, in the tablemain.tex
layout: Benchmark > Backbone > Condition down the page, four absolute
columns, no paired changes.

Reads the mentor's two summary.csv files under artifacts/results/mentor_csv/
and writes paper/tabletricks.tex.

summary.csv is the mentor's selection: for every family the variant with the
highest success rate in that environment. The condition label names the
chosen variant ("Depth prune 2", "Foveation keep 50%"), so the reader sees
which setting a row is without a footnote. The per-setting sweeps are the
later per-family tables.

Columns, all absolute:

  Success (%)     success_rate_pct
  Step time (ms)  avg_episode_time_s * 1000 / avg_steps. Wall-clock per
                  environment step, end to end, the cost of an intervention's
                  own signals included. The one latency defined the same way
                  in every harness. Falls under action repeat on all six
                  backbones (28 to 44 percent at k = 2).
  Per call (ms)   policy_median_latency_ms. Model time for one call. Falls
                  with depth pruning, unchanged under action repeat, so the
                  pair shows whether a saving came from cheaper calls or
                  fewer calls.
  Avg. steps      avg_steps, environment steps per episode.

Not used: cycle_median_latency_ms. Its definition differs between harnesses
(per environment step for OpenVLA, MiniVLA, CronusVLA; per policy call
including the repeated steps for CogACT, SpatialVLA, UniVLA), so under
action repeat it falls for the first three and rises for the other three.
Also, under repeat the per-step distribution is bimodal (most steps near
zero, one in k at full cost), so neither its median nor its p95 describes a
step; MiniVLA repeat 4 has median 7.7 ms and p95 106 ms. The mean, episode
time over steps, is the only summary that is a cost per step.

Output: paper/tabletricks.tex  (needs booktabs, multirow, graphicx)
"""
import csv
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_DIR = os.path.join(ROOT, "artifacts", "results", "mentor_csv")
OUT = os.path.join(HERE, "paper", "tabletricks.tex")

# (csv folder, label in the rotated first column)
ENVS = [
    ("simpler_widowx", "Bridge"),
    ("google_robot_fractal", "Fractal"),
]

# Row order within a benchmark, largest backbone first. The substring
# identifies the backbone in the CSV model_name column.
BACKBONES = [
    ("UniVLA", "univla"),
    ("CogACT", "cogact"),
    ("OpenVLA", "openvla"),
    ("SpatialVLA", "spatialvla"),
    ("CronusVLA", "cronusvla"),
    ("MiniVLA", "minivla"),
]

# Family order within a backbone, and the CSV configuration prefix.
FAMILIES = [
    "original",
    "fixed_foveation",
    "action_repeat",
    "depth_pruning",
    "guarded_reuse",
    "temporal_fusion",
]

# Condition label per CSV configuration name.
LABEL = {
    "original": "Original policy",
    "fixed_foveation_keep20": "Foveation keep 20\\%",
    "fixed_foveation_keep50": "Foveation keep 50\\%",
    "action_repeat2": "Action repeat 2",
    "action_repeat4": "Action repeat 4",
    "depth_pruning1": "Depth prune 1",
    "depth_pruning2": "Depth prune 2",
    "depth_pruning4": "Depth prune 4",
    "guarded_reuse_strict": "Guarded reuse strict",
    "guarded_reuse_moderate": "Guarded reuse moderate",
    "guarded_reuse_aggressive": "Guarded reuse aggressive",
    "temporal_fusion_motion_entropy": "Temporal fusion motion-entropy",
    "temporal_fusion_task_aware": "Temporal fusion task-aware",
    "temporal_fusion_conservative_adaptive": "Temporal fusion conservative",
}


def load(env_key):
    """{backbone_key: {family_prefix: row}} for one environment's summary."""
    path = os.path.join(CSV_DIR, env_key, "summary.csv")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            bk = next(k for _, k in BACKBONES if k in row["model_name"])
            fam = next(p for p in FAMILIES
                       if row["configuration"].startswith(p))
            if fam in out.setdefault(bk, {}):
                raise SystemExit(f"{path}: two rows for {bk}/{fam}")
            out[bk][fam] = row
    return out


def fmt_row(row):
    steps = float(row["avg_steps"])
    step_ms = 1000.0 * float(row["avg_episode_time_s"]) / steps
    per_call_ms = float(row["policy_median_latency_ms"])
    return (f"${float(row['success_rate_pct']):.1f}$ & ${step_ms:.0f}$ & "
            f"${per_call_ms:.0f}$ & ${steps:.1f}$")


def main():
    lines = []
    for ei, (env, env_label) in enumerate(ENVS):
        data = load(env)
        present = [(n, k) for n, k in BACKBONES if k in data]
        n_rows = len(present) * len(FAMILIES)
        env_cell = (f"\\multirow{{{n_rows}}}{{*}}"
                    f"{{\\rotatebox[origin=c]{{90}}{{{env_label}}}}}")
        for bi, (name, bk) in enumerate(present):
            rows = data[bk]
            missing = [f for f in FAMILIES if f not in rows]
            if missing:
                raise SystemExit(f"{env}/{bk}: no row for {missing}")
            for fi, fam in enumerate(FAMILIES):
                row = rows[fam]
                c1 = env_cell if (bi == 0 and fi == 0) else ""
                c2 = (f"\\multirow{{{len(FAMILIES)}}}{{*}}{{{name}}}"
                      if fi == 0 else "")
                lines.append(f"{c1} & {c2} & {LABEL[row['configuration']]} & "
                             f"{fmt_row(row)} \\\\")
            if bi < len(present) - 1:
                lines.append("\\cmidrule(l){2-7}")
        lines.append("\\midrule" if ei < len(ENVS) - 1 else "\\bottomrule")

    body = "\n".join(lines)
    tex = f"""%% ---------------------------------------------------------------------------
%% Generated by make_tricks_table.py on {date.today().isoformat()}. Do not
%% edit; regenerate. Input: artifacts/results/mentor_csv/*/summary.csv, the
%% mentor's best-variant-per-family selection. The condition label names the
%% chosen variant.
%%
%% "Step time" is wall-clock per environment step, avg_episode_time_s over
%% avg_steps, end to end with each intervention's own signal cost included. It
%% is the one latency defined identically in every harness and it falls under
%% action repeat on all six backbones. "Per call" is policy_median_latency_ms,
%% model time for one call, unchanged under action repeat and lower under depth
%% pruning, so the pair separates cheaper calls from fewer calls.
%% cycle_median_latency_ms is NOT used: per environment step in one harness,
%% per policy call including repeated steps in the other, so it rises under
%% action repeat for CogACT, SpatialVLA and UniVLA.
%%
%% MiniVLA and UniVLA have no Fractal checkpoint and appear under Bridge only.
%% Needs booktabs, multirow, graphicx. Spans both columns: table*, not table.
%% ---------------------------------------------------------------------------
\\begin{{table*}}[t]
\\centering
\\caption{{Absolute success rate, wall-clock time per environment step, model time per call, and mean episode length, at the strongest setting of each intervention family.}}
\\label{{tab:grid}}
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{ll l rrrr}}
\\toprule
& Backbone & Condition & Success (\\%) & Step time (ms) & Per call (ms) & Avg.\\ steps \\\\
\\midrule
{body}
\\end{{tabular}}
\\vspace{{2pt}}
\\parbox{{\\textwidth}}{{\\footnotesize Each family is shown at the setting with the highest success rate in that benchmark, named in the condition column. Step time is end-to-end wall-clock per environment step, so it falls under action repeat and guarded reuse even though the per-call model time does not. The per-setting sweeps are in the ablation tables of Section~IV.}}
\\end{{table*}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(lines)} lines")


if __name__ == "__main__":
    main()
