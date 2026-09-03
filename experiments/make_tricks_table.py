#!/usr/bin/env python3
"""Table I for the six-family (bag of tricks) paper.

Reads the mentor's two summary.csv files under artifacts/results/mentor_csv/
and writes paper/tabletricks.tex. Layout follows the draft table the team
agreed on 2026-09-03: backbones down the page, one row per intervention
family, WidowX and Google Robot / Fractal side by side, each with success,
per-step latency and mean episode length, and the change against the
original policy of the same backbone and environment in parentheses.

summary.csv is the mentor's selection: for every family the variant with the
highest success rate in that environment. The variants (keep20 / keep50,
repeat 2 / 4, depth 1 / 2 / 4, strict / moderate / aggressive, three fusion
settings) are compared in the later per-family tables, so this table names
the chosen variant only in the .tex comments and in the footnote line.

Latency. The CSV column cycle_median_latency_ms is NOT used. Its definition
differs between harnesses: for OpenVLA, MiniVLA and CronusVLA a cycle is one
environment step, for CogACT, SpatialVLA and UniVLA it is one policy call
including the environment steps that call drives. Under action repeat the
first family falls and the second rises for the same intervention. The one
quantity defined the same way in every harness is wall-clock time per
environment step, avg_episode_time_s * 1000 / avg_steps, and that is what
the "Step time" column reports.

Output: paper/tabletricks.tex  (needs booktabs, multirow, xcolor)
"""
import csv
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_DIR = os.path.join(ROOT, "artifacts", "results", "mentor_csv")
OUT = os.path.join(HERE, "paper", "tabletricks.tex")

ENVS = [
    ("simpler_widowx", "WidowX"),
    ("google_robot_fractal", "Google Fractal"),
]

# Display name, parameter count, and the substring that identifies the
# backbone in the CSV model_name column. Order is the row order of the table,
# largest backbone first. Parameter counts are the full VLA (VLM plus action
# head) as listed by each release; CronusVLA and MiniVLA are the two whose
# checkpoint names carry the LLM size (0.5B) instead, so confirm those two
# with the mentor before camera-ready.
BACKBONES = [
    ("UniVLA",     "8.5B", "univla"),
    ("CogACT",     "7.6B", "cogact"),
    ("OpenVLA",    "7.5B", "openvla"),
    ("SpatialVLA", "4.0B", "spatialvla"),
    ("CronusVLA",  "1.5B", "cronusvla"),
    ("MiniVLA",    "1.4B", "minivla"),
]

# Family label in the table -> prefix of the configuration name in the CSV.
FAMILIES = [
    ("Original",        "original"),
    ("Foveation",       "fixed_foveation"),
    ("Action repeat",   "action_repeat"),
    ("Depth pruning",   "depth_pruning"),
    ("Guarded reuse",   "guarded_reuse"),
    ("Temporal fusion", "temporal_fusion"),
]

VARIANT_SHORT = {
    "fixed_foveation_keep20": "keep 20\\%",
    "fixed_foveation_keep50": "keep 50\\%",
    "action_repeat2": "$k{=}2$",
    "action_repeat4": "$k{=}4$",
    "depth_pruning1": "1 block",
    "depth_pruning2": "2 blocks",
    "depth_pruning4": "4 blocks",
    "guarded_reuse_strict": "strict",
    "guarded_reuse_moderate": "moderate",
    "guarded_reuse_aggressive": "aggressive",
    "temporal_fusion_motion_entropy": "motion-entropy",
    "temporal_fusion_task_aware": "task-aware",
    "temporal_fusion_conservative_adaptive": "conservative",
}


def load(env_key):
    """{backbone_key: {family_prefix: row}} for one environment's summary."""
    path = os.path.join(CSV_DIR, env_key, "summary.csv")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            bk = next(k for _, _, k in BACKBONES if k in row["model_name"])
            fam = next(p for _, p in FAMILIES
                       if row["configuration"].startswith(p))
            if fam in out.setdefault(bk, {}):
                raise SystemExit(f"{path}: two rows for {bk}/{fam}")
            out[bk][fam] = row
    return out


def metrics(row):
    steps = float(row["avg_steps"])
    return {
        "success": float(row["success_rate_pct"]),
        "step_ms": 1000.0 * float(row["avg_episode_time_s"]) / steps,
        "steps": steps,
    }


def delta(v, ref, lower_is_better):
    d = v - ref
    if abs(d) < 0.005:
        return "\\textcolor{gray}{(+0.00)}"
    good = (d < 0) if lower_is_better else (d > 0)
    col = "green!60!black" if good else "red"
    return f"\\textcolor{{{col}}}{{({d:+.2f})}}"


def cell(m, ref, key, lower):
    if m is None:
        return "--"
    if ref is None:  # original row
        return f"{m[key]:.2f}"
    return f"{m[key]:.2f} {{\\scriptsize {delta(m[key], ref[key], lower)}}}"


def main():
    data = {env: load(env) for env, _ in ENVS}
    lines = []
    chosen = []  # (backbone, env label, family, variant)

    for name, params, bk in BACKBONES:
        first = True
        for fam_label, fam in FAMILIES:
            cells = []
            for env, env_label in ENVS:
                rows = data[env].get(bk, {})
                row = rows.get(fam)
                orig = rows.get("original")
                m = metrics(row) if row else None
                ref = metrics(orig) if (orig and fam != "original") else None
                if row and fam != "original":
                    chosen.append((name, env_label, fam_label,
                                   row["configuration"]))
                cells += [
                    cell(m, ref, "success", False),
                    cell(m, ref, "step_ms", True),
                    cell(m, ref, "steps", True),
                ]
            lead = (f"\\multirow{{{len(FAMILIES)}}}{{*}}{{{name}}} & "
                    f"\\multirow{{{len(FAMILIES)}}}{{*}}{{{params}}}"
                    if first else "& ")
            first = False
            lines.append(f"{lead} & {fam_label} & " + " & ".join(cells)
                         + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"

    variant_note = "; ".join(
        f"{b} {e} {f.lower()}: {VARIANT_SHORT[v]}"
        for b, e, f, v in chosen
        if v not in ("fixed_foveation_keep20", "action_repeat2",
                     "depth_pruning1", "guarded_reuse_moderate",
                     "temporal_fusion_motion_entropy"))

    header = f"""%% ---------------------------------------------------------------------------
%% Generated by make_tricks_table.py on {date.today().isoformat()}. Do not
%% edit; regenerate. Input: artifacts/results/mentor_csv/*/summary.csv.
%%
%% Each family row is the variant with the highest success rate in that
%% environment (the mentor's summary.csv selection). The per-family tables
%% later in Results show all variants. Chosen variant per cell:
"""
    for b, e, f, v in chosen:
        header += f"%%   {b:11s} {e:15s} {f:16s} {v}\n"
    header += """%%
%% "Step time" is wall-clock milliseconds per environment step,
%% avg_episode_time_s / avg_steps. The CSV column cycle_median_latency_ms is
%% deliberately not used: it is per environment step in the OpenVLA, MiniVLA
%% and CronusVLA harness and per policy call in the CogACT, SpatialVLA and
%% UniVLA harness, so it moves in opposite directions under action repeat.
%%
%% Needs \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{xcolor}.
%% Spans both columns, so table* and not table.
%% ---------------------------------------------------------------------------
"""

    caption = (
        "SimplerEnv results on WidowX and Google Robot/Fractal at the strongest "
        "setting of each intervention family. Step time is wall-clock "
        "milliseconds per environment step, so it falls under action repeat "
        "even though the cost of one model call does not. Values in "
        "parentheses are the change against the original policy of the same "
        "model and environment. A dash marks a backbone with no checkpoint "
        "for that environment."
    )

    body = "\n".join(lines)
    tex = f"""{header}\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{tab:tricks}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lll rrr rrr@{{}}}}
\\toprule
& & & \\multicolumn{{3}}{{c}}{{WidowX}} & \\multicolumn{{3}}{{c}}{{Google Fractal}} \\\\
\\cmidrule(lr){{4-6}} \\cmidrule(lr){{7-9}}
Model & Params & Policy & Success $\\uparrow$ (\\%) & Step time $\\downarrow$ (ms) & Avg.\\ steps $\\downarrow$
& Success $\\uparrow$ (\\%) & Step time $\\downarrow$ (ms) & Avg.\\ steps $\\downarrow$ \\\\
\\midrule
{body}
\\end{{tabular}}

\\vspace{{2pt}}
{{\\scriptsize Variants other than the default (foveation keep 20\\%, repeat $k{{=}}2$,
depth 1 block, guarded reuse moderate, fusion motion-entropy): {variant_note}.}}
\\end{{table*}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: "
          f"{len(BACKBONES)} backbones x {len(FAMILIES)} rows")


if __name__ == "__main__":
    main()
