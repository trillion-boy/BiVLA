#!/usr/bin/env python3
"""Table I in the exact Overleaf format (tab:simplerenv-results): Model,
Parameters and Policy columns, one row group per backbone, one column group
per SimplerEnv environment, and each cell as the absolute value with the
change from Original in a small coloured parenthesis.

Reads the mentor's two summary.csv files under artifacts/results/mentor_csv/
(the best-success variant per family) and writes paper/tablesimpler.tex.

Latency column: avg_episode_time_s * 1000 / avg_steps, wall-clock per
environment step, as the mentor fixed on 2026-09-08 ("replace
cycle_median_latency_ms with avg_episode_time_s / avg_steps"). The earlier
Overleaf table carried cycle_median_latency_ms and is replaced by this file.

Format follows the author's Overleaf source of 2026-09-08 line for line:
\\small, \\tabcolsep 2.2pt, tabular{cclcccccc}, \\multirow[c]{6}{*} for the
model and its parameter count, cells "52.50 {\\scriptsize\\textcolor{...}
{(+2.50)}}", green!50!black when the change is in the arrow's direction or
zero and red otherwise, \\cmidrule(lr){1-9} between backbones, model order
CogACT, OpenVLA, SpatialVLA, CronusVLA, UniVLA, MiniVLA.

Rounding is half-up on the decimal strings in the CSV (50.845 -> 50.85), the
convention the author's table uses, not Python's float formatting. The
parenthesised change is the difference of the two printed values, so the
arithmetic on the page closes. Table II should follow the same rule when it
is regenerated.

CronusVLA depth pruning rows print "--": the mentor is rerunning those six
settings (three budgets in two environments). Drop the pair from PENDING
when the new CSVs land. UniVLA and MiniVLA have no Fractal checkpoint.

Ties on success within a family (seven in the current CSVs) are broken the
way the mentor's summary.csv breaks them; the rule is not documented and is
on the mentor question list, as are the parameter counts printed here.
"""
import csv
import os
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_DIR = os.path.join(ROOT, "artifacts", "results", "mentor_csv")
OUT = os.path.join(HERE, "paper", "tablesimpler.tex")

ENVS = [
    ("simpler_widowx", "WidowX"),
    ("google_robot_fractal", "Google Fractal"),
]

# (display name, parameter count as printed in Overleaf, CSV substring), in
# the row order of the Overleaf table.
BACKBONES = [
    ("CogACT", "7.6B", "cogact"),
    ("OpenVLA", "7.5B", "openvla"),
    ("SpatialVLA", "4.0B", "spatialvla"),
    ("CronusVLA", "1.5B", "cronusvla"),
    ("UniVLA", "8.5B", "univla"),
    ("MiniVLA", "1.4B", "minivla"),
]

FAMILIES = [
    ("original", "Original"),
    ("fixed_foveation", "Foveation"),
    ("action_repeat", "Action repeat"),
    ("depth_pruning", "Depth pruning"),
    ("guarded_reuse", "Guarded reuse"),
    ("temporal_fusion", "Temporal fusion"),
]

# (backbone key, family prefix) cells printed as "--" until the rerun lands.
PENDING = {("cronusvla", "depth_pruning")}

DASH = "--"
GOOD = "green!50!black"
BAD = "red"
CENT = Decimal("0.01")


def q(x):
    """Half-up rounding to two decimals, on a Decimal."""
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def load(env_key):
    path = os.path.join(CSV_DIR, env_key, "summary.csv")
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            bk = next(k for _, _, k in BACKBONES if k in row["model_name"])
            fam = next(p for p, _ in FAMILIES
                       if row["configuration"].startswith(p))
            if fam in out.setdefault(bk, {}):
                raise SystemExit(f"{path}: two rows for {bk}/{fam}")
            out[bk][fam] = row
    return out


def metrics(row):
    """(success, latency ms, steps) as printed, i.e. rounded half-up."""
    steps = Decimal(row["avg_steps"])
    latency = Decimal(row["avg_episode_time_s"]) * 1000 / steps
    return q(Decimal(row["success_rate_pct"])), q(latency), q(steps)


def cell(value, base, higher_is_better):
    if base is None:
        return f"{value:.2f}"
    d = value - base
    good = d >= 0 if higher_is_better else d <= 0
    colour = GOOD if good else BAD
    return f"{value:.2f} {{\\scriptsize\\textcolor{{{colour}}}{{({d:+.2f})}}}}"


def triple(row, base_row):
    s, l, n = metrics(row)
    if base_row is None:
        return " & ".join(f"{v:.2f}" for v in (s, l, n))
    bs, bl, bn = metrics(base_row)
    return " & ".join([
        cell(s, bs, True),
        cell(l, bl, False),
        cell(n, bn, False),
    ])


def main():
    data = {env: load(env) for env, _ in ENVS}
    blocks = []
    for name, params, bk in BACKBONES:
        lines = [f"\\multirow[c]{{{len(FAMILIES)}}}{{*}}{{{name}}} ",
                 f"& \\multirow[c]{{{len(FAMILIES)}}}{{*}}{{{params}}} "]
        for fi, (fam, label) in enumerate(FAMILIES):
            cells = []
            for env, _ in ENVS:
                rows = data[env].get(bk)
                if rows is None or (bk, fam) in PENDING:
                    cells.append(" & ".join([DASH] * 3))
                    continue
                if fam not in rows:
                    raise SystemExit(f"{env}/{bk}: no row for {fam}")
                base = None if fam == "original" else rows["original"]
                cells.append(triple(rows[fam], base))
            prefix = "& " if fi == 0 else "& & "
            lines.append(f"{prefix}{label} & " + " & ".join(cells) + " \\\\")
        blocks.append("\n".join(lines))
    body = "\n\\cmidrule(lr){1-9}\n\n".join(blocks)
    tex = f"""%% ---------------------------------------------------------------------------
%% Generated by make_simpler_table.py on {date.today().isoformat()}. Do not
%% edit; regenerate. Input: artifacts/results/mentor_csv/*/summary.csv, the
%% mentor's best-success variant per family. Format: the author's Overleaf
%% Table I source of 2026-09-08.
%%
%% Latency is avg_episode_time_s * 1000 / avg_steps, wall-clock per
%% environment step, per the mentor's decision of 2026-09-08. It replaces the
%% cycle_median_latency_ms values of the earlier Overleaf table. Rounding is
%% half-up; the parenthesis is the difference of the two printed values.
%% CronusVLA depth pruning rows are "--" until the mentor's rerun of those six
%% settings arrives. UniVLA and MiniVLA have no Fractal checkpoint.
%% Needs booktabs, multirow, xcolor. Spans both columns.
%% ---------------------------------------------------------------------------
\\begin{{table*}}[t]
\\centering
\\caption{{SimplerEnv results on WidowX and Google Robot/Fractal. Values in parentheses indicate the change relative to the original policy of the same model and environment.}}
\\label{{tab:simplerenv-results}}
\\small
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{cclcccccc}}
\\toprule
\\multirow{{2}}{{*}}{{Model}} & \\multirow{{2}}{{*}}{{Parameters}} & \\multirow{{2}}{{*}}{{Policy}}
& \\multicolumn{{3}}{{c}}{{WidowX}}
& \\multicolumn{{3}}{{c}}{{Google Fractal}} \\\\
\\cmidrule(lr){{4-6}}
\\cmidrule(lr){{7-9}}
& & & Success $\\uparrow$ (\\%) & Latency $\\downarrow$ (ms) & Avg. Steps $\\downarrow$
& Success $\\uparrow$ (\\%) & Latency $\\downarrow$ (ms) & Avg. Steps $\\downarrow$ \\\\
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
