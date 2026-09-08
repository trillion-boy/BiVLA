#!/usr/bin/env python3
"""Table I in the Overleaf layout (tab:simplerenv-results): one row group per
backbone, one column group per SimplerEnv environment, and each cell as the
absolute value with the change from Original in parentheses.

Reads the mentor's two summary.csv files under artifacts/results/mentor_csv/
(the best-success variant per family) and writes paper/tablesimpler.tex.

Latency column: avg_episode_time_s * 1000 / avg_steps, wall-clock per
environment step, as the mentor fixed on 2026-09-08 ("replace
cycle_median_latency_ms with avg_episode_time_s / avg_steps"). The earlier
Overleaf table carried cycle_median_latency_ms, which is per call on some
harnesses, and is replaced by this file.

CronusVLA depth pruning rows print "--": the mentor is rerunning those six
settings (three budgets in two environments) and will send new CSVs.

Colours: \\up{} and \\dn{} wrap the parenthesised change, green when the
change is in the arrow's direction or zero, red otherwise. Both are
\\providecommand, so an existing definition in main.tex wins.

The parenthesised change is the difference of the two printed (rounded)
values, not of the raw values, so the arithmetic on the page closes. Table II
should follow the same rule when it is regenerated.

Ties on success within a family (seven in the current CSVs) are broken the
way the mentor's summary.csv breaks them; the rule is not documented and is
on the mentor question list.
"""
import csv
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_DIR = os.path.join(ROOT, "artifacts", "results", "mentor_csv")
OUT = os.path.join(HERE, "paper", "tablesimpler.tex")

ENVS = [
    ("simpler_widowx", "WidowX"),
    ("google_robot_fractal", "Google Fractal"),
]

# (display name, parameter count as printed in Overleaf, CSV substring)
BACKBONES = [
    ("UniVLA", "8.5B", "univla"),
    ("CogACT", "7.6B", "cogact"),
    ("OpenVLA", "7.5B", "openvla"),
    ("SpatialVLA", "4.0B", "spatialvla"),
    ("CronusVLA", "1.5B", "cronusvla"),
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
    steps = float(row["avg_steps"])
    return (
        float(row["success_rate_pct"]),
        1000.0 * float(row["avg_episode_time_s"]) / steps,
        steps,
    )


def cell(value, base, higher_is_better):
    if base is None:
        return f"{value:.2f}"
    # The change is the difference of the two printed values, so a reader
    # who subtracts the numbers on the page recovers the parenthesis exactly.
    d = round(value, 2) - round(base, 2)
    d = 0.0 if abs(d) < 0.005 else d
    good = d >= 0 if higher_is_better else d <= 0
    macro = "up" if good else "dn"
    return f"{value:.2f} \\{macro}{{{d:+.2f}}}"


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
    lines = []
    for bi, (name, params, bk) in enumerate(BACKBONES):
        for fi, (fam, label) in enumerate(FAMILIES):
            head = (f"\\multirow{{{len(FAMILIES)}}}{{*}}"
                    f"{{\\rotatebox[origin=c]{{90}}{{{name} ({params})}}}}"
                    if fi == 0 else "")
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
            lines.append(f"{head} & {label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if bi < len(BACKBONES) - 1
                     else "\\bottomrule")
    body = "\n".join(lines)
    tex = f"""%% ---------------------------------------------------------------------------
%% Generated by make_simpler_table.py on {date.today().isoformat()}. Do not
%% edit; regenerate. Input: artifacts/results/mentor_csv/*/summary.csv, the
%% mentor's best-success variant per family.
%%
%% Latency is avg_episode_time_s * 1000 / avg_steps, wall-clock per
%% environment step, per the mentor's decision of 2026-09-08. It replaces the
%% cycle_median_latency_ms values of the earlier Overleaf table.
%% CronusVLA depth pruning rows are "--" until the mentor's rerun of those six
%% settings arrives. UniVLA and MiniVLA have no Fractal checkpoint.
%% Needs booktabs, multirow, graphicx, xcolor. Spans both columns.
%% ---------------------------------------------------------------------------
\\providecommand{{\\up}}[1]{{\\textcolor{{green!60!black}}{{(#1)}}}}
\\providecommand{{\\dn}}[1]{{\\textcolor{{red}}{{(#1)}}}}
\\begin{{table*}}[t]
\\centering
\\caption{{SimplerEnv results on WidowX and Google Robot/Fractal. Values in parentheses indicate the change relative to the original policy of the same model and environment.}}
\\label{{tab:simplerenv-results}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{cl rrr rrr}}
\\toprule
\\multirow{{2}}{{*}}{{Model}} & \\multirow{{2}}{{*}}{{Policy}} & \\multicolumn{{3}}{{c}}{{WidowX}} & \\multicolumn{{3}}{{c}}{{Google Fractal}} \\\\
\\cmidrule(lr){{3-5}} \\cmidrule(lr){{6-8}}
& & Success $\\uparrow$ (\\%) & Latency $\\downarrow$ (ms) & Avg.\\ Steps $\\downarrow$ & Success $\\uparrow$ (\\%) & Latency $\\downarrow$ (ms) & Avg.\\ Steps $\\downarrow$ \\\\
\\midrule
{body}
\\end{{tabular}}
\\end{{table*}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(lines)} lines")


if __name__ == "__main__":
    main()
