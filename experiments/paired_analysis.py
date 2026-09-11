#!/usr/bin/env python3
"""Paired, per-episode comparison of every Table I and Table II row with its
Original row, from the mentor's episodes.jsonl files.

For each backbone, environment (or LIBERO suite) and family, the variant is
the one the tables show: highest success rate, then fewer average steps (the
mentor's rule), selected from the per-backbone and per-suite CSVs. Episodes are paired by
(task, episode_index); every backbone runs the same episode set in all 14
configurations (seed = 42 + episode_index), checked on arrival.

Reported per cell:
  n            paired episodes
  win / loss   episodes the variant turns from failure to success / from
               success to failure (discordant pairs); the rest are ties
  delta        success-rate change in points, = (win - loss) / n * 100
  p            exact two-sided McNemar test on the discordant pairs, i.e. the
               binomial probability of a split at least this uneven under
               p = 0.5; with no discordant pair the test is undefined (p = 1)
  dsteps       mean paired change in environment steps, with a 95 percent
               bootstrap percentile interval over episodes (2000 resamples,
               fixed seed)
  dtime        mean paired change in episode wall-clock seconds

Outputs experiments/paper/paired_results.csv (one row per cell) and
experiments/paper/PairedResults.md (readable tables). No scipy is needed.
"""
import csv
import glob
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "artifacts", "results", "mentor_2026-09-11")
RES = os.path.join(BASE, "results")
SUMM = os.path.join(BASE, "summarization")
OUT_CSV = os.path.join(HERE, "paper", "paired_results.csv")
OUT_MD = os.path.join(HERE, "paper", "PairedResults.md")

FAMILIES = ["original", "fixed_foveation", "action_repeat", "depth_pruning",
            "guarded_reuse", "temporal_fusion"]
LABEL = {"fixed_foveation": "Foveation", "action_repeat": "Action repeat",
         "depth_pruning": "Depth pruning", "guarded_reuse": "Guarded reuse",
         "temporal_fusion": "Temporal fusion"}

# (summary csv, environment filter or None) -> (results dir by model_name)
SIMPLER = {
    "simpler_widowx": {
        "cogact": "cogact_base_simplerenv_bridge",
        "cronusvla": "cronusvla_simplerenv_bridge",
        "minivla": "minivla_simplerenv",
        "openvla": "openvla_simplerenv",
        "spatialvla": "spatialvla_simplerenv_bridge",
        "univla": "univla_simplerenv_bridge",
    },
    "google_robot_fractal": {
        "cogact": "cogact_base_simplerenv_fractal",
        "cronusvla": "cronusvla_simplerenv_fractal",
        "openvla": "openvla_simplerenv_fractal",
        "spatialvla": "spatialvla_simplerenv_fractal",
    },
}
LIBERO_DIR = {"univla": "univla_libero", "openvla": "openvla_libero",
              "smolvla": "smolvla_libero"}
LIBERO_SUITE = {"Libero Long": "libero_10", "LIBERO-Goal": "libero_goal",
                "LIBERO-Object": "libero_object",
                "LIBERO-Spatial": "libero_spatial"}
NAME = {"cogact": "CogACT", "cronusvla": "CronusVLA", "minivla": "MiniVLA",
        "openvla": "OpenVLA", "spatialvla": "SpatialVLA", "univla": "UniVLA",
        "smolvla": "SmolVLA"}
ENV_LABEL = {"simpler_widowx": "WidowX", "google_robot_fractal": "Fractal",
             "Libero Long": "LIBERO Long", "LIBERO-Goal": "LIBERO Goal",
             "LIBERO-Object": "LIBERO Object", "LIBERO-Spatial": "LIBERO Spatial"}


def episodes(run_dir, suite=None):
    """{(task, episode_index): (success, steps, elapsed_s)}"""
    pattern = os.path.join(run_dir, suite, "episodes.jsonl") if suite \
        else os.path.join(run_dir, "**", "episodes.jsonl")
    out = {}
    for f in glob.glob(pattern, recursive=True):
        for line in open(f):
            if not line.strip():
                continue
            d = json.loads(line)
            out[(d["task"], d["episode_index"])] = (
                bool(d["success"]), d["steps_executed"],
                d["episode_elapsed_ms"] / 1000.0)
    return out


def mcnemar_exact(win, loss):
    n = win + loss
    if n == 0:
        return 1.0
    k = min(win, loss)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def compare(orig, var):
    keys = sorted(set(orig) & set(var))
    win = sum(1 for k in keys if not orig[k][0] and var[k][0])
    loss = sum(1 for k in keys if orig[k][0] and not var[k][0])
    dsteps = [var[k][1] - orig[k][1] for k in keys]
    dtime = [var[k][2] - orig[k][2] for k in keys]
    lo, hi = bootstrap_ci(dsteps)
    return {
        "n": len(keys), "win": win, "loss": loss,
        "delta_pts": (win - loss) / len(keys) * 100,
        "p_mcnemar": mcnemar_exact(win, loss),
        "dsteps": sum(dsteps) / len(dsteps), "dsteps_lo": lo, "dsteps_hi": hi,
        "dtime_s": sum(dtime) / len(dtime),
        "unpaired": len(set(orig) ^ set(var)),
    }


def selected_from(paths, env_name):
    """{(model key, env_name, family): configuration}: highest success, then
    fewer average steps, over the rows of the given per-backbone CSVs."""
    best = {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            bk = next(k for k in NAME if k in r["model_name"])
            fam = next(p for p in FAMILIES if r["configuration"].startswith(p))
            rank = (float(r["success_rate_pct"]), -float(r["avg_steps"]))
            key = (bk, env_name, fam)
            if key not in best or rank > best[key][0]:
                best[key] = (rank, r["configuration"])
    return {k: c for k, (_, c) in best.items()}


def main():
    rows = []
    for env, dirs in SIMPLER.items():
        paths = [p for p in glob.glob(os.path.join(SUMM, env, "*.csv"))
                 if not p.endswith("summary.csv")]
        sel = selected_from(paths, env)
        for bk, rd in dirs.items():
            orig = episodes(os.path.join(RES, rd, "original"))
            for fam in FAMILIES[1:]:
                cfg = sel.get((bk, env, fam))
                if cfg is None:
                    continue
                var = episodes(os.path.join(RES, rd, cfg))
                r = compare(orig, var)
                rows.append(dict(backbone=NAME[bk], env=ENV_LABEL[env],
                                 family=LABEL[fam], configuration=cfg, **r))
    for bk, rd in LIBERO_DIR.items():
        for env, suite in LIBERO_SUITE.items():
            sel = selected_from([os.path.join(SUMM, "libero", f"{bk}_{suite}.csv")], env)
            orig = episodes(os.path.join(RES, rd, "original"), suite)
            for fam in FAMILIES[1:]:
                cfg = sel.get((bk, env, fam))
                if cfg is None:
                    continue
                var = episodes(os.path.join(RES, rd, cfg), suite)
                if not var:
                    continue
                r = compare(orig, var)
                rows.append(dict(backbone=NAME[bk], env=ENV_LABEL[env],
                                 family=LABEL[fam], configuration=cfg, **r))

    fields = ["backbone", "env", "family", "configuration", "n", "win", "loss",
              "delta_pts", "p_mcnemar", "dsteps", "dsteps_lo", "dsteps_hi",
              "dtime_s", "unpaired"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in fields})

    md = ["# Paired per-episode comparison with the Original policy",
          "",
          "Generated by experiments/paired_analysis.py from "
          "artifacts/results/mentor_2026-09-11. Each row is a Table I or Table "
          "II cell: the variant the table shows (highest success, then fewer steps), paired "
          "with Original by (task, episode index). win = episodes turned from "
          "failure to success, loss = the reverse, delta = (win - loss)/n in "
          "points, p = exact two-sided McNemar test on the discordant pairs, "
          "dsteps = mean paired change in environment steps with a 95% "
          "bootstrap interval, dtime = mean paired change in episode "
          "wall-clock seconds. One seed, one run per cell.", ""]
    envs = []
    for r in rows:
        if r["env"] not in envs:
            envs.append(r["env"])
    for env in envs:
        md += [f"## {env}", "",
               "| Backbone | Family | Setting | n | win | loss | delta (pts) | p | dsteps [95% CI] | dtime (s) |",
               "|---|---|---|---:|---:|---:|---:|---:|---|---:|"]
        for r in rows:
            if r["env"] != env:
                continue
            flag = "**" if r["p_mcnemar"] < 0.05 else ""
            md.append(
                f"| {r['backbone']} | {r['family']} | {r['configuration']} | "
                f"{r['n']} | {r['win']} | {r['loss']} | {r['delta_pts']:+.1f} | "
                f"{flag}{r['p_mcnemar']:.3f}{flag} | "
                f"{r['dsteps']:+.1f} [{r['dsteps_lo']:+.1f}, {r['dsteps_hi']:+.1f}] | "
                f"{r['dtime_s']:+.2f} |")
        md.append("")
    sig = [r for r in rows if r["p_mcnemar"] < 0.05]
    md += ["## Cells with p < 0.05", ""]
    for r in sorted(sig, key=lambda r: r["p_mcnemar"]):
        md.append(f"- {r['backbone']} / {r['env']} / {r['family']} "
                  f"({r['configuration']}): {r['delta_pts']:+.1f} pts, "
                  f"win {r['win']} loss {r['loss']}, p = {r['p_mcnemar']:.4f}")
    md.append("")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(md))
    print(f"wrote {os.path.relpath(OUT_CSV, ROOT)} ({len(rows)} cells) and "
          f"{os.path.relpath(OUT_MD, ROOT)}; {len(sig)} cells with p < 0.05")


if __name__ == "__main__":
    main()
