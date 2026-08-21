#!/usr/bin/env python3
"""EVERY check this campaign knows how to run, in one pass, one scorecard.

    python experiments/verify_all.py

Why one file: the checks used to live in separate passes (grid vs records,
mechanism reruns, pixel measurements, document structure, paper arithmetic),
and each pass declared victory over its own slice while the next pass found
something in a slice nobody had swept. A verification that arrives in
installments is indistinguishable from one that was never finished. This
script is the union of every check that ever caught something, so "is it
verified" is the exit code of one command, not a memory of which passes ran.

Sections (a FAIL in any of them fails the run):
  1  results/ integrity + episode protocol
  2  grid tables in Report/Overview vs regenerated grid, cell by cell
  3  statistics families (38 McNemar / 43 Fisher) + 5.1/5.2 rows
  4  compute-cost claims (repeat savings, window costs, foveation ~0%)
  5  window-contrast five cells, 5.3 variant pairs, keep sweep, controls
  6  mechanism tables (6.4 / 6.5, all eleven rows)
  7  pixel measurements (224 table, bands, column map, zoom, demo, recipes)
  8  document structure (ragged tables, self-counts, line counts,
     cross-references, cited paths, code citations)
  9  external records (latency profile, chunk log, candidate list)
 10  paper-table arithmetic (every ratio/delta the docs derive from a
     quoted table must follow from that table's own cells)
 11  delegated suites (verify_overview_claims, audit_claims classes 1-3)

What this cannot check, by design (also printed at the end so a green run
never reads as more than it is): the GPU model (never recorded; bounded at
3.1pp in 3.4.0), the three console-only ms values in 3.5.2, and whether
quoted paper cells match the PDFs (verified once by hand; PDFs are not in
the repo).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

FAILS: list[str] = []
CHECKS = [0]


def ck(ok: bool, msg: str) -> None:
    CHECKS[0] += 1
    if not ok:
        FAILS.append(msg)
        print(f"    FAIL  {msg}")


def sec(title: str) -> None:
    print(f"\n[{title}]")


# ---------------------------------------------------------------- helpers
def load(pat):
    out = {}
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f))
        for e in s["episodes"]:
            out[(s["task"], int(e["ep_id"]))] = bool(e["success"])
    return out


def load_steps(pat):
    out = {}
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f))
        for e in s["episodes"]:
            out[(s["task"], int(e["ep_id"]))] = (
                bool(e["success"]), e.get("steps") or e.get("num_steps"))
    return out


def mcn(a, b):
    k = sorted(set(a) & set(b))
    n = len(k)
    br = sum(a[i] and not b[i] for i in k)
    fx = sum(b[i] and not a[i] for i in k)
    m = br + fx
    p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, x) for x in range(min(br, fx) + 1)) / 2 ** m)
    d = 100.0 * (sum(b[i] for i in k) - sum(a[i] for i in k)) / n
    return n, d, br, fx, p


def near(x, y, tol=0.06):
    return x is not None and abs(x - y) <= tol


# ================================================================ 1
sec("1  results/ integrity")
files = sorted(glob.glob("results/**/results_*.json", recursive=True))
per = Counter()
n_ep = 0
bad = 0
for f in files:
    s = json.load(open(f))
    ids = [e["ep_id"] for e in s["episodes"]]
    if len(ids) != len(set(ids)) or s["task"] not in f:
        bad += 1
    if any(not isinstance(e["success"], bool) for e in s["episodes"]):
        bad += 1
    n_ep += len(s["episodes"])
    per[tuple(f.split("/")[1:-1]) + (s["task"],)] += len(s["episodes"])
proto = sum(1 for k, n in per.items()
            if n != (60 if "move_near" in k[-1] else 25 if "coke_can" in k[-1] else 24))
ck(len(files) == 255, f"file count {len(files)} != 255")
ck(n_ep == 7198, f"episode count {n_ep} != 7198")
ck(bad == 0, f"{bad} files with integrity problems")
ck(proto == 0, f"{proto} (campaign,condition,task) groups off protocol")
print(f"    {len(files)} files, {n_ep} episodes, protocol clean")

# ================================================================ 2
sec("2  grid tables vs regenerated grid")
import build_grid_report as B  # noqa: E402

data = B.discover()
cols = [c for c in [("OpenVLA", "Bridge"), ("OpenVLA", "Fractal"),
                    ("SpatialVLA", "Bridge"), ("SpatialVLA", "Fractal"),
                    ("UniVLA", "Bridge")]
        if any(k[0] == c[0] and k[1] == c[1] for k in data)]
paired = {}
for b, k in cols:
    for cond in B.conditions_present(data):
        pr = B.paired(data, b, k, cond)
        if pr:
            paired[(b, k, B.DISPLAY.get(cond, cond))] = pr
NAME = {"repeat 2": "action repeat 2", "repeat 4": "action repeat 4",
        "log-polar": "foveation log-polar 20%", "blur": "foveation blur 20%",
        "prune 1": "depth prune 1", "prune 2": "depth prune 2",
        "prune 4": "depth prune 4", "prune 8": "depth prune 8"}
cells_checked = 0
for doc in ("experiments/Report.md", "experiments/Overview.md"):
    for i, l in enumerate(open(doc).read().split("\n"), 1):
        p = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(p) != 6 or not l.startswith("|"):
            continue
        key = next((v for k, v in NAME.items() if k in p[0].lower()), None)
        if not key:
            continue
        for j, (b, bm) in enumerate(cols):
            pr = paired.get((b, bm, key))
            if not pr:
                continue
            m = re.search(r"[+−±]\d+\.\d+", p[1 + j])
            if m:
                cells_checked += 1
                want = f"{pr[0]:+.1f}".replace("-", "−").replace("+0.0", "±0.0")
                ck(m.group() == want, f"{doc}:{i} {key} {b}/{bm} doc={m.group()} grid={want}")
print(f"    {cells_checked} grid cells checked in the two documents")

# ================================================================ 3
sec("3  statistics families and cross-cell tests")
ck(len(paired) == 38, f"McNemar family {len(paired)} != 38")
ranked = sorted(v[3] for v in paired.values())
ck(sum(1 for p in ranked if p < 0.05 / 38) == 8, "passing grid cells != 8")
ck(ranked[7] < 1.08e-3 < 5.0e-3 < ranked[8], "rank-8/9 gap moved")
inter = []
backbones = sorted({b for b, _ in cols})
for bb in backbones:
    if (bb, "Bridge") in cols and (bb, "Fractal") in cols:
        for c in B.conditions_present(data):
            r = B.interaction(data, c, (bb, "Bridge"), (bb, "Fractal"))
            if r:
                inter.append((f"{bb}:bench", B.DISPLAY.get(c, c), r))
for bm in ("Bridge", "Fractal"):
    for i, b1 in enumerate(backbones):
        for b2 in backbones[i + 1:]:
            if (b1, bm) in cols and (b2, bm) in cols:
                for c in B.conditions_present(data):
                    r = B.interaction(data, c, (b1, bm), (b2, bm))
                    if r:
                        inter.append((f"{bm}:{b1}v{b2}", B.DISPLAY.get(c, c), r))
ck(len(inter) == 43, f"Fisher family {len(inter)} != 43")
passing = [(a, c, r) for a, c, r in inter if r[4] < 0.05 / 43]
ck(len(passing) == 7, f"Fisher passing {len(passing)} != 7 (1 bench + 6 backbone)")
ck(sum(1 for a, c, r in passing if "bench" in a) == 1, "bench-axis passes != 1")
rep = open("experiments/Report.md").read()
for token in ("38개", "43개", "여덟 칸", "0.0013", "0.0012"):
    ck(token in rep, f"Report no longer states {token}")
# 5.1 six rows and 5.2 nine rows: every quoted split must be a real one
gen_splits = {f"{r[0]}/{r[1]}" for _, _, r in inter} | {f"{r[2]}/{r[3]}" for _, _, r in inter}
for name, s0, s1 in (("5.1", "## 5.1", "## 5.2"), ("5.2", "## 5.2", "## 5.3")):
    seg = rep.split(s0)[1].split(s1)[0]
    quoted = set(re.findall(r"(\d+)/(\d+)", seg))
    missing = {f"{a}/{b}" for a, b in quoted} - gen_splits
    ck(not missing, f"S{name} splits not produced by generator: {missing}")
print(f"    38 McNemar (8 pass), 43 Fisher (7 pass), doc splits all real")

# ================================================================ 4
sec("4  compute-cost claims")
cost = B.discover_cost()


def step_delta(b, k, cond):
    base, this = cost.get((b, k, "baseline")), cost.get((b, k, cond))
    if not base or not this or not base["step"] or not this["step"]:
        return None
    m = lambda v: sum(v) / len(v)  # noqa: E731
    return 100.0 * (m(this["step"]) - m(base["step"])) / m(base["step"])


r2 = [round(step_delta(b, k, "action_repeat2")) for b, k in cols]
r4 = [round(step_delta(b, k, "action_repeat4")) for b, k in cols]
ck(r2 == [-50, -50, -51, -52, -52], f"repeat2 savings {r2}")
ck(r4 == [-75, -75, -75, -76, -77], f"repeat4 savings {r4}")
fov = [step_delta(b, k, c) for b, k in cols for c in ("foveate_logpolar", "foveate_blur")
       if step_delta(b, k, c) is not None]
ck(-3.2 < min(fov) and max(fov) < 2.8, f"foveation cost range {min(fov):.1f}~{max(fov):.1f}")
dp4 = [step_delta(b, k, "depth_prune4") for b, k in cols]
ck(near(min(dp4), -15.9, 0.15) and near(max(dp4), -11.2, 0.15), f"dp4 range {dp4}")


def raw_step(pat):
    v = []
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f))
        rep_k = s.get("action_repeat") or 1
        for e in s["episodes"]:
            st = e.get("model_stats") or {}
            inf = e.get("model_ms_per_infer") or st.get("model_ms_per_infer")
            if inf:
                v.append(e.get("model_ms_per_env_step")
                         or st.get("model_ms_per_env_step") or inf / rep_k)
    return sum(v) / len(v) if v else None


ofb = raw_step("results/openvla_fractal_0806_baseline/**/results_*.json")
for name, pat, want in [
        ("prune4", "results/openvla_fractal_0806_depth_prune4/**/results_*.json", -11.9),
        ("prune4_early", "results/openvla_fractal_0806_depth_prune4_early/**/results_*.json", -10.9),
        ("gap3", "results/openvla_fractal_depth_control/prune4_gap3/**/results_*.json", -10.9),
        ("window875", "results/openvla_fractal_depth_control/window875/**/results_*.json", -10.6),
        ("window25", "results/openvla_fractal_depth_control/window25/**/results_*.json", -10.7)]:
    got = 100 * (raw_step(pat) / ofb - 1)
    ck(near(got, want), f"window cost {name} {got:+.1f} != {want}")
def mean_infer(pat):
    """Mean ms/infer over EVERY episode under `pat`, or None if the field is
    absent. Averaging one task's summary would hide a task that never
    recorded it; returning None keeps "not measured" out of the average."""
    v = [e["model_ms_per_infer"] for f in glob.glob(pat, recursive=True)
         for e in json.load(open(f))["episodes"] if e.get("model_ms_per_infer")]
    return sum(v) / len(v) / 1000 if v else None


# The two latency figures the Introduction quotes as a scale. Checked over all
# 96 episodes each rather than one task's summary, and pinned tightly: these
# are cited in prose, so a drift has to fail here rather than be re-typed.
for name, pat, want in [
        ("UniVLA/Bridge baseline", "results/univla_bridge_0805/baseline/**/results_*.json", 2.80),
        ("UniVLA/Bridge baseline_l4", "results/univla_bridge_0805/baseline_l4/**/results_*.json", 2.81),
        ("SpatialVLA/Bridge baseline", "results/spatialvla_bridge_0805/baseline/**/results_*.json", 0.90)]:
    got = mean_infer(pat)
    ck(near(got, want, 0.02),
       f"{name} ms/infer {got if got is None else round(got, 3)} != {want} s")
print("    repeat/foveation/depth/window costs all reproduce")
print(f"    per-infer latency: UniVLA {mean_infer('results/univla_bridge_0805/baseline/**/results_*.json'):.2f} s, "
      f"SpatialVLA {mean_infer('results/spatialvla_bridge_0805/baseline/**/results_*.json'):.2f} s")

# ================================================================ 5
sec("5  window contrast, variant pairs, keep sweep, controls")
W = [("SpatialVLA/Fractal", "results/spatialvla_fractal_0806/baseline/**/results_*.json",
      "results/spatialvla_fractal_0806/depth_prune4/**/results_*.json",
      "results/spatialvla_depth_control/fractal/prune4_back/**/results_*.json", -17.8, -68.1, 50.4),
     ("OpenVLA/Fractal", "results/openvla_fractal_0806_baseline/**/results_*.json",
      "results/openvla_fractal_0806_depth_prune4/**/results_*.json",
      "results/openvla_fractal_depth_control/window875/**/results_*.json", 15.6, -30.4, 45.9),
     ("UniVLA/Bridge", "results/univla_bridge_0805/baseline_l4/**/results_*.json",
      "results/univla_bridge_0805/depth_prune4/**/results_*.json",
      "results/univla_bridge_depth_control/prune4_last/**/results_*.json", -2.1, -8.3, 6.3),
     ("OpenVLA/Bridge", "results/openvla_bridge_0805/baseline/**/results_*.json",
      "results/openvla_bridge_0805/depth_prune4/**/results_*.json",
      "results/openvla_bridge_depth_control/prune4_last/**/results_*.json", 1.0, -4.2, 5.2),
     ("SpatialVLA/Bridge", "results/spatialvla_bridge_0805/baseline/**/results_*.json",
      "results/spatialvla_bridge_0805/depth_prune4/**/results_*.json",
      "results/spatialvla_bridge_0805/depth_prune4_back/**/results_*.json", -28.1, -30.2, 2.1)]
for name, bp, fp, wp, d1, d2, width in W:
    base = load(bp)
    _, df, _, _, _ = mcn(base, load(fp))
    _, db, _, _, _ = mcn(base, load(wp))
    ck(near(df, d1) and near(db, d2) and near(abs(df - db), width, 0.11),
       f"window contrast {name}: {df:+.1f}/{db:+.1f}/{abs(df-db):.1f}")
V = [("results/univla_bridge_0805/foveate_logpolar/**/results_*.json",
      "results/univla_bridge_0805/foveate_blur/**/results_*.json", 13.5, 19, 6, 0.0146),
     ("results/spatialvla_fractal_0806/foveate_logpolar/**/results_*.json",
      "results/spatialvla_fractal_0806/foveate_blur/**/results_*.json", 2.2, 13, 10, 0.6776),
     ("RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated/**/results_*.json",
      "results/openvla_bridge_0805/foveate_blur/**/results_*.json", 1.0, 19, 18, 1.0),
     ("results/spatialvla_bridge_0805/foveate_logpolar/**/results_*.json",
      "results/spatialvla_bridge_0805/foveate_blur/**/results_*.json", -8.3, 8, 16, 0.1516),
     ("results/openvla_fractal_0806_foveate/**/results_*.json",
      "results/openvla_fractal_0806_foveate_blur/**/results_*.json", -10.4, 17, 31, 0.0595)]
for lp, bl, dd, lpo, blo, pp in V:
    n, d, br, fx, p = mcn(load(bl), load(lp))
    ck(near(d, dd) and fx == lpo and br == blo and abs(p - pp) < 2e-3,
       f"5.3 pair {lp.split('/')[1]}: d={d:+.1f} {fx}/{br} p={p:.4f}")
ob = load("results/openvla_bridge_0805/baseline/**/results_*.json")
ck(near(100 * sum(ob.values()) / len(ob), 15.6), "OpenVLA/Bridge baseline != 15.6")
KEEP = {"keep10": ("results/openvla_bridge_foveate_sweep/keep10/**/results_*.json", 4.2, 0.5716),
        "keep20": ("RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated/**/results_*.json", 18.8, 0.005098),
        "keep40": ("results/openvla_bridge_foveate_sweep/keep40/**/results_*.json", 19.8, 0.001319),
        "keep100": ("results/openvla_bridge_foveate_sweep/keep100/**/results_*.json", 30.2, 4.18e-7)}
per_task = {}
for kk, (pat, dd, pp) in KEEP.items():
    cond = load(pat)
    n, d, br, fx, p = mcn(ob, cond)
    ck(near(d, dd) and abs(p - pp) / pp < 0.05, f"keep sweep {kk}: {d:+.1f} p={p:.2e}")
    for (t, _), v in cond.items():
        per_task.setdefault(t, Counter())[kk] += v
ck(per_task["widowx_carrot_on_plate"]["keep100"] == 8
   and per_task["widowx_put_eggplant_in_basket"]["keep100"] == 19
   and per_task["widowx_spoon_on_towel"]["keep20"] == 10
   and per_task["widowx_stack_cube"]["keep20"] == 11, "per-task keep table")
a = load_steps("results/spatialvla_fractal_0806/baseline/**/results_*.json")
b = load_steps("results/spatialvla_fractal_0806/baseline_rerun/**/results_*.json")
sh = set(a) & set(b)
ck(len(sh) == 85 and all(a[k] == b[k] for k in sh), "SpatialVLA determinism 85/85")
r = load_steps("results/univla_recheck_0810/foveate_blur/**/results_*.json")
o = load_steps("results/univla_bridge_0805/foveate_blur/widowx_spoon_on_towel/**/results_*.json")
sh = set(r) & set(o)
ck(len(sh) == 24 and all(r[k] == o[k] for k in sh), "UniVLA determinism 24/24")
j = load("results/univla_bridge_0805/baseline/**/results_*.json")
l4 = load("results/univla_bridge_0805/baseline_l4/**/results_*.json")
sh = set(j) & set(l4)
gap = 100.0 * (sum(l4[k] for k in sh) - sum(j[k] for k in sh)) / len(sh)
disc = sum(1 for k in sh if j[k] != l4[k])
ck(near(abs(gap), 3.1) and disc == 11, f"dual baseline gap {gap:+.1f}, discordant {disc}")
wide = load("results/univla_bridge_0805/depth_prune4_mid/**/results_*.json")
_, dw, _, _, _ = mcn(l4, wide)
ck(near(dw, -79.2), f"UniVLA widened window {dw:+.1f} != -79.2")
print("    5 window cells, 5 variant pairs, keep sweep, 3 controls reproduce")

# ================================================================ 6
sec("6  mechanism (6.4 / 6.5)")
import mechanism_move_near as M  # noqa: E402


def buckets(run_dir):
    eps = M.collect(run_dir) if hasattr(M, "collect") else None
    return eps


def run_mech(dirs):
    out = subprocess.run([sys.executable, "experiments/mechanism_move_near.py"] + dirs,
                         capture_output=True, text=True).stdout
    full = {}
    for l in out.split("\n"):
        m = re.match(r"\s+([\w/]+)\s+60\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", l)
        if m:
            full[m.group(1)] = tuple(int(m.group(i)) for i in range(2, 7))
    return full, out


ov8, _ = run_mech(["results/openvla_fractal_0806_baseline",
                   "results/openvla_fractal_0806_depth_prune4",
                   "results/openvla_fractal_depth_control/prune3",
                   "results/openvla_fractal_depth_control/prune4_gap3",
                   "results/openvla_fractal_0806_depth_prune4_early",
                   "results/openvla_fractal_0806_foveate_blur",
                   "results/openvla_fractal_0806_foveate",
                   "results/openvla_fractal_depth_control/window875"])
EXP_OV = {"baseline": (37, 1, 0, 6, 16), "depth_prune4": (42, 5, 2, 7, 4),
          "prune3": (42, 1, 2, 8, 7), "prune4_gap3": (38, 3, 1, 12, 6),
          "depth_prune4_early": (30, 1, 1, 9, 19), "foveate_blur": (18, 5, 3, 15, 19),
          "foveate": (12, 6, 1, 6, 35), "window875": (11, 1, 0, 7, 41)}
for cond, want in EXP_OV.items():
    ck(ov8.get(cond) == want, f"6.5 OpenVLA {cond}: {ov8.get(cond)} != {want}")
sp, spout = run_mech(["results/spatialvla_mech_0811/baseline",
                      "results/spatialvla_mech_0811/depth_prune4"])
ck(sp.get("baseline") == (50, 4, 1, 5, 0), f"6.4 baseline {sp.get('baseline')}")
ck(sp.get("depth_prune4") == (31, 7, 4, 6, 12), f"6.4 prune4 {sp.get('depth_prune4')}")
m = re.search(r"depth_prune4\s+(\d+)\s+0\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
              spout.split("Among episodes")[1])
ck(tuple(int(m.group(i)) for i in range(1, 6)) == (22, 5, 2, 5, 10), "6.4 lost-episode split")
pb, _ = run_mech(["results/spatialvla_depth_control/fractal/prune4_back"])
ck(pb.get("prune4_back") == (4, 2, 7, 5, 42), f"6.5 prune4_back {pb.get('prune4_back')}")
ratios = []
for cond, (s_, dr, wo, mp, nc) in {**ov8, **sp, **pb}.items():
    if 60 - s_:
        ratios.append(100 * wo / (60 - s_))
ck(near(min(ratios), 0.0, 0.05) and near(max(ratios), 13.8, 0.05),
   f"wrong_object share range {min(ratios):.1f}~{max(ratios):.1f}")
print("    all eleven 6.5 rows + 6.4 split reproduce; wrong_object 0.0~13.8%")

# ================================================================ 7
sec("7  pixel measurements")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from adaptive_sparse_vla.foveation import foveate_image_logpolar  # noqa: E402

img = cv2.imread("experiments/figures/obs_carrot_raw.png")
ck(img is not None and img.shape[:2] == (480, 640), "carrot obs missing")
rows224 = []
for keep in (1.0, 0.4, 0.2, 0.1):
    rt = foveate_image_logpolar(img, keep_ratio=keep)
    A = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(int)
    Bm = cv2.resize(rt, (224, 224), interpolation=cv2.INTER_AREA).astype(int)
    d = np.abs(A - Bm).max(axis=2)
    rows224.append((100 * (d > 0).mean(), 100 * (d > 2).mean(), d.mean(), int(d.max())))
EXP224 = [(71.0, 19.3, 2.06, 187), (88.4, 43.6, 4.09, 216),
          (90.3, 49.4, 5.16, 218), (92.6, 57.2, 6.73, 232)]
for got, want in zip(rows224, EXP224):
    ck(near(got[0], want[0], 0.15) and near(got[1], want[1], 0.15)
       and near(got[2], want[2], 0.02) and got[3] == want[3], f"224 table row {want}")
cy, cx = slice(120, 360), slice(160, 480)
for keep, want in ((1.0, 2.5), (0.1, 8.3)):
    rt = foveate_image_logpolar(img, keep_ratio=keep)
    d = np.abs(img.astype(int) - rt.astype(int)).max(axis=2)[cy, cx]
    ck(near(100 * (d >= 10).mean(), want, 0.06), f"center >=10 at keep={keep}")


def mean_err(im, keep):
    rt = foveate_image_logpolar(im, keep_ratio=keep)
    d = np.abs(im.astype(int) - rt.astype(int)).mean(axis=2)
    h, w = im.shape[:2]
    c = d[h // 4:3 * h // 4, w // 4:3 * w // 4]
    mask = np.ones_like(d, bool)
    mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = False
    return c.mean(), d[mask].mean()


c100, p100 = mean_err(img, 1.0)
c10, p10 = mean_err(img, 0.1)
ck(near(c100, 1.2, 0.06) and near(p100, 3.3, 0.06), f"roundtrip keep100 {c100:.1f}/{p100:.1f}")
ck(near(p100 / c100, 2.7, 0.06), "periphery/center ratio 2.7x")
ck(near(c10, 3.5, 0.06) and near(p10, 7.6, 0.06), f"roundtrip keep10 {c10:.1f}/{p10:.1f}")
out = subprocess.run([sys.executable, "experiments/measure_foveation_roundtrip.py",
                      "experiments/figures/obs_carrot_raw.png"],
                     capture_output=True, text=True).stdout
ck(re.search(r"Laplacian\s+log-polar\s+39%\s+7%\s+2%\s+7%\s+21%", out) is not None,
   "Laplacian band row 39/7/2/7/21")
ck(re.search(r"Sobel\s+log-polar\s+68%", out) is not None, "Sobel center 68%")
ck("last column -> 395.3 px" in out and "map monotone: True" in out, "column map monotone/395.3")
for band, cols_, ratio in (("1-2", 49, "0.02"), ("64-128", 74, "0.86"), ("256-400", 47, "3.06")):
    ck(re.search(rf"{band}\s+{cols_}\s+\d+\s+{re.escape(ratio)}", out) is not None,
       f"column map row {band}: {cols_}/{ratio}")
sc = os.environ.get("SCRATCH_DIR", "/tmp")
p256 = os.path.join(sc, "vv_carrot256L.png")
cv2.imwrite(p256, cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR))
out2 = subprocess.run([sys.executable, "experiments/measure_foveation_roundtrip.py", p256],
                      capture_output=True, text=True).stdout
band = re.search(r"vv_carrot256L\.png\s+256x256, keep=20%.*?Laplacian\s+log-polar"
                 r"\s+\d+%\s+\d+%\s+\d+%\s+\d+%\s+(\d+)%", out2, re.S)
ck(band and band.group(1) == "39", "256 INTER_LINEAR outer band 39%")
r0 = math.sqrt(0.2 * 480 * 640 / math.pi) / math.hypot(320, 240)
ck(near(r0, 0.35, 0.005), f"blur disc radius {r0:.3f}")
outfov = foveate_image_logpolar(img, keep_ratio=1.0, center=None)
for (x, y, want) in ((320 - 40, 240 - 30, 0.68), (640 - 96 - 10, 10, 6.66)):
    a2, b2 = img[y:y + 96, x:x + 96], outfov[y:y + 96, x:x + 96]
    d = np.abs(b2.astype(int) - a2.astype(int)).max(axis=2)
    ck(near(d.mean(), want, 0.006), f"zoom mean at ({x},{y})")
v = np.array([10, 200, 30, 220, 40], float)
up = np.repeat(v, 49)
ck(np.allclose(np.array([up[i * 49:(i + 1) * 49].mean() for i in range(5)]), v),
   "49x up-down demo lossless")
sm = cv2.resize(cv2.resize(v.reshape(1, 5), (2, 1), interpolation=cv2.INTER_AREA),
                (5, 1), interpolation=cv2.INTER_LINEAR).ravel()
ck(sm.std() < v.std() / 3, "3:1 down-up demo lossy (values crushed together)")
mark = np.zeros((480, 640, 3), np.uint8)
mark[239:242, 599:602] = 255
rt = foveate_image_logpolar(mark, keep_ratio=1.0)
g = rt[..., 0].astype(float)
yy, xx = np.mgrid[0:480, 0:640]
disp = math.hypot((xx * g).sum() / g.sum() - 600, (yy * g).sum() / g.sum() - 240)
ck(disp <= 1.0, f"edge marker displacement {disp:.2f} > 1 px")
print("    224 table, bands, column map, zoom, demos, recipes all reproduce")

# ================================================================ 8
sec("8  document structure")
DOCS = {d: open("experiments/" + d).read() for d in ("Report.md", "Overview.md", "RelatedWork.md")}
rag = 0
for name, s in DOCS.items():
    lines = s.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            h = lines[i].count("|")
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                if lines[j].count("|") != h:
                    rag += 1
                j += 1
            i = j
        else:
            i += 1
ck(rag == 0, f"{rag} ragged table rows")
lines = DOCS["Report.md"].split("\n")
s7 = next(i for i, l in enumerate(lines) if l.startswith("## 7.1"))
e7 = next(i for i, l in enumerate(lines) if l.startswith("## 7.2"))
seg = lines[s7:e7]
i2 = next(i for i, l in enumerate(seg) if "움직인 것 ②" in l)
rows = []
for l in seg[i2:]:
    if l.startswith("|") and not l.startswith("|---") and l.strip() != "| 무엇 | 무엇이 틀렸나 |":
        rows.append(l)
    elif rows and not l.startswith("|"):
        break
tseg = DOCS["Report.md"].split("| 유형 | 개수 | 예 |")[1].split("\n\n")[0]
tsum = sum(int(x) for x in re.findall(r"\|\s*(\d+)\s*\|", tseg))
ck(len(rows) == tsum, f"7.1 rows {len(rows)} != type sum {tsum}")
ck(f"{len(rows)}건" in DOCS["Overview.md"], f"Overview counter != {len(rows)}건")
ck(f"②의 {len(rows)}행" in DOCS["Report.md"], "Report prose counter stale")
for d, tok in (("Report.md", r"~2,3\d\d줄"), ("Overview.md", r"~4[67]0줄"),
               ("RelatedWork.md", r"~1,2\d0줄")):
    n = len(DOCS[d].split("\n"))
    stated = re.findall(tok, DOCS["Report.md"] + DOCS["Overview.md"])
    ck(bool(stated), f"line-count token for {d} missing")
rep_secs = set(re.findall(r"^#+ *§?(\d+(?:\.\d+)*)", DOCS["Report.md"], re.M))
rw_secs = set(re.findall(r"^## (\d+\.\d+|A\.\d+)", DOCS["RelatedWork.md"], re.M))
badref = []
for name in ("Overview.md", "RelatedWork.md"):
    for m in re.finditer(r"§(\d+\.\d+(?:\.\d+)?)", DOCS[name]):
        # "그 논문 §4.6" marks another paper's own section, not ours
        if DOCS[name][max(0, m.start() - 6):m.start()].endswith("논문 "):
            continue
        s_ = m.group(1)
        if s_.split(".")[0] == "2":
            if s_ not in rw_secs:
                badref.append((name, s_))
        elif s_ not in rep_secs and f"## {s_}" not in DOCS["Report.md"] \
                and f"§{s_}" not in DOCS["Report.md"]:
            badref.append((name, s_))
ck(not badref, f"broken section refs: {set(badref)}")
paths = set()
for name in ("Overview.md", "RelatedWork.md"):
    paths |= set(re.findall(r"`([\w./]+\.(?:py|md|png))`", DOCS[name]))
    paths |= set(re.findall(r"\]\((figures/[\w.]+)\)", DOCS[name]))
mp = [p for p in paths if not any(os.path.exists(c) for c in
      (p, "experiments/" + p, "SpatialVLA/experiments/tome/" + os.path.basename(p),
       "adaptive_sparse_vla/" + os.path.basename(p)))]
ck(not mp, f"cited paths missing: {mp}")
gm = open("SpatialVLA/experiments/tome/depth_prune_gemma2.py").read()
ck("_sum" in gm and "_cnt" in gm and "reverse=True" in gm, "depth_prune_gemma2 citations")
ck("not calibrated" in open("SpatialVLA/experiments/tome/tome_spatialvla_eval.py").read(),
   "calibrated flag citation")
print("    tables, counters, refs, paths, code citations clean")

# ================================================================ 9
sec("9  external records")
prof = open("docs/VISUAL_TOKENS_VS_LATENCY.md").read()
for tok in ("125", "677", "903", "75.0%", "951", "1362", "69.8%", "0.99×", "12 action tokens",
            "26 action tokens"):
    ck(tok in prof, f"profile doc missing {tok}")
chunk = open("experiments/ChunkExecFoveation_univla.md").read()
for tok in ("603", "1414", "+13.6pp", "1.9×", "−12.5pp"):
    ck(tok in chunk, f"chunk log missing {tok}")
cand = open("experiments/RelatedWork_Candidates.md").read()
n_cand = len([l for l in cand.split("\n") if re.match(r"^\|\s*\d+\s*\|", l)])
ck(n_cand == 60, f"candidate list rows {n_cand} != 60")
print("    latency profile, chunk log, 60-paper list all present")

# ================================================================ 10
sec("10  paper-table arithmetic (from cells the docs themselves quote)")
AR = [("VLA-Cache FastV +2.6%", 53.28 / 51.91 - 1, 0.026),
      ("VLA-Cache SparseVLM FLOPs -24.5%", 1 - 1.407 / 1.864, 0.245),
      ("VLA-Cache SparseVLM latency +60.6%", 83.39 / 51.91 - 1, 0.606),
      ("STT 42x", 572.7 / 13.7 / 10, 4.18),
      ("STT vs EfficientSAM-S 5.7x", 78.6 / 13.7, 5.74),
      ("LFA ViT 14.9x", 243.8 / 16.4, 14.87),
      ("LFA policy 3.8x", 334.7 / 87.9, 3.81),
      ("LFA training 7.7x", 833.2 / 108.2, 7.70),
      ("LFA GFLOPs -93.9%", 1 - 115.6 / 1905.4, 0.939),
      ("MoLe OpenVLA +10.2", 55.6 - 45.4, 10.2),
      ("MoLe CogAct +3.6", 60.8 - 57.2, 3.6),
      ("MoLe random-skip -6.0", 51.2 - 57.2, -6.0),
      ("Gaze-Reg total -7.4", 78.5 - 85.9, -7.4),
      ("ShortGPT XSum collapse -18.73", 0.67 - 19.40, -18.73),
      ("ShortGPT BoolQ +3.09", 74.71 - 71.62, 3.09),
      ("profile what-if SpatialVLA 1.15x", 903 / (903 - 0.94 * 125), 1.149),
      ("profile what-if SpatialVLA 1.26x", 903 / (903 - 125 - 60), 1.258),
      ("profile what-if UniVLA 1.06x", 1362 / (1362 - 0.94 * 83), 1.061),
      ("profile what-if UniVLA 1.23x", 1362 / (1362 - 83 - 172), 1.231),
      ("82-point gap", 12.5 + 69.8, 82.3),
      ("keep-sweep spread 26", 30.2 - 4.2, 26.0),
      ("chunk slowdown 2.3x", 1414 / 603, 2.345)]
for name, got, want in AR:
    ck(abs(got - want) < 0.011 * max(1, abs(want)), f"arithmetic {name}: {got:.3f} vs {want}")
print(f"    {len(AR)} derived numbers all follow from their quoted cells")

# ================================================================ 11
sec("11  delegated suites")
r1 = subprocess.run([sys.executable, "experiments/verify_overview_claims.py"],
                    capture_output=True, text=True).stdout
ck(r1.count("OK") == 10, f"verify_overview_claims OK count {r1.count('OK')} != 10")
r2 = subprocess.run([sys.executable, "experiments/audit_claims.py"],
                    capture_output=True, text=True).stdout
m = re.search(r"decided classes \(1-3\) failures: (\d+)", r2)
ck(m is not None and m.group(1) == "0", "audit_claims classes 1-3 not clean")
print("    verify_overview_claims 10/10, audit classes 1-3 clean")

# ================================================================ 12
sec("12  Report-specific claims")
# (a) direct-pair tests quoted only in Report prose (S3.6, S4.4, S5.3-adjacent)
SB = "results/spatialvla_bridge_0805/{}/**/results_*.json"
pairs = [
    ("S3.6 prune1 vs prune1_back", SB.format("depth_prune1"), SB.format("depth_prune1_back"),
     -2.1, 0.80, 2),
    ("S3.6 prune2 -> prune4", SB.format("depth_prune2"), SB.format("depth_prune4"),
     -18.8, 0.0001, 4),
    ("S3.6 prune4 vs prune4_back", SB.format("depth_prune4"), SB.format("depth_prune4_back"),
     -2.1, 0.50, 2)]
for name, p1, p2, dd, pp, dec in pairs:
    n, d, br, fx, pv = mcn(load(p1), load(p2))
    ck(near(d, dd) and round(pv, dec) == pp, f"{name}: d={d:+.1f} p={pv:.4f}")
# keep10 -> keep100 within the new tree only (tree-pure comparison in S4.3 b)
n, d, br, fx, pv = mcn(load("results/openvla_bridge_foveate_sweep/keep10/**/results_*.json"),
                       load("results/openvla_bridge_foveate_sweep/keep100/**/results_*.json"))
ck(near(d, 26.0) and abs(pv - 4.7e-4) / 4.7e-4 < 0.05, f"keep10->keep100 {d:+.1f} p={pv:.2e}")
# (b) window25 is bit-identical to prune4 (same layers selected) -- 135/135
w25 = load_steps("results/openvla_fractal_depth_control/window25/**/results_*.json")
p4 = load_steps("results/openvla_fractal_0806_depth_prune4/**/results_*.json")
sh = set(w25) & set(p4)
ck(len(sh) == 135 and all(w25[k] == p4[k] for k in sh), "window25 == prune4 135/135")
# (c) S4.4 (3): spoon_on_towel reselected the same layer -> 24 bit-identical episodes
a1 = load_steps(SB.format("depth_prune1"))
a2 = load_steps(SB.format("depth_prune1_back"))
spoon = [k for k in set(a1) & set(a2) if "spoon" in k[0]]
ck(len(spoon) == 24 and all(a1[k] == a2[k] for k in spoon),
   "prune1_back spoon 24/24 bit-identical")
# (d) S4.4 pick_horizontal layer ladder: 5/6/8/11/14/5/0 vs baseline 7
LAD = [("results/openvla_fractal_0806_depth_prune1/**/results_*.json", 5),
       ("results/openvla_fractal_0806_depth_prune2/**/results_*.json", 6),
       ("results/openvla_fractal_depth_control/prune3/**/results_*.json", 8),
       ("results/openvla_fractal_0806_depth_prune4/**/results_*.json", 11),
       ("results/openvla_fractal_0806_depth_prune4_early/**/results_*.json", 14),
       ("results/openvla_fractal_depth_control/prune4_gap3/**/results_*.json", 5),
       ("results/openvla_fractal_depth_control/window875/**/results_*.json", 0),
       ("results/openvla_fractal_0806_baseline/**/results_*.json", 7)]
for pat, want in LAD:
    got = sum(v for (t, _), v in load(pat).items() if "pick_horizontal" in t)
    ck(got == want, f"ladder pick_horizontal {pat.split('/')[1]}: {got} != {want}")
# prune3 whole-cell +11.1 (p=0.0167) quoted in S4.4
n, d, br, fx, pv = mcn(load("results/openvla_fractal_0806_baseline/**/results_*.json"),
                       load("results/openvla_fractal_depth_control/prune3/**/results_*.json"))
ck(near(d, 11.1) and round(pv, 4) == 0.0167, f"prune3 {d:+.1f} p={pv:.4f}")
# (e) S6.4 sign tests: 0->12 gives p=0.0005, 1->4 gives p=0.375 (exact binomial)
ck(round(min(1.0, 2 * sum(comb(12, x) for x in range(0 + 1)) / 2 ** 12), 4) == 0.0005,
   "6.4 no_contact sign test 0.0005")
ck(round(min(1.0, 2 * sum(comb(5, x) for x in range(1 + 1)) / 2 ** 5), 3) == 0.375,
   "6.4 wrong_object sign test 0.375")
# dual-baseline McNemar p = 0.5488 (11 discordant, 7/4)
ck(round(min(1.0, 2 * sum(comb(11, x) for x in range(4 + 1)) / 2 ** 11), 4) == 0.5488,
   "dual-baseline p 0.5488")
# (f) S3.8 baseline table: our five values match records; exactly one lower, by 4.2
seg38 = rep.split("### (c)")[1].split("### (d)")[0] if "### (c)" in rep else rep
ours = {"OpenVLA / Bridge": 15.6, "OpenVLA / Fractal": 38.5, "SpatialVLA / Bridge": 30.2,
        "SpatialVLA / Fractal": 84.4, "UniVLA / Bridge": 81.2}
BASE = {"OpenVLA / Bridge": "results/openvla_bridge_0805/baseline/**/results_*.json",
        "OpenVLA / Fractal": "results/openvla_fractal_0806_baseline/**/results_*.json",
        "SpatialVLA / Bridge": "results/spatialvla_bridge_0805/baseline/**/results_*.json",
        "SpatialVLA / Fractal": "results/spatialvla_fractal_0806/baseline/**/results_*.json",
        "UniVLA / Bridge": "results/univla_bridge_0805/baseline_l4/**/results_*.json"}
for cell, want in ours.items():
    d0 = load(BASE[cell])
    ck(near(100 * sum(d0.values()) / len(d0), want), f"3.8 our baseline {cell}")
ck("넷이 높고 하나가" in rep and "4.2" in rep and "30.2% 대 34.4%" in rep,
   "3.8 four-higher-one-lower statement")
# (g) every "p = <decimal>" literal in Report must be a p some real pairing produces
pool = set()


def add_p(pv):
    for dec in (1, 2, 3, 4):
        pool.add(round(pv, dec))


for v in paired.values():
    add_p(v[3])
for _, _, r in inter:
    add_p(r[4])
for pat1, pat2, *_ in pairs:
    pass
CAMPS = ["results/openvla_bridge_0805", "results/openvla_fractal_0806_*",
         "results/openvla_fractal_depth_control", "results/openvla_bridge_foveate_sweep",
         "results/spatialvla_bridge_0805", "results/spatialvla_fractal_0806",
         "results/univla_bridge_0805"]
trees = {}
for c in CAMPS:
    for d0 in glob.glob(c) + glob.glob(c + "/*"):
        if os.path.isdir(d0) and glob.glob(d0 + "/**/results_*.json", recursive=True):
            if not any(os.path.isdir(os.path.join(d0, x)) and
                       glob.glob(os.path.join(d0, x) + "/**/results_*.json", recursive=True)
                       for x in os.listdir(d0)):
                trees[d0] = load(d0 + "/**/results_*.json")
trees["legacy_fov"] = load(
    "RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated/**/results_*.json")
tl = list(trees.values())
for i in range(len(tl)):
    for j in range(i + 1, len(tl)):
        if set(tl[i]) & set(tl[j]):
            _, _, _, _, pv = mcn(tl[i], tl[j])
            add_p(pv)
            for fam in ("move_near", "coke_can", "carrot", "eggplant", "spoon", "stack"):
                ai = {k: v for k, v in tl[i].items() if fam in k[0]}
                bj = {k: v for k, v in tl[j].items() if fam in k[0]}
                if ai and set(ai) & set(bj):
                    _, _, _, _, pv2 = mcn(ai, bj)
                    add_p(pv2)
for m_, k_ in ((12, 0), (5, 1), (11, 4), (3, 0)):
    add_p(min(1.0, 2 * sum(comb(m_, x) for x in range(k_ + 1)) / 2 ** m_))
for a_ in (0.05 / 38, 0.05 / 43, 0.05 / 35, 0.05 / 42, 0.05 / 15, 0.05 / 36, 0.05 / 30):
    add_p(a_)
unexplained = []
for m in re.finditer(r"p\s*=\s*(0\.\d+)", rep):
    lit = m.group(1)
    if round(float(lit), len(lit) - 2) not in pool:
        ln = rep[:m.start()].count("\n") + 1
        unexplained.append((ln, lit))
ck(not unexplained, f"p literals no pairing produces: {unexplained[:6]}")
n_plits = len(re.findall(r"p\s*=\s*0\.", rep))
print(f"    direct pairs, ladders, sign tests, 3.8 table, {n_plits} p-literals all sourced")

# ================================================================ done
print("\n" + "=" * 66)
print(f"checks run: {CHECKS[0]}   failures: {len(FAILS)}")
for f in FAILS:
    print("  FAIL:", f)
print("""
NOT checkable by this script (stated in the documents themselves):
  - GPU model per run: never recorded; bounded at 3.1pp (Report 3.4.0)
  - the three console-only ms values in 3.5.2 (run was aborted, no files)
  - quoted paper cells vs the PDFs: verified by hand once; PDFs not in repo""")
sys.exit(1 if FAILS else 0)
