"""Recompute every numeric claim in Overview.md from results/, independently."""
import glob
import json
import os
from math import comb

ROOT = "/home/user/BiVLA"


def load(pattern, task_filter=None):
    """-> {(task, ep_id): success}"""
    out = {}
    for f in glob.glob(os.path.join(ROOT, pattern), recursive=True):
        s = json.load(open(f))
        t = s["task"]
        if task_filter and task_filter not in t:
            continue
        for e in s["episodes"]:
            out[(t, int(e["ep_id"]))] = bool(e["success"])
    return out


def mcnemar(a, b):
    """Paired exact test on shared episodes. -> (n, rate_a, rate_b, delta, b01, b10, p)"""
    k = sorted(set(a) & set(b))
    n = len(k)
    ra = sum(a[i] for i in k) / n
    rb = sum(b[i] for i in k) / n
    broke = sum(a[i] and not b[i] for i in k)   # a solved, b lost
    fixed = sum(b[i] and not a[i] for i in k)   # b solved, a lost
    m = broke + fixed
    if m == 0:
        p = 1.0
    else:
        obs = abs(fixed - broke)
        p = sum(comb(m, x) for x in range(m + 1) if abs(m - 2 * x) >= obs) / 2 ** m
        p = min(1.0, p)
    return n, 100 * ra, 100 * rb, 100 * (rb - ra), broke, fixed, p


def show(label, base, cond, expect=None):
    n, ra, rb, d, br, fx, p = mcnemar(base, cond)
    flag = ""
    if expect is not None:
        flag = "  OK" if abs(d - expect) < 0.15 else f"  <-- doc says {expect:+.1f}"
    print(f"  {label:34s} n={n:3d}  {ra:5.1f}% -> {rb:5.1f}%  d={d:+6.1f}  "
          f"{br}/{fx}  p={p:.4g}{flag}")
    return d, p


print("=" * 78)
print("(1) window sweep table  [Overview lines 73-78]")
print("=" * 78)
base = load("results/openvla_fractal_0806_baseline/**/results_*.json")
for name, path, exp in [
    ("prune4        [17,20,23,26]", "results/openvla_fractal_0806_depth_prune4/**/results_*.json", +15.6),
    ("prune4_early  [2,4,23,26]", "results/openvla_fractal_0806_depth_prune4_early/**/results_*.json", +5.9),
    ("prune4_gap3   [17,23,27,31]", "results/openvla_fractal_depth_control/prune4_gap3/**/results_*.json", +1.5),
    ("window875     [28,29,30,31]", "results/openvla_fractal_depth_control/window875/**/results_*.json", -30.4),
    ("window25", "results/openvla_fractal_depth_control/window25/**/results_*.json", None),
    ("prune3", "results/openvla_fractal_depth_control/prune3/**/results_*.json", None),
]:
    show(name, base, load(path), exp)

print("\n  window875 per-task (Overview line 80: 'pick 3 tasks all 0/25')")
w = load("results/openvla_fractal_depth_control/window875/**/results_*.json")
tasks = sorted({t for t, _ in w})
for t in tasks:
    e = [v for (tt, _), v in w.items() if tt == t]
    print(f"    {t:40s} {sum(e)}/{len(e)}")

print("\n" + "=" * 78)
print("(2) keep sweep  [Overview lines 103-108]")
print("=" * 78)
ob = load("results/openvla_bridge_0805/baseline/**/results_*.json")
print(f"  baseline rate = {100*sum(ob.values())/len(ob):.1f}%  (Overview line 143 says 15.6%)")
for keep, exp in (("keep10", +4.2), ("keep20", +18.8), ("keep40", +19.8), ("keep100", +30.2)):
    # keep=20 is the grid cell; it lives in the legacy RetinaBased tree, not
    # results/. Verified bit-identical to results/openvla_bridge_0805/baseline
    # on the baseline condition (96/96 incl. step counts).
    path = ("RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated/**/results_*.json"
            if keep == "keep20" else f"results/openvla_bridge_foveate_sweep/{keep}/**/results_*.json")
    show(keep, ob, load(path), exp)

print("\n" + "=" * 78)
print("(3) benchmark sign flip  [Overview line 156]")
print("=" * 78)
# The OpenVLA/Bridge log-polar cell has no directory under results/ -- it is the
# same legacy RetinaBased run used for keep20 above. Report 7.1 logs that.
show("OpenVLA/Bridge  log-polar", ob,
     load("RetinaBased/GoogleColab/results_reproduction_eager/openvla_foveated"
          "/**/results_*.json"), +18.8)
show("OpenVLA/Fractal log-polar", base,
     load("results/openvla_fractal_0806_foveate/**/results_*.json"), -19.3)
