#!/usr/bin/env python3
import csv, json, math, os, statistics
from collections import defaultdict, Counter, OrderedDict
ROOT = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11"
RES = os.path.join(ROOT, "results"); SUMM = os.path.join(ROOT, "summarization")
med = statistics.median
def mean(x): return sum(x)/len(x)

folders = OrderedDict()
for dp, dn, fn in os.walk(RES):
    dn.sort()
    if "summary.json" in fn:
        rel = os.path.relpath(dp, RES); p = rel.split("/")
        folders[rel] = dict(backbone=p[0], config=p[1], leaf=p[2] if len(p) > 2 else None,
                            summary=json.load(open(os.path.join(dp, "summary.json"))),
                            episodes=[json.loads(l) for l in open(os.path.join(dp, "episodes.jsonl")) if l.strip()])

# 1. latency median deviation table
print("# 1. summary median_ms vs median of per-episode medians, per backbone (folders with the field)")
print("| backbone_env | group | folders | policy: n>0.5ms | policy max|dev| | cycle: n>0.5ms | cycle max|dev| |")
print("|---|---|---|---|---|---|---|")
rows = defaultdict(lambda: dict(n=0, pn=0, pm=0.0, cn=0, cm=0.0))
for rel, e in folders.items():
    s = e["summary"]; eps = e["episodes"]
    if not s.get("policy_query_latency_ms"): continue
    grp = "action_repeat" if e["config"].startswith("action_repeat") else "other"
    r = rows[(e["backbone"], grp)]; r["n"] += 1
    for sk, ek, kn, km in [("policy_query_latency_ms", "query_latency_ms", "pn", "pm"), ("inference_cycle_latency_ms", "inference_cycle_ms", "cn", "cm")]:
        if not s.get(sk) or not eps[0].get(ek): continue
        d = abs(s[sk]["median_ms"] - med([x[ek]["median_ms"] for x in eps if x.get(ek)]))
        r[km] = max(r[km], d)
        if d > 0.5: r[kn] += 1
for (bb, grp), r in sorted(rows.items()):
    print(f"| {bb} | {grp} | {r['n']} | {r['pn']} | {r['pm']:.2f} | {r['cn']} | {r['cm']:.2f} |")

# 2. as-stated reuse bound violators
print("\n# 2. episodes with reuses > calls*cap/(cap+1) (as-stated bound); all satisfy reuses <= cap*calls")
for rel, e in folders.items():
    if not e["config"].startswith("guarded_reuse"): continue
    cap = (e["summary"].get("arguments") or {}).get("reuse_max_consecutive")
    if cap is None: continue
    for r in e["episodes"]:
        if r["reuses"] > r["policy_calls"] * cap / (cap + 1):
            print(f"  {rel} ep{r['episode_index']}: steps={r['steps_executed']} calls={r['policy_calls']} reuses={r['reuses']} cap={cap} success={r['success']}")

# 3. CronusVLA bridge guarded reuse identity across settings; fusion identity
print("\n# 3. episode-level identity across settings (CronusVLA)")
for bb in ["cronusvla_simplerenv_bridge", "cronusvla_simplerenv_fractal"]:
    for fam, cfgs in [("guarded_reuse", ["guarded_reuse_strict", "guarded_reuse_moderate", "guarded_reuse_aggressive"]),
                      ("temporal_fusion", ["temporal_fusion_motion_entropy", "temporal_fusion_task_aware", "temporal_fusion_conservative_adaptive"])]:
        sigs = {}
        for c in cfgs:
            recs = [r for rel, e in folders.items() if e["backbone"] == bb and e["config"] == c for r in e["episodes"]]
            sigs[c] = tuple(sorted((r["task"], r["episode_index"], r["success"], r["steps_executed"], r["policy_calls"], r["reuses"], r.get("temporal_fused_patches")) for r in recs))
            elapsed = tuple(sorted((r["task"], r["episode_index"], round(r["episode_elapsed_ms"], 3)) for r in recs))
            sigs[c + "_elapsed"] = elapsed
        same = sigs[cfgs[0]] == sigs[cfgs[1]] == sigs[cfgs[2]]
        same_el = sigs[cfgs[0] + "_elapsed"] == sigs[cfgs[1] + "_elapsed"] == sigs[cfgs[2] + "_elapsed"]
        print(f"  {bb} {fam}: success/steps/calls/reuses/fused identical across 3 settings = {same}; elapsed_ms identical = {same_el}")
    # also compare guarded strict vs original
    o = {(r["task"], r["episode_index"]): (r["success"], r["steps_executed"]) for rel, e in folders.items() if e["backbone"] == bb and e["config"] == "original" for r in e["episodes"]}
    g = {(r["task"], r["episode_index"]): (r["success"], r["steps_executed"]) for rel, e in folders.items() if e["backbone"] == bb and e["config"] == "guarded_reuse_strict" for r in e["episodes"]}
    print(f"  {bb}: guarded_reuse_strict == original at episode level: {o == g}")

# 4. folders lacking arguments / latency
print("\n# 4. folders whose summary lacks 'arguments' or latency blocks")
for rel, e in folders.items():
    s = e["summary"]
    miss = [k for k in ["arguments", "policy_query_latency_ms", "inference_cycle_latency_ms", "episode_errors", "success_rate", "average_steps"] if s.get(k) is None]
    if miss:
        print(f"  {rel}: missing {miss}")

# 5. UniVLA task-aware CSV row and the OOM folders
print("\n# 5. CSV rows for univla_simpler_widowx temporal_fusion_task_aware")
for row in csv.DictReader(open(os.path.join(SUMM, "simpler_widowx/univla_simpler_widowx.csv"))):
    if row["configuration"] in ("temporal_fusion_task_aware", "original"):
        print("  ", {k: row[k] for k in ["configuration", "episodes", "successes", "success_rate_pct", "avg_steps", "avg_policy_calls", "avg_episode_time_s", "policy_median_latency_ms", "median_reusable_visual_tokens"]})
for rel in ["univla_simplerenv_bridge/temporal_fusion_task_aware/" + t for t in ["widowx_carrot_on_plate", "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube"]]:
    e = folders[rel]; errs = [r for r in e["episodes"] if r.get("error")]
    print(f"  {rel}: episodes={len(e['episodes'])} errors={len(errs)} successes={sum(r['success'] for r in e['episodes'])} summary.successes={e['summary']['successes']} summary.episode_errors={e['summary'].get('episode_errors')} status={e['summary']['status']}")
    ok = [r for r in e["episodes"] if not r.get("error")]
    if errs:
        print(f"     non-error episodes: {len(ok)}, successes among them {sum(r['success'] for r in ok)}; error episodes steps min/max {min(r['steps_executed'] for r in errs)}/{max(r['steps_executed'] for r in errs)}")

# 6. depth: was protection binding? simulate without protection
print("\n# 6. depth pruning: recorded set vs greedy without any protection (ascending influence, gap>=2)")
LAY = {"openvla": 32, "cogact": 32, "spatialvla": 26, "univla": 32, "minivla": 24, "cronusvla": 12, "smolvla": 16}
def greedy(infl, budget, start, protect_last):
    n = len(infl); c = list(range(start, n - (1 if protect_last else 0)))
    ranked = sorted(c, key=lambda i: (infl[i], i)); ch = []
    for i in ranked:
        if any(abs(i - p) <= 1 for p in ch): continue
        ch.append(i)
        if len(ch) == budget: break
    return sorted(ch)
seen = set()
for rel, e in folders.items():
    if not e["config"].startswith("depth_pruning"): continue
    dc = e["summary"].get("depth_calibration") or {}
    infl = dc.get("influence"); sel = dc.get("selected_layers")
    if not infl or not sel: continue
    key = (e["backbone"], e["config"], tuple(sel))
    if key in seen: continue
    seen.add(key)
    b = len(sel); L = len(infl)
    nop = greedy(infl, b, 0, False); lead = greedy(infl, b, math.floor(0.25 * L), False); last = greedy(infl, b, 0, True)
    lowest = sorted(range(L), key=lambda i: (infl[i], i))[:b]
    flags = []
    if nop != sorted(sel): flags.append(f"protection binding (unprotected greedy -> {nop})")
    if sorted(lowest) != sorted(sel): flags.append(f"gap rule binding (plain lowest-{b} -> {sorted(lowest)})")
    print(f"  {e['backbone']}/{e['config']} {e['leaf'] or ''}: recorded {sorted(sel)}; {'; '.join(flags) or 'neither protection nor gap rule was binding'}")

# 7. CSV latency: extra candidates for the odd OpenVLA action-repeat cycle medians
print("\n# 7. CSV cycle_median candidates for OpenVLA action_repeat rows")
CSVS = {"simpler_widowx/openvla_simplerenv_widowx.csv": ("openvla_simplerenv", None), "google_robot_fractal/openvla_simplerenv_fractal.csv": ("openvla_simplerenv_fractal", None)}
for s_ in ["10", "goal", "object", "spatial"]:
    CSVS[f"libero/openvla_libero_{s_}.csv"] = ("openvla_libero", f"libero_{s_}")
for relcsv, (bb, suite) in CSVS.items():
    for row in csv.DictReader(open(os.path.join(SUMM, relcsv))):
        if not row["configuration"].startswith("action_repeat"): continue
        fl = [e for e in folders.values() if e["backbone"] == bb and e["config"] == row["configuration"] and (suite is None or e["leaf"] == suite)]
        eps = [r for e in fl for r in e["episodes"]]
        got = float(row["cycle_median_latency_ms"])
        c = {"summary_median(mean over folders)": mean([e["summary"]["inference_cycle_latency_ms"]["median_ms"] for e in fl]),
             "summary_mean_ms(mean over folders)": mean([e["summary"]["inference_cycle_latency_ms"]["mean_ms"] for e in fl]),
             "median_of_episode_medians": med([r["inference_cycle_ms"]["median_ms"] for r in eps]),
             "mean_of_episode_means": mean([r["inference_cycle_ms"]["mean_ms"] for r in eps]),
             "median_of_episode_means": med([r["inference_cycle_ms"]["mean_ms"] for r in eps])}
        print(f"  {relcsv}:{row['configuration']} csv={got} " + ", ".join(f"{k}={v:.3f}" for k, v in c.items()))

# 8. smolvla success by implementation within mixed folders
print("\n# 8. SmolVLA mixed-implementation folders: success by implementation")
for rel, e in folders.items():
    if e["backbone"] != "smolvla_libero": continue
    c = defaultdict(lambda: [0, 0])
    for r in e["episodes"]:
        k = r.get("implementation") or "legacy(no tag)"; c[k][0] += 1; c[k][1] += r["success"]
    if len(c) > 1:
        print(f"  {rel}: " + "; ".join(f"{k}: {v[1]}/{v[0]}" for k, v in c.items()))

# 9. cogact / spatialvla action repeat: reuses recorded as 0
print("\n# 9. backbones recording reuses=0 under action repeat (repeated steps not counted as reuses)")
for bb in sorted({e["backbone"] for e in folders.values()}):
    eps = [r for e in folders.values() if e["backbone"] == bb and e["config"] == "action_repeat2" for r in e["episodes"]]
    print(f"  {bb}: sum reuses={sum(r['reuses'] for r in eps)}, sum(steps-calls)={sum(r['steps_executed']-r['policy_calls'] for r in eps)}")
