import pickle, collections, statistics, math, json, sys
import numpy as np

OUT = "/home/user/BiVLA/experiments/data_audit_2026-09-11/"
recs = pickle.load(open(OUT + "recs.pkl", "rb"))
B = sorted(set(r["_bdir"] for r in recs))
C = sorted(set(r["_cfg"] for r in recs))

def grp(keyfn):
    d = collections.defaultdict(list)
    for r in recs:
        d[keyfn(r)].append(r)
    return d

def sr(g):
    return sum(1 for r in g if r.get("success") is True) / len(g)

def tg(r):  # task group (suite for libero, task otherwise)
    return r["_taskdir"] or r["_task"]

def pct(x):
    return "%.0f%%" % (100 * x)

by_bc = grp(lambda r: (r["_bdir"], r["_cfg"]))
by_bct = grp(lambda r: (r["_bdir"], r["_cfg"], tg(r)))
by_bcft = grp(lambda r: (r["_bdir"], r["_cfg"], r["_task"]))

sec = lambda t: print("\n" + "=" * 100 + "\n## " + t + "\n")

# ---------------------------------------------------------------- overall matrix
sec("OVERALL success rate per backbone x config (n episodes)")
print("| backbone | " + " | ".join(c.replace("temporal_fusion_", "tf_").replace("fixed_foveation_", "fov_").replace("guarded_reuse_", "gr_").replace("depth_pruning", "dp").replace("action_repeat", "ar") for c in C) + " |")
print("|---" * (len(C) + 1) + "|")
for b in B:
    cells = []
    for c in C:
        g = by_bc.get((b, c))
        cells.append("%s (%d)" % (pct(sr(g)), len(g)) if g else "MISSING")
    print("| %s | %s |" % (b, " | ".join(cells)))

# ---------------------------------------------------------------- Q1
sec("Q1 per-task original vs config for named collapses")
targets = [
    ("univla_simplerenv_bridge", ["action_repeat2", "action_repeat4"]),
    ("cogact_base_simplerenv_bridge", ["action_repeat2", "action_repeat4"]),
    ("cronusvla_simplerenv_bridge", ["action_repeat2", "action_repeat4"]),
    ("openvla_simplerenv", ["action_repeat2", "action_repeat4"]),
    ("minivla_simplerenv", ["depth_pruning1", "depth_pruning2", "depth_pruning4"]),
    ("spatialvla_simplerenv_bridge", ["depth_pruning1", "depth_pruning2", "depth_pruning4"]),
    ("smolvla_libero", ["depth_pruning1", "depth_pruning2", "depth_pruning4"]),
    ("openvla_simplerenv", ["fixed_foveation_keep20", "fixed_foveation_keep50"]),
    ("openvla_simplerenv_fractal", ["fixed_foveation_keep20", "fixed_foveation_keep50"]),
    ("openvla_libero", ["fixed_foveation_keep20", "fixed_foveation_keep50"]),
]
for b, cfgs in targets:
    tasks = sorted(set(tg(r) for r in recs if r["_bdir"] == b))
    print("\n### %s" % b)
    print("| task_group | original | " + " | ".join(cfgs) + " |")
    print("|---" * (len(cfgs) + 2) + "|")
    for t in tasks:
        o = by_bct.get((b, "original", t))
        row = ["%d/%d=%s" % (sum(r["success"] for r in o), len(o), pct(sr(o))) if o else "n/a"]
        for c in cfgs:
            g = by_bct.get((b, c, t))
            row.append("%d/%d=%s" % (sum(r["success"] for r in g), len(g), pct(sr(g))) if g else "MISSING")
        print("| %s | %s |" % (t, " | ".join(row)))
    o = by_bc[(b, "original")]
    row = ["%d/%d=%s" % (sum(r["success"] for r in o), len(o), pct(sr(o)))]
    for c in cfgs:
        g = by_bc.get((b, c))
        row.append("%d/%d=%s" % (sum(r["success"] for r in g), len(g), pct(sr(g))) if g else "MISSING")
    print("| **ALL** | %s |" % " | ".join(row))

# smolvla libero per fine task for depth pruning (to see if 0% tasks)
print("\n### smolvla_libero depth_pruning per fine task (successes/10), original vs dp1/dp2/dp4")
ft = sorted(set(r["_task"] for r in recs if r["_bdir"] == "smolvla_libero"))
print("| task | orig | dp1 | dp2 | dp4 |\n|---|---|---|---|---|")
for t in ft:
    row = []
    for c in ["original", "depth_pruning1", "depth_pruning2", "depth_pruning4"]:
        g = by_bcft.get(("smolvla_libero", c, t))
        row.append("%d" % sum(r["success"] for r in g) if g else "-")
    print("| %s | %s |" % (t, " | ".join(row)))

# ---------------------------------------------------------------- Q2
sec("Q2 concentration: for every (backbone,config) with drop >= 15pp vs original, per-task-group rates")
print("| backbone | config | orig | cfg | drop | per-task-group cfg rates (orig->cfg) | #groups at 0% (orig>0) | #fine tasks at 0% / total (orig>0) |")
print("|---|---|---|---|---|---|---|---|")
for b in B:
    o = sr(by_bc[(b, "original")])
    for c in C:
        if c == "original" or (b, c) not in by_bc:
            continue
        s = sr(by_bc[(b, c)])
        if o - s >= 0.15:
            tasks = sorted(set(tg(r) for r in by_bc[(b, c)]))
            parts = []
            z = 0
            for t in tasks:
                so = sr(by_bct[(b, "original", t)]) if (b, "original", t) in by_bct else float("nan")
                sc = sr(by_bct[(b, c, t)])
                parts.append("%s %s->%s" % (t.replace("google_robot_", "gr_").replace("widowx_", "wx_"), pct(so), pct(sc)))
                if sc == 0 and so > 0:
                    z += 1
            fts = sorted(set(r["_task"] for r in by_bc[(b, c)]))
            fz = sum(1 for t in fts if sr(by_bcft[(b, c, t)]) == 0 and (b, "original", t) in by_bcft and sr(by_bcft[(b, "original", t)]) > 0)
            print("| %s | %s | %s | %s | %.0fpp | %s | %d/%d | %d/%d |" % (b, c, pct(o), pct(s), 100 * (o - s), "; ".join(parts), z, len(tasks), fz, len(fts)))

# ---------------------------------------------------------------- Q3
sec("Q3 truncation")
has_trunc = collections.Counter((r["_bdir"], "truncated" in r) for r in recs)
print("records with 'truncated' field per backbone:", {b: (has_trunc[(b, True)], has_trunc[(b, False)]) for b in B})
print("\nmax steps_executed per backbone (all configs) and per config; steps distribution of failures:")
print("| backbone | max steps | min steps | configs max steps | share of episodes at max | share of FAILURES at max | share of SUCCESSES at max |")
print("|---|---|---|---|---|---|---|")
capv = {}
for b in B:
    g = [r for r in recs if r["_bdir"] == b]
    mx = max(r["steps_executed"] for r in g)
    mn = min(r["steps_executed"] for r in g)
    capv[b] = mx
    per_c = {c: max(r["steps_executed"] for r in by_bc[(b, c)]) for c in C if (b, c) in by_bc}
    pc = collections.Counter(per_c.values())
    at = sum(1 for r in g if r["steps_executed"] == mx) / len(g)
    fails = [r for r in g if not r["success"]]
    succ = [r for r in g if r["success"]]
    fat = sum(1 for r in fails if r["steps_executed"] == mx) / len(fails) if fails else float("nan")
    sat = sum(1 for r in succ if r["steps_executed"] == mx) / len(succ) if succ else float("nan")
    print("| %s | %d | %d | %s | %.3f | %.3f | %.3f |" % (b, mx, mn, dict(pc), at, fat, sat))

print("\nmax steps per LIBERO suite (per backbone):")
for b in [x for x in B if "libero" in x]:
    d = collections.defaultdict(int)
    for r in recs:
        if r["_bdir"] == b:
            d[tg(r)] = max(d[tg(r)], r["steps_executed"])
    print("  ", b, dict(d))

print("\ntruncated=true fraction per backbone x config (only where field exists); '-' = field absent")
print("| backbone | " + " | ".join(c.replace("temporal_fusion_", "tf_").replace("fixed_foveation_", "fov_").replace("guarded_reuse_", "gr_").replace("depth_pruning", "dp").replace("action_repeat", "ar") for c in C) + " |")
print("|---" * (len(C) + 1) + "|")
for b in B:
    cells = []
    for c in C:
        g = by_bc.get((b, c))
        if not g:
            cells.append("MISSING"); continue
        gt = [r for r in g if "truncated" in r]
        if not gt:
            cells.append("-"); continue
        cells.append("%.2f" % (sum(1 for r in gt if r["truncated"] is True) / len(gt)))
    print("| %s | %s |" % (b, " | ".join(cells)))

print("\nconsistency of truncated flag with steps==cap (records with field):")
cc = collections.Counter()
for r in recs:
    if "truncated" in r:
        cap = capv[r["_bdir"]]
        cc[(r["_bdir"], r["truncated"], r["steps_executed"] == cap, r["success"])] += 1
for k in sorted(cc):
    print("   %s truncated=%s steps==cap=%s success=%s : %d" % (k[0], k[1], k[2], k[3], cc[k]))

print("\nruns (backbone,config) where EVERY failure is a truncation (truncated=true where field exists, else steps==cap):")
print("| backbone | config | n fail | n fail truncated | all-fail-truncated? | inferred? |")
print("|---|---|---|---|---|---|")
allft = []
for b in B:
    for c in C:
        g = by_bc.get((b, c))
        if not g: continue
        fails = [r for r in g if not r["success"]]
        inferred = not all("truncated" in r for r in fails)
        if inferred:
            nt = sum(1 for r in fails if r["steps_executed"] == capv[b])
        else:
            nt = sum(1 for r in fails if r["truncated"] is True)
        flag = (nt == len(fails)) and len(fails) > 0
        allft.append((b, c, len(fails), nt, flag, inferred))
for b, c, nf, nt, flag, inf in allft:
    if flag:
        print("| %s | %s | %d | %d | YES | %s |" % (b, c, nf, nt, "inferred(steps==cap)" if inf else "flag"))
print("count of runs with all failures truncated:", sum(1 for x in allft if x[4]), "of", len(allft))
print("\nruns where truncated share of failures < 100%% (i.e. some early-terminated failures):")
print("| backbone | config | n fail | truncated fails | share |")
print("|---|---|---|---|---|")
for b, c, nf, nt, flag, inf in allft:
    if nf and not flag:
        print("| %s | %s | %d | %d | %.2f |" % (b, c, nf, nt, nt / nf))

# ---------------------------------------------------------------- Q4
sec("Q4 errors")
errs = [r for r in recs if r.get("error") is not None]
print("records with error not null:", len(errs))
ec = collections.Counter((r["_bdir"], r["_cfg"], str(r["error"])[:200]) for r in errs)
for k, v in sorted(ec.items()):
    print("  ", v, k)
tbs = [r for r in recs if r.get("error_traceback") is not None]
print("records with error_traceback not null:", len(tbs))
# error field types
print("error field value types:", collections.Counter(type(r.get("error")).__name__ for r in recs))

# ---------------------------------------------------------------- Q5
sec("Q5 latency")
print("records with query_latency_ms per backbone:", {b: sum(1 for r in recs if r["_bdir"] == b and r["ql_median"] is not None) for b in B})
print("records with inference_cycle_ms per backbone:", {b: sum(1 for r in recs if r["_bdir"] == b and r["ic_median"] is not None) for b in B})
print("\nORIGINAL per-episode query_latency median distribution per backbone:")
print("| backbone | n | min | p10 | median | p90 | max | #unique | #zeros | #>3x median | elapsed/policy_call median (ms) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for b in B:
    g = by_bc[(b, "original")]
    v = [r["ql_median"] for r in g if r["ql_median"] is not None]
    epc = [r["episode_elapsed_ms"] / r["policy_calls"] for r in g if r["policy_calls"]]
    if not v:
        print("| %s | 0 | - | - | - | - | - | - | - | - | %.1f |" % (b, statistics.median(epc)))
        continue
    a = np.array(v)
    med = np.median(a)
    print("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %d | %d | %d | %.1f |" % (
        b, len(v), a.min(), np.percentile(a, 10), med, np.percentile(a, 90), a.max(), len(set(v)), int((a == 0).sum()), int((a > 3 * med).sum()), statistics.median(epc)))

print("\nSUSPICIOUS runs (backbone,config): identical ql_median across all episodes, zeros, or outliers >3x run median; also ql_median > ql_mean anomalies, p95<median")
print("| backbone | config | n | #unique ql_median | #zeros | #>3x | max/median | #p95<median | #median>mean |")
print("|---|---|---|---|---|---|---|---|---|")
for b in B:
    for c in C:
        g = by_bc.get((b, c))
        if not g: continue
        v = [r["ql_median"] for r in g if r["ql_median"] is not None]
        if not v: continue
        a = np.array(v)
        med = np.median(a)
        nz = int((a == 0).sum())
        no = int((a > 3 * med).sum()) if med > 0 else 0
        nu = len(set(v))
        p95lt = sum(1 for r in g if r["ql_p95"] is not None and r["ql_median"] is not None and r["ql_p95"] < r["ql_median"])
        mgm = sum(1 for r in g if r["ql_mean"] is not None and r["ql_median"] is not None and r["ql_median"] > r["ql_mean"] * 1.0001)
        if nu <= 3 or nz or no or p95lt or (a.max() / med > 3 if med > 0 else False):
            print("| %s | %s | %d | %d | %d | %d | %.2f | %d | %d |" % (b, c, len(v), nu, nz, no, a.max() / med if med > 0 else float("nan"), p95lt, mgm))

print("\nPer-backbone median of per-episode ql_median: original vs depth_pruning1/2/4 vs temporal_fusion_task_aware vs other tf; also inference_cycle median, and elapsed/policy_call")
cols = ["original", "depth_pruning1", "depth_pruning2", "depth_pruning4", "temporal_fusion_task_aware", "temporal_fusion_conservative_adaptive", "temporal_fusion_motion_entropy", "fixed_foveation_keep20", "fixed_foveation_keep50", "guarded_reuse_strict", "guarded_reuse_moderate", "guarded_reuse_aggressive", "action_repeat2", "action_repeat4"]
for metric, fn in [("ql_median (ms)", lambda r: r["ql_median"]), ("inference_cycle median (ms)", lambda r: r["ic_median"]), ("episode_elapsed_ms/policy_calls (ms)", lambda r: r["episode_elapsed_ms"] / r["policy_calls"] if r["policy_calls"] else None), ("episode_elapsed_ms/steps_executed (ms)", lambda r: r["episode_elapsed_ms"] / r["steps_executed"] if r["steps_executed"] else None)]:
    print("\n#### median over episodes of %s" % metric)
    print("| backbone | " + " | ".join(c.replace("temporal_fusion_", "tf_").replace("fixed_foveation_", "fov_").replace("guarded_reuse_", "gr_").replace("depth_pruning", "dp").replace("action_repeat", "ar") for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for b in B:
        cells = []
        for c in cols:
            g = by_bc.get((b, c))
            if not g: cells.append("MISSING"); continue
            v = [fn(r) for r in g]
            v = [x for x in v if x is not None]
            cells.append("%.1f" % statistics.median(v) if v else "-")
        print("| %s | %s |" % (b, " | ".join(cells)))

print("\nratio of config median ql_median to original (per backbone):")
print("| backbone | dp1 | dp2 | dp4 | tf_task_aware | tf_cons_adaptive | tf_motion_entropy | fov20 | fov50 |")
print("|---|---|---|---|---|---|---|---|---|")
for b in B:
    o = [r["ql_median"] for r in by_bc[(b, "original")] if r["ql_median"] is not None]
    if not o:
        print("| %s | no query_latency field |" % b); continue
    om = statistics.median(o)
    cells = []
    for c in ["depth_pruning1", "depth_pruning2", "depth_pruning4", "temporal_fusion_task_aware", "temporal_fusion_conservative_adaptive", "temporal_fusion_motion_entropy", "fixed_foveation_keep20", "fixed_foveation_keep50"]:
        g = by_bc.get((b, c))
        v = [r["ql_median"] for r in g if r["ql_median"] is not None] if g else []
        cells.append("%.3f" % (statistics.median(v) / om) if v else "-")
    print("| %s | %s |" % (b, " | ".join(cells)))

# smolvla: schema split and latency by schema
print("\nsmolvla_libero schema per config (S3 = plain, S4 = with slurm/gpu metadata):")
sc = collections.defaultdict(collections.Counter)
for r in recs:
    if r["_bdir"] == "smolvla_libero":
        sc[r["_cfg"]][(r["_schema"], tg(r))] += 1
for c in C:
    print("  ", c, dict(sc[c]))
print("smolvla S4 metadata values:")
for f in ["gpu", "implementation", "qos", "selected_depth_layers", "slurm_job_id"]:
    print("  ", f, collections.Counter(json.dumps(r.get(f)) for r in recs if r["_bdir"] == "smolvla_libero" and r["_schema"] == "S4").most_common(12))
print("  recorded_at_utc range:", min(r["recorded_at_utc"] for r in recs if r.get("recorded_at_utc")), max(r["recorded_at_utc"] for r in recs if r.get("recorded_at_utc")))
print("  fusion_reusable_tokens_median:", collections.Counter((r["_cfg"], json.dumps(r.get("fusion_reusable_tokens_median"))) for r in recs if r["_bdir"] == "smolvla_libero" and r["_schema"] == "S4").most_common(20))

# cogact/univla/spatialvla benchmark_protocol
print("\nbenchmark_protocol values:", collections.Counter((r["_bdir"], json.dumps(r.get("benchmark_protocol"))[:150]) for r in recs if "benchmark_protocol" in r).most_common(30))
print("benchmark_episode_id sample:", [r["benchmark_episode_id"] for r in recs if "benchmark_episode_id" in r][:3])
# cronusvla temporal_fused_patches
print("\ncronusvla temporal_fused_patches per config:", )
for c in C:
    v = [r["temporal_fused_patches"] for r in recs if r["_bdir"].startswith("cronusvla") and r["_cfg"] == c]
    print("   ", c, collections.Counter(json.dumps(x)[:60] for x in v).most_common(4))

# ---------------------------------------------------------------- Q6
sec("Q6 reuses")
print("| backbone | config | n | max reuses | mean reuses | share zero reuses | mean policy_calls | mean steps | mean (steps - policy_calls) | reuses==steps-policy_calls share |")
print("|---|---|---|---|---|---|---|---|---|---|")
for b in B:
    for c in ["guarded_reuse_strict", "guarded_reuse_moderate", "guarded_reuse_aggressive", "action_repeat2", "action_repeat4"]:
        g = by_bc.get((b, c))
        if not g: print("| %s | %s | MISSING |" % (b, c)); continue
        ru = [r["reuses"] for r in g]
        eq = sum(1 for r in g if r["reuses"] == r["steps_executed"] - r["policy_calls"]) / len(g)
        print("| %s | %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
            b, c, len(g), max(ru), statistics.mean(ru), sum(1 for x in ru if x == 0) / len(ru),
            statistics.mean(r["policy_calls"] for r in g), statistics.mean(r["steps_executed"] for r in g),
            statistics.mean(r["steps_executed"] - r["policy_calls"] for r in g), eq))
print("\nreuses in NON-reuse configs (original, dp, fov, tf): any nonzero?")
rc = collections.Counter((r["_bdir"], r["_cfg"]) for r in recs if r["reuses"] != 0 and not (r["_cfg"].startswith("guarded_reuse") or r["_cfg"].startswith("action_repeat")))
for k, v in sorted(rc.items()):
    g = by_bc[k]
    print("   ", k, v, "of", len(g), "mean reuses %.2f" % statistics.mean(r["reuses"] for r in g))
print("\nreuse share (reuses/steps) per backbone for guarded_reuse, mean over episodes:")
for b in B:
    print("   ", b, {c: "%.3f" % statistics.mean(r["reuses"] / r["steps_executed"] for r in by_bc[(b, c)] if r["steps_executed"]) for c in ["guarded_reuse_strict", "guarded_reuse_moderate", "guarded_reuse_aggressive"] if (b, c) in by_bc})

# ---------------------------------------------------------------- Q7
sec("Q7 seeds and pairing")
seed_present = [r for r in recs if "seed" in r]
seed_null = [r for r in recs if "seed" in r and r["seed"] is None]
seed_absent = [r for r in recs if "seed" not in r]
print("seed field present:", len(seed_present), "null:", len(seed_null), "absent:", len(seed_absent))
print("seed absent per (backbone):", collections.Counter(r["_bdir"] for r in seed_absent))
print("seed null per (backbone,config):", collections.Counter((r["_bdir"], r["_cfg"]) for r in seed_null))
bad = [r for r in seed_present if r["seed"] is not None and r["seed"] != 42 + r["episode_index"]]
print("seed != 42+episode_index:", len(bad))
print("  by run:", collections.Counter((r["_bdir"], r["_cfg"]) for r in bad).most_common(20))
if bad:
    print("  examples:", [(r["_bdir"], r["_cfg"], r["_task"], r["episode_index"], r["seed"]) for r in bad[:10]])
print("episode_index range per backbone:", {b: (min(r["episode_index"] for r in recs if r["_bdir"] == b), max(r["episode_index"] for r in recs if r["_bdir"] == b)) for b in B})
# task_index
ti = [r for r in recs if "task_index" in r]
print("task_index present:", len(ti), "; per backbone:", collections.Counter(r["_bdir"] for r in ti))
tim = collections.defaultdict(set)
for r in ti:
    tim[(r["_bdir"], r["_task"])].add(r["task_index"])
multi = {k: v for k, v in tim.items() if len(v) > 1}
print("tasks with >1 task_index value:", len(multi), list(multi.items())[:10])
print("task_index -> task map sample per backbone:")
for b in B:
    m = sorted(set((r["task_index"], r["_task"]) for r in ti if r["_bdir"] == b))
    if m: print("   ", b, m[:12], "..." if len(m) > 12 else "")

print("\n(task, episode_index) set identical across configs per backbone?")
for b in B:
    sets = {c: set((r["_task"], r["episode_index"]) for r in by_bc[(b, c)]) for c in C if (b, c) in by_bc}
    ref = sets["original"]
    diff = {c: (len(s - ref), len(ref - s)) for c, s in sets.items() if s != ref}
    print("   %s: original has %d pairs; configs differing: %s" % (b, len(ref), diff if diff else "none"))

print("\nduplicate (backbone, config, task, episode_index) records:")
dc = collections.Counter((r["_bdir"], r["_cfg"], r["_task"], r["episode_index"]) for r in recs)
dups = {k: v for k, v in dc.items() if v > 1}
print("   count of duplicated keys:", len(dups), "; total extra records:", sum(v - 1 for v in dups.values()))
for k, v in list(sorted(dups.items()))[:20]:
    print("   ", v, k)
# are duplicates identical records?
if dups:
    ident = 0
    for k in dups:
        rs = [r for r in recs if (r["_bdir"], r["_cfg"], r["_task"], r["episode_index"]) == k]
        core = set(json.dumps({kk: vv for kk, vv in r.items() if not kk.startswith("_") and kk not in ("ql_mean", "ql_median", "ql_p95", "ic_mean", "ic_median", "ic_p95")}, sort_keys=True) for r in rs)
        if len(core) == 1: ident += 1
    print("   duplicated keys whose records are byte-identical:", ident)
    print("   duplicates by (backbone,config):", collections.Counter((k[0], k[1]) for k in dups))
    print("   duplicates by file:", collections.Counter(r["_file"] for r in recs if (r["_bdir"], r["_cfg"], r["_task"], r["episode_index"]) in dups).most_common(10))

# ---------------------------------------------------------------- Q8
sec("Q8 other anomalies")
lt = [r for r in recs if r["steps_executed"] < r["policy_calls"]]
print("steps_executed < policy_calls:", len(lt), collections.Counter((r["_bdir"], r["_cfg"]) for r in lt).most_common(10))
z = [r for r in recs if r["steps_executed"] == 0]
print("steps_executed == 0:", len(z), collections.Counter((r["_bdir"], r["_cfg"]) for r in z).most_common(10))
z = [r for r in recs if r["policy_calls"] == 0]
print("policy_calls == 0:", len(z), collections.Counter((r["_bdir"], r["_cfg"]) for r in z).most_common(10))
print("\naction_repeat k consistency: expected policy_calls == ceil(steps/k) and reuses == steps - policy_calls")
print("| backbone | config | n | pc==ceil(steps/k) share | pc==floor(steps/k) share | reuses==steps-pc share | pc==steps share | example (steps,pc,reuses) |")
print("|---|---|---|---|---|---|---|---|")
for b in B:
    for c, k in [("action_repeat2", 2), ("action_repeat4", 4)]:
        g = by_bc.get((b, c))
        if not g: continue
        ce = sum(1 for r in g if r["policy_calls"] == math.ceil(r["steps_executed"] / k)) / len(g)
        fl = sum(1 for r in g if r["policy_calls"] == r["steps_executed"] // k) / len(g)
        re = sum(1 for r in g if r["reuses"] == r["steps_executed"] - r["policy_calls"]) / len(g)
        eq = sum(1 for r in g if r["policy_calls"] == r["steps_executed"]) / len(g)
        ex = collections.Counter((r["steps_executed"], r["policy_calls"], r["reuses"]) for r in g).most_common(3)
        print("| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %s |" % (b, c, len(g), ce, fl, re, eq, ex))
print("\nnon-action-repeat configs: policy_calls + reuses == steps_executed share; policy_calls==steps share")
print("| backbone | config | pc+reuses==steps | pc==steps | pc>steps |")
print("|---|---|---|---|---|")
for b in B:
    for c in C:
        g = by_bc.get((b, c))
        if not g or c.startswith("action_repeat"): continue
        a = sum(1 for r in g if r["policy_calls"] + r["reuses"] == r["steps_executed"]) / len(g)
        e = sum(1 for r in g if r["policy_calls"] == r["steps_executed"]) / len(g)
        gt = sum(1 for r in g if r["policy_calls"] > r["steps_executed"]) / len(g)
        if a < 1 or gt > 0:
            print("| %s | %s | %.2f | %.2f | %.2f |" % (b, c, a, e, gt))

print("\nsuccess field types:", collections.Counter(type(r.get("success")).__name__ for r in recs))
print("truncated field types:", collections.Counter(type(r.get("truncated")).__name__ for r in recs if "truncated" in r))
print("steps types:", collections.Counter(type(r.get("steps_executed")).__name__ for r in recs))
print("successes with steps==cap per backbone:", {b: sum(1 for r in recs if r["_bdir"] == b and r["success"] and r["steps_executed"] == capv[b]) for b in B})
print("successes with truncated=True:", sum(1 for r in recs if r.get("truncated") is True and r["success"]))
print("failures with truncated=False (early termination) per backbone:", {b: sum(1 for r in recs if r["_bdir"] == b and r.get("truncated") is False and not r["success"]) for b in B})
print("failures steps distribution (non-cap) per backbone: min/median/max of steps among non-truncated failures")
for b in B:
    v = [r["steps_executed"] for r in recs if r["_bdir"] == b and not r["success"] and r["steps_executed"] != capv[b]]
    if v: print("   ", b, len(v), min(v), statistics.median(v), max(v))
print("\nsuccess steps distribution per backbone (min/median/max):")
for b in B:
    v = [r["steps_executed"] for r in recs if r["_bdir"] == b and r["success"]]
    if v: print("   ", b, len(v), min(v), statistics.median(v), max(v))

print("\nLIBERO: fine tasks per suite per backbone per config (expect 10):")
for b in [x for x in B if "libero" in x]:
    d = collections.defaultdict(set)
    for r in recs:
        if r["_bdir"] == b:
            d[(r["_cfg"], tg(r))].add(r["_task"])
    cnt = collections.Counter(len(v) for v in d.values())
    print("   ", b, "suites x configs:", len(d), "task-count distribution:", dict(cnt))
    suites = collections.defaultdict(set)
    for (c, s) in d: suites[c].add(s)
    for c in C:
        if len(suites[c]) != 4: print("      config %s has suites %s" % (c, sorted(suites[c])))

print("\nepisodes per (backbone, config, task): distribution")
ec = collections.Counter(len(g) for g in by_bcft.values())
print("   ", dict(ec))
for k, g in by_bcft.items():
    if len(g) != 10: print("    non-10:", k, len(g))

print("\nepisode_elapsed_ms anomalies: <=0 or > 10x run median")
for b in B:
    for c in C:
        g = by_bc.get((b, c))
        if not g: continue
        v = np.array([r["episode_elapsed_ms"] for r in g])
        med = np.median(v)
        nz = int((v <= 0).sum()); no = int((v > 10 * med).sum())
        if nz or no: print("   ", b, c, "n<=0:", nz, "n>10x:", no, "max/med %.1f" % (v.max() / med))

print("\nelapsed per step (ms) median per backbone for original, and elapsed vs (policy_calls*ql_mean) ratio:")
for b in B:
    g = by_bc[(b, "original")]
    r1 = statistics.median(r["episode_elapsed_ms"] / r["steps_executed"] for r in g)
    rat = [r["episode_elapsed_ms"] / (r["policy_calls"] * r["ql_mean"]) for r in g if r["ql_mean"]]
    print("   %s elapsed/step=%.1f ms; elapsed/(pc*ql_mean) median=%s" % (b, r1, ("%.2f" % statistics.median(rat)) if rat else "n/a"))

print("\ncondition field vs config prefix mismatches:")
cm = collections.Counter((r["_cfg"], r.get("condition")) for r in recs)
for k, v in sorted(cm.items(), key=str): print("   ", v, k)
