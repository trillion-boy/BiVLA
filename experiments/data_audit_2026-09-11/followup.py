import pickle, collections, statistics, json
import numpy as np
OUT = "/home/user/BiVLA/experiments/data_audit_2026-09-11/"
recs = pickle.load(open(OUT + "recs.pkl", "rb"))
B = sorted(set(r["_bdir"] for r in recs))
C = sorted(set(r["_cfg"] for r in recs))
tg = lambda r: r["_taskdir"] or r["_task"]
sec = lambda t: print("\n" + "=" * 90 + "\n## " + t + "\n")

# ---- per-task cap
sec("A. per (backbone, task_group) max steps (cap), and per-config max for that task")
cap = {}
for b in B:
    tasks = sorted(set(tg(r) for r in recs if r["_bdir"] == b))
    for t in tasks:
        g = [r for r in recs if r["_bdir"] == b and tg(r) == t]
        mx = max(r["steps_executed"] for r in g)
        cap[(b, t)] = mx
        pc = {}
        for c in C:
            gg = [r for r in g if r["_cfg"] == c]
            if gg: pc[c] = max(r["steps_executed"] for r in gg)
        low = {c: v for c, v in pc.items() if v != mx}
        print("  %-32s %-48s cap=%4d  configs below cap: %s" % (b, t, mx, low if low else "none"))

def capof(r): return cap[(r["_bdir"], tg(r))]

sec("B. truncated flag vs steps==per-task-cap vs success (records with flag)")
cc = collections.Counter()
for r in recs:
    if "truncated" in r:
        cc[(r["_bdir"], r["truncated"], r["steps_executed"] == capof(r), r["success"])] += 1
print("| backbone | truncated | steps==cap | success | n |\n|---|---|---|---|---|")
for k in sorted(cc): print("| %s | %s | %s | %s | %d |" % (k + (cc[k],)))

sec("B2. fractal dirs: truncated=True vs False at steps==cap, by task and config")
for b in [x for x in B if "fractal" in x and x.startswith(("cogact", "openvla", "spatialvla"))]:
    d = collections.Counter()
    for r in recs:
        if r["_bdir"] == b and r["steps_executed"] == capof(r) and not r["success"]:
            d[(tg(r), r["truncated"])] += 1
    print("  ", b, {k: v for k, v in sorted(d.items())})
    d2 = collections.Counter()
    for r in recs:
        if r["_bdir"] == b and r["steps_executed"] == capof(r) and not r["success"] and r["truncated"]:
            d2[r["_cfg"]] += 1
    print("     truncated=True by config:", dict(sorted(d2.items())))

sec("B3. successes with truncated=True (18) and successes at cap")
for r in recs:
    if r.get("truncated") is True and r["success"]:
        print("   ", r["_bdir"], r["_cfg"], r["_task"], "ep", r["episode_index"], "steps", r["steps_executed"], "cap", capof(r))
print("successes at per-task cap by (backbone, task):")
d = collections.Counter((r["_bdir"], tg(r)) for r in recs if r["success"] and r["steps_executed"] == capof(r))
for k, v in sorted(d.items()): print("   ", v, k)

sec("C. truncation recomputed as steps==per-task cap (failures only): per backbone x config share of failures at cap; and share of all episodes at cap")
short = lambda c: c.replace("temporal_fusion_", "tf_").replace("fixed_foveation_", "fov_").replace("guarded_reuse_", "gr_").replace("depth_pruning", "dp").replace("action_repeat", "ar")
print("share of FAILURES with steps==cap")
print("| backbone | " + " | ".join(short(c) for c in C) + " |")
print("|---" * (len(C) + 1) + "|")
allcap = []
for b in B:
    cells = []
    for c in C:
        g = [r for r in recs if r["_bdir"] == b and r["_cfg"] == c]
        if not g: cells.append("MISSING"); continue
        f = [r for r in g if not r["success"]]
        n = sum(1 for r in f if r["steps_executed"] == capof(r))
        cells.append("%.2f" % (n / len(f)) if f else "-")
        allcap.append((b, c, len(f), n))
    print("| %s | %s |" % (b, " | ".join(cells)))
print("\nshare of ALL EPISODES with steps==cap (step-cap hit rate)")
print("| backbone | " + " | ".join(short(c) for c in C) + " |")
print("|---" * (len(C) + 1) + "|")
for b in B:
    cells = []
    for c in C:
        g = [r for r in recs if r["_bdir"] == b and r["_cfg"] == c]
        if not g: cells.append("MISSING"); continue
        n = sum(1 for r in g if r["steps_executed"] == capof(r))
        cells.append("%.2f" % (n / len(g)))
    print("| %s | %s |" % (b, " | ".join(cells)))
print("\nruns where every failure has steps==cap:", sum(1 for b, c, nf, n in allcap if nf and n == nf), "of", len(allcap))
print("per backbone count of such runs:", collections.Counter(b for b, c, nf, n in allcap if nf and n == nf))
print("runs where NO failure is at cap:", [(b, c) for b, c, nf, n in allcap if nf and n == 0])
print("\nfailure steps when NOT at cap (per backbone): n, min, median, max")
for b in B:
    v = [r["steps_executed"] for r in recs if r["_bdir"] == b and not r["success"] and r["steps_executed"] != capof(r)]
    if v: print("   ", b, len(v), min(v), statistics.median(v), max(v))
    else: print("   ", b, 0)

sec("D. OOM errors anatomy (univla_simplerenv_bridge temporal_fusion_task_aware)")
errs = [r for r in recs if r.get("error") is not None]
print("by task:", collections.Counter(r["_task"] for r in errs))
print("episode_index of errors by task:", {t: sorted(r["episode_index"] for r in errs if r["_task"] == t) for t in set(r["_task"] for r in errs)})
print("steps_executed:", collections.Counter(r["steps_executed"] for r in errs))
print("policy_calls:", collections.Counter(r["policy_calls"] for r in errs))
print("success:", collections.Counter(r["success"] for r in errs), "truncated:", collections.Counter(r["truncated"] for r in errs))
print("elapsed_ms median:", statistics.median(r["episode_elapsed_ms"] for r in errs))
print("ql_median present:", sum(1 for r in errs if r["ql_median"] is not None), "sample:", [round(r["ql_median"], 1) for r in errs[:5]])
g = [r for r in recs if r["_bdir"] == "univla_simplerenv_bridge" and r["_cfg"] == "temporal_fusion_task_aware"]
ok = [r for r in g if r["error"] is None]
print("task_aware success excluding errored episodes: %d/%d = %.1f%%" % (sum(r["success"] for r in ok), len(ok), 100 * sum(r["success"] for r in ok) / len(ok)))
for t in sorted(set(r["_task"] for r in g)):
    gt = [r for r in ok if r["_task"] == t]
    go = [r for r in recs if r["_bdir"] == "univla_simplerenv_bridge" and r["_cfg"] == "original" and r["_task"] == t and r["episode_index"] in set(x["episode_index"] for x in gt)]
    print("   %-32s non-error eps %d: task_aware %d succ (%.0f%%) vs original same eps %d succ (%.0f%%)" % (t, len(gt), sum(r["success"] for r in gt), 100 * sum(r["success"] for r in gt) / max(1, len(gt)), sum(r["success"] for r in go), 100 * sum(r["success"] for r in go) / max(1, len(go))))
print("traceback tail sample:", errs[0]["error_traceback"][-300:].replace("\n", " | "))

sec("E. univla chunking: steps/policy_calls in original and other configs")
for b in ["univla_libero", "univla_simplerenv_bridge"]:
    for c in ["original", "guarded_reuse_aggressive", "action_repeat2", "action_repeat4", "depth_pruning4"]:
        g = [r for r in recs if r["_bdir"] == b and r["_cfg"] == c]
        ratio = [r["steps_executed"] / r["policy_calls"] for r in g]
        exact = collections.Counter((r["steps_executed"], r["policy_calls"], r["reuses"]) for r in g).most_common(4)
        # check pc == ceil(steps / chunk)
        import math
        for chunk in [5, 8, 10, 12, 16, 20, 40]:
            share = sum(1 for r in g if r["policy_calls"] == math.ceil(r["steps_executed"] / chunk)) / len(g)
            if share > 0.5: print("      chunk=%d fits %.2f" % (chunk, share))
        print("   %s %s steps/pc median %.2f min %.2f max %.2f; top (steps,pc,reuses): %s" % (b, c, statistics.median(ratio), min(ratio), max(ratio), exact))

sec("F. smolvla latency by schema/GPU per config")
d = collections.defaultdict(list)
for r in recs:
    if r["_bdir"] == "smolvla_libero":
        d[(r["_cfg"], r.get("gpu", "S3-no-gpu-field"))].append(r["ql_median"])
print("| config | gpu | n | median ql_median |\n|---|---|---|---|")
for k in sorted(d): print("| %s | %s | %d | %.1f |" % (k[0], k[1], len(d[k]), statistics.median(d[k])))
d = collections.defaultdict(list)
for r in recs:
    if r["_bdir"] == "smolvla_libero":
        d[(r["_cfg"], r.get("gpu", "S3-no-gpu-field"))].append(r["success"])
print("smolvla success by (config, gpu):")
for k in sorted(d): print("   %s %s n=%d succ=%.2f" % (k[0], k[1], len(d[k]), sum(d[k]) / len(d[k])))
# selected_depth_layers vs config
print("selected_depth_layers by config:", collections.Counter((r["_cfg"], json.dumps(r.get("selected_depth_layers"))) for r in recs if r["_bdir"] == "smolvla_libero" and "selected_depth_layers" in r))

sec("G. benchmark_episode_id vs episode_index")
s5 = [r for r in recs if "benchmark_episode_id" in r]
print("bench_id == episode_index:", sum(1 for r in s5 if r["benchmark_episode_id"] == r["episode_index"]), "of", len(s5))
print("bench_id range per backbone:", {b: (min(r["benchmark_episode_id"] for r in s5 if r["_bdir"] == b), max(r["benchmark_episode_id"] for r in s5 if r["_bdir"] == b)) for b in set(r["_bdir"] for r in s5)})
ex = [(r["_bdir"], r["_task"], r["episode_index"], r["benchmark_episode_id"]) for r in s5 if r["benchmark_episode_id"] != r["episode_index"]][:5]
print("examples where differ:", ex)

sec("H. openvla_libero temporal_fusion_conservative_adaptive missing suite; like-for-like comparison on 3 suites")
for c in ["original", "temporal_fusion_conservative_adaptive", "temporal_fusion_motion_entropy", "temporal_fusion_task_aware"]:
    g = [r for r in recs if r["_bdir"] == "openvla_libero" and r["_cfg"] == c and tg(r) != "libero_10"]
    print("   %s on goal/object/spatial: %d/%d = %.1f%%" % (c, sum(r["success"] for r in g), len(g), 100 * sum(r["success"] for r in g) / len(g)))

sec("I. per-episode elapsed vs cap check; action_repeat wallclock speedup per step (elapsed/step ratio config/original)")
for b in B:
    o = statistics.median(r["episode_elapsed_ms"] / r["steps_executed"] for r in recs if r["_bdir"] == b and r["_cfg"] == "original")
    a2 = statistics.median(r["episode_elapsed_ms"] / r["steps_executed"] for r in recs if r["_bdir"] == b and r["_cfg"] == "action_repeat2")
    a4 = statistics.median(r["episode_elapsed_ms"] / r["steps_executed"] for r in recs if r["_bdir"] == b and r["_cfg"] == "action_repeat4")
    print("   %-32s per-step time ratio ar2/orig=%.2f (ideal 0.50) ar4/orig=%.2f (ideal 0.25)" % (b, a2 / o, a4 / o))

sec("J. per-task success for ALL collapse configs at fine-task level: count of fine tasks at 0% under config where original >= 50%")
for b in ["univla_libero", "openvla_libero", "smolvla_libero"]:
    for c in ["action_repeat2", "action_repeat4", "depth_pruning4", "fixed_foveation_keep20"]:
        ft = sorted(set(r["_task"] for r in recs if r["_bdir"] == b))
        z = []
        for t in ft:
            go = [r for r in recs if r["_bdir"] == b and r["_cfg"] == "original" and r["_task"] == t]
            gc = [r for r in recs if r["_bdir"] == b and r["_cfg"] == c and r["_task"] == t]
            if gc and sum(r["success"] for r in gc) == 0 and sum(r["success"] for r in go) >= 5:
                z.append(t)
        print("   %s %s: %d fine tasks at 0%% (orig>=50%%): %s" % (b, c, len(z), z[:12]))
