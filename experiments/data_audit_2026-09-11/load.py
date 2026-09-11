import json, os, pickle, csv, statistics, collections, sys

ROOT = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11/results/"
OUT = "/home/user/BiVLA/experiments/data_audit_2026-09-11/"

files = []
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if f == "episodes.jsonl":
            files.append(os.path.join(dp, f))
files.sort()
print("files found:", len(files))

recs = []
per_file = []
bad_lines = 0
keys_seen = collections.Counter()
schema_ids = {}
for fp in files:
    rel = os.path.relpath(fp, ROOT)
    parts = rel.split("/")
    bdir, cfg_dir = parts[0], parts[1]
    task_dir = parts[2] if len(parts) == 4 else None
    n = 0
    with open(fp) as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                bad_lines += 1
                print("BAD LINE", fp, ln, e)
                continue
            ks = tuple(sorted(r.keys()))
            keys_seen[ks] += 1
            if ks not in schema_ids:
                schema_ids[ks] = "S%d" % len(schema_ids)
            r["_schema"] = schema_ids[ks]
            r["_bdir"] = bdir
            r["_cfgdir"] = cfg_dir
            r["_taskdir"] = task_dir
            r["_file"] = rel
            r["_line"] = ln
            r["_cfg"] = r.get("config_name") or cfg_dir
            r["_task"] = r.get("task") or task_dir
            ql = r.get("query_latency_ms")
            ic = r.get("inference_cycle_ms")
            r["ql_mean"] = ql.get("mean_ms") if isinstance(ql, dict) else None
            r["ql_median"] = ql.get("median_ms") if isinstance(ql, dict) else None
            r["ql_p95"] = ql.get("p95_ms") if isinstance(ql, dict) else None
            r["ic_mean"] = ic.get("mean_ms") if isinstance(ic, dict) else None
            r["ic_median"] = ic.get("median_ms") if isinstance(ic, dict) else None
            r["ic_p95"] = ic.get("p95_ms") if isinstance(ic, dict) else None
            recs.append(r)
            n += 1
    per_file.append((rel, n, bdir, cfg_dir, task_dir))

print("total records:", len(recs), "bad lines:", bad_lines)
print("key sets:")
for k, c in keys_seen.items():
    print("  ", schema_ids[k], c, k)

# schema by backbone dir
sb = collections.defaultdict(collections.Counter)
for r in recs:
    sb[r["_bdir"]][r["_schema"]] += 1
print("schema per backbone dir:")
for b in sorted(sb):
    print("  ", b, dict(sb[b]))

# file layout per backbone dir
lay = collections.defaultdict(lambda: collections.Counter())
for rel, n, b, c, t in per_file:
    lay[b][("config-level" if t is None else "task-level", n)] += 1
print("file layout per backbone dir (level, records-per-file): count")
for b in sorted(lay):
    print("  ", b, dict(lay[b]))

# check dir vs field consistency
mism = collections.Counter()
for r in recs:
    if r.get("model_name") is not None and r["model_name"] != r["_bdir"]:
        mism["model_name!=bdir:%s:%s" % (r["_bdir"], r["model_name"])] += 1
    if r.get("config_name") is not None and r["config_name"] != r["_cfgdir"]:
        mism["config_name!=cfgdir:%s/%s:%s" % (r["_bdir"], r["_cfgdir"], r["config_name"])] += 1
    if r["_taskdir"] is not None and r.get("task") != r["_taskdir"]:
        mism["task!=taskdir:%s/%s/%s:%s" % (r["_bdir"], r["_cfgdir"], r["_taskdir"], r.get("task"))] += 1
    if r.get("task") is None:
        mism["task missing:%s/%s" % (r["_bdir"], r["_cfgdir"])] += 1
print("dir/field mismatches:", len(mism))
for k, v in sorted(mism.items()):
    print("  ", v, k)

with open(OUT + "recs.pkl", "wb") as fh:
    pickle.dump(recs, fh)

# ---- per_task_success.csv ----
grp = collections.defaultdict(list)
for r in recs:
    grp[(r["_bdir"], r["_cfg"], r["_task"])].append(r)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def rnd(x, d):
    return None if x is None else round(x, d)

rows = []
for (b, c, t) in sorted(grp, key=lambda k: (k[0], k[1], str(k[2]))):
    g = grp[(b, c, t)]
    n = len(g)
    succ = sum(1 for r in g if r.get("success") is True)
    trunc_known = [r for r in g if "truncated" in r]
    trunc = sum(1 for r in trunc_known if r["truncated"] is True)
    qmeds = [r["ql_median"] for r in g if r["ql_median"] is not None]
    rows.append({
        "backbone_dir": b, "configuration": c, "task": t,
        "task_group": (g[0]["_taskdir"] or t),
        "n": n, "successes": succ, "success_rate": round(succ / n, 4),
        "mean_steps": rnd(mean([r.get("steps_executed") for r in g]), 2),
        "truncation_rate": (round(trunc / len(trunc_known), 4) if trunc_known else None),
        "n_with_truncated_field": len(trunc_known),
        "mean_policy_calls": rnd(mean([r.get("policy_calls") for r in g]), 2),
        "mean_reuses": rnd(mean([r.get("reuses") for r in g]), 2),
        "mean_episode_elapsed_s": rnd(mean([r.get("episode_elapsed_ms") for r in g]), 1) and round(mean([r.get("episode_elapsed_ms") for r in g]) / 1000.0, 3),
        "median_of_episode_query_median_ms": round(statistics.median(qmeds), 3) if qmeds else None,
        "n_with_query_latency": len(qmeds),
    })
with open(OUT + "per_task_success.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("csv rows:", len(rows))

print("backbone dirs:", sorted(set(r["_bdir"] for r in recs)))
print("configs:", sorted(set(r["_cfg"] for r in recs)))
print("conditions:", sorted(set(str(r.get("condition")) for r in recs)))
tasks_by_b = collections.defaultdict(set)
for r in recs:
    tasks_by_b[r["_bdir"]].add(r["_task"])
for b in sorted(tasks_by_b):
    print(b, len(tasks_by_b[b]), sorted(tasks_by_b[b], key=str))
