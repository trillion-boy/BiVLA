#!/usr/bin/env python3
"""Read-only verification of mentor_2026-09-11 results. Prints a report."""
import csv, json, math, os, re, statistics, sys
from collections import defaultdict, Counter, OrderedDict

ROOT = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11"
RES = os.path.join(ROOT, "results")
SUMM = os.path.join(ROOT, "summarization")

TOL_RATE = 0.01
TOL_MEAN = 0.006
TOL_LAT = 0.5

LAYERS = {  # backbone folder prefix -> decoder layer count
    "openvla": 32, "cogact": 32, "spatialvla": 26, "univla": 32,
    "minivla": 24, "cronusvla": 12, "smolvla": 16,
}
def nlayers(backbone):
    for k, v in LAYERS.items():
        if backbone.startswith(k):
            return v
    raise KeyError(backbone)

def med(xs):
    return statistics.median(xs) if xs else None
def mean(xs):
    return sum(xs) / len(xs) if xs else None

# ---------------------------------------------------------------- load all
folders = OrderedDict()   # rel path -> dict(summary=..., episodes=[...])
n_summary = n_eps_files = n_records = 0
bad_json = []
for dirpath, dirnames, filenames in os.walk(RES):
    dirnames.sort()
    if "summary.json" in filenames or "episodes.jsonl" in filenames:
        rel = os.path.relpath(dirpath, RES)
        parts = rel.split("/")
        backbone, config = parts[0], parts[1]
        leaf = parts[2] if len(parts) > 2 else None
        entry = {"backbone": backbone, "config": config, "leaf": leaf, "path": dirpath}
        if "summary.json" in filenames:
            n_summary += 1
            with open(os.path.join(dirpath, "summary.json")) as f:
                entry["summary"] = json.load(f)
        else:
            entry["summary"] = None
        if "episodes.jsonl" in filenames:
            n_eps_files += 1
            recs = []
            with open(os.path.join(dirpath, "episodes.jsonl")) as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except Exception as e:
                        bad_json.append((rel, ln, str(e)))
            entry["episodes"] = recs
            n_records += len(recs)
        else:
            entry["episodes"] = None
        folders[rel] = entry

print("# FILE COUNTS")
print(f"summary.json opened: {n_summary}")
print(f"episodes.jsonl opened: {n_eps_files}")
print(f"episode records parsed: {n_records}")
print(f"unparseable lines: {len(bad_json)} {bad_json[:5]}")
missing = [r for r, e in folders.items() if e["summary"] is None or e["episodes"] is None]
print(f"folders missing one of the two files: {missing}")
per_backbone = Counter(e["backbone"] for e in folders.values())
print("folders per backbone_env:", dict(per_backbone))
csvs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(SUMM) for f in fs if f.endswith(".csv"))
print(f"CSV files: {len(csvs)}")

# ---------------------------------------------------------------- Lens A.1: summary vs episodes
print("\n# LENS A.1  summary.json vs episodes.jsonl (per folder)")
mismatches = []
status_bad = []
errors_nonzero = []
field_presence = defaultdict(Counter)
latency_median_maxdev = defaultdict(float)
latency_mean_check = defaultdict(list)

def cmp(rel, field, got, exp, tol):
    if got is None or exp is None:
        return
    if abs(got - exp) > tol:
        mismatches.append((rel, field, got, exp, got - exp))

for rel, e in folders.items():
    s, eps = e["summary"], e["episodes"]
    bb = e["backbone"]
    if s.get("status") != "completed":
        status_bad.append((rel, s.get("status")))
    if s.get("episode_errors") not in (None, 0):
        errors_nonzero.append((rel, "summary.episode_errors", s.get("episode_errors")))
    err_recs = [r for r in eps if r.get("error") not in (None, "")]
    if err_recs:
        errors_nonzero.append((rel, "episode records with error", len(err_recs)))
    for k in ["episodes", "successes", "success_rate", "average_steps", "average_policy_calls",
              "average_reuses", "average_episode_ms", "policy_query_latency_ms",
              "inference_cycle_latency_ms", "per_task", "depth_calibration", "fusion"]:
        field_presence[bb][k] += (k in s and s[k] is not None)
    n = len(eps)
    succ = sum(1 for r in eps if r.get("success"))
    cmp(rel, "episodes", s.get("episodes"), n, 0)
    cmp(rel, "successes", s.get("successes"), succ, 0)
    cmp(rel, "success_rate", s.get("success_rate"), succ / n if n else None, TOL_RATE)
    cmp(rel, "average_steps", s.get("average_steps"), mean([r["steps_executed"] for r in eps]), TOL_MEAN)
    cmp(rel, "average_policy_calls", s.get("average_policy_calls"), mean([r["policy_calls"] for r in eps]), TOL_MEAN)
    cmp(rel, "average_reuses", s.get("average_reuses"), mean([r.get("reuses", 0) for r in eps]), TOL_MEAN)
    cmp(rel, "average_episode_ms", s.get("average_episode_ms"), mean([r["episode_elapsed_ms"] for r in eps]), TOL_MEAN)
    for sk, ek in [("policy_query_latency_ms", "query_latency_ms"), ("inference_cycle_latency_ms", "inference_cycle_ms")]:
        if s.get(sk) and eps and eps[0].get(ek):
            ep_meds = [r[ek]["median_ms"] for r in eps if r.get(ek)]
            got = s[sk].get("median_ms")
            exp = med(ep_meds)
            cmp(rel, sk + ".median_ms(median of episode medians)", got, exp, TOL_LAT)
            if got is not None and exp is not None:
                latency_median_maxdev[bb] = max(latency_median_maxdev[bb], abs(got - exp))
            ep_means = [r[ek]["mean_ms"] for r in eps if r.get(ek)]
            calls = [r["policy_calls"] for r in eps if r.get(ek)]
            gm = s[sk].get("mean_ms")
            if gm is not None:
                plain = mean(ep_means)
                wtd = sum(m * c for m, c in zip(ep_means, calls)) / max(1, sum(calls))
                latency_mean_check[bb].append((abs(gm - plain), abs(gm - wtd)))
    pt = s.get("per_task") or {}
    by_task = defaultdict(list)
    for r in eps:
        by_task[r["task"]].append(r)
    for t, agg in pt.items():
        recs = by_task.get(t, [])
        if not recs:
            mismatches.append((rel, f"per_task[{t}]", "present in summary", "no episodes with that task", None))
            continue
        tn = len(recs); ts = sum(1 for r in recs if r.get("success"))
        cmp(rel, f"per_task[{t}].episodes", agg.get("episodes"), tn, 0)
        cmp(rel, f"per_task[{t}].successes", agg.get("successes"), ts, 0)
        cmp(rel, f"per_task[{t}].success_rate", agg.get("success_rate"), ts / tn, TOL_RATE)
        cmp(rel, f"per_task[{t}].average_steps", agg.get("average_steps"), mean([r["steps_executed"] for r in recs]), TOL_MEAN)
        cmp(rel, f"per_task[{t}].average_policy_calls", agg.get("average_policy_calls"), mean([r["policy_calls"] for r in recs]), TOL_MEAN)
        cmp(rel, f"per_task[{t}].average_episode_ms", agg.get("average_episode_ms"), mean([r["episode_elapsed_ms"] for r in recs]), TOL_MEAN)
    for t in by_task:
        if pt and t not in pt:
            mismatches.append((rel, f"per_task[{t}]", "missing in summary", "present in episodes", None))

print(f"folders checked: {len(folders)}")
print(f"mismatches beyond tolerance: {len(mismatches)}")
for m in mismatches:
    print("  MISMATCH", m)
print(f"status != completed: {status_bad}")
print(f"non-empty episode_errors / error records: {errors_nonzero}")
print("field presence in summary.json per backbone (count of folders having field):")
for bb, c in field_presence.items():
    print(f"  {bb} (n={per_backbone[bb]}):", {k: v for k, v in c.items() if v != per_backbone[bb]} or "all fields present in all folders")
print("max |summary median - median(episode medians)| per backbone (ms):",
      {k: round(v, 4) for k, v in latency_median_maxdev.items()})
print("summary mean_ms vs mean of episode means / calls-weighted mean, max abs dev per backbone:")
for bb, lst in latency_mean_check.items():
    print(f"  {bb}: plain={max(a for a, b in lst):.4f} weighted={max(b for a, b in lst):.4f}")

# ---------------------------------------------------------------- README claims: seeds, identical episode sets
print("\n# README claims: seed = 42 + episode_index; identical episode sets across configs")
seed_bad = []
for rel, e in folders.items():
    for r in e["episodes"]:
        if "seed" in r and r["seed"] != 42 + r["episode_index"]:
            seed_bad.append((rel, r["episode_index"], r["seed"]))
print(f"seed != 42+episode_index: {len(seed_bad)} {seed_bad[:5]}")
sets = defaultdict(dict)
for rel, e in folders.items():
    key = frozenset((r["task"], r["episode_index"]) for r in e["episodes"])
    sets[e["backbone"]].setdefault(e["config"], set()).update(key)
for bb, cfgs in sets.items():
    ref = cfgs["original"]
    diff = {c: (len(s), len(s ^ ref)) for c, s in cfgs.items() if s != ref}
    print(f"  {bb}: original has {len(ref)} episodes; configs differing: {diff or 'none'}")

# ---------------------------------------------------------------- hardware / implementation heterogeneity
print("\n# Heterogeneous implementation/GPU inside a run folder (fields present only in SmolVLA records)")
for rel, e in folders.items():
    c = Counter((r.get("implementation"), r.get("gpu")) for r in e["episodes"])
    if len(c) > 1 or any(k != (None, None) for k in c):
        if len(c) > 1:
            print(f"  {rel}: {dict(c)}")

# ---------------------------------------------------------------- Lens A.2: CSV vs union of folders
print("\n# LENS A.2  per-backbone CSV rows vs union of the configuration's folders")
CSV_MAP = {
    "simpler_widowx/cogact_base_simpler_widowx.csv": ("cogact_base_simplerenv_bridge", None),
    "simpler_widowx/cronusvla_simpler_widowx.csv": ("cronusvla_simplerenv_bridge", None),
    "simpler_widowx/minivla_simpler_widowx.csv": ("minivla_simplerenv", None),
    "simpler_widowx/openvla_simplerenv_widowx.csv": ("openvla_simplerenv", None),
    "simpler_widowx/spatialvla_simpler_widowx.csv": ("spatialvla_simplerenv_bridge", None),
    "simpler_widowx/univla_simpler_widowx.csv": ("univla_simplerenv_bridge", None),
    "google_robot_fractal/cogact_base_simplerenv_fractal.csv": ("cogact_base_simplerenv_fractal", None),
    "google_robot_fractal/cronusvla_simplerenv_fractal.csv": ("cronusvla_simplerenv_fractal", None),
    "google_robot_fractal/openvla_simplerenv_fractal.csv": ("openvla_simplerenv_fractal", None),
    "google_robot_fractal/spatialvla_simplerenv_fractal.csv": ("spatialvla_simplerenv_fractal", None),
}
for bb in ["openvla", "smolvla", "univla"]:
    for suite in ["10", "goal", "object", "spatial"]:
        CSV_MAP[f"libero/{bb}_libero_{suite}.csv"] = (f"{bb}_libero", f"libero_{suite}")

def fnum(x):
    try:
        return float(x)
    except Exception:
        return None

def union_folders(backbone, config, suite):
    out = []
    for rel, e in folders.items():
        if e["backbone"] == backbone and e["config"] == config:
            if suite is None or e["leaf"] == suite:
                out.append(e)
    return out

csv_rows = {}
csv_mism = []
csv_notes = []
latency_method = Counter()
p95_method = Counter()
reusable_method = Counter()
rows_checked = 0
for relcsv, (backbone, suite) in CSV_MAP.items():
    path = os.path.join(SUMM, relcsv)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    csv_rows[relcsv] = rows
    for row in rows:
        cfg = row["configuration"]
        fl = union_folders(backbone, cfg, suite)
        tag = f"{relcsv}:{cfg}"
        if not fl:
            csv_mism.append((tag, "no folder found", None, None))
            continue
        rows_checked += 1
        eps = [r for e in fl for r in e["episodes"]]
        n = len(eps); succ = sum(1 for r in eps if r.get("success"))
        def c2(field, got, exp, tol):
            if got is None or exp is None:
                return
            if abs(got - exp) > tol:
                csv_mism.append((tag, field, got, exp))
        c2("episodes", fnum(row["episodes"]), n, 0)
        c2("successes", fnum(row["successes"]), succ, 0)
        c2("success_rate_pct", fnum(row["success_rate_pct"]), 100 * succ / n, TOL_RATE)
        c2("avg_steps", fnum(row["avg_steps"]), mean([r["steps_executed"] for r in eps]), TOL_MEAN)
        c2("avg_policy_calls", fnum(row["avg_policy_calls"]), mean([r["policy_calls"] for r in eps]), TOL_MEAN)
        c2("avg_reuses", fnum(row["avg_reuses"]), mean([r.get("reuses", 0) for r in eps]), TOL_MEAN)
        c2("avg_episode_time_s", fnum(row["avg_episode_time_s"]), mean([r["episode_elapsed_ms"] for r in eps]) / 1000, TOL_MEAN)
        # latency medians
        for col, sk, ek in [("policy_median_latency_ms", "policy_query_latency_ms", "query_latency_ms"),
                            ("cycle_median_latency_ms", "inference_cycle_latency_ms", "inference_cycle_ms")]:
            got = fnum(row[col])
            if got is None:
                continue
            ep_meds = [r[ek]["median_ms"] for r in eps if r.get(ek)]
            folder_meds = [e["summary"][sk]["median_ms"] for e in fl if e["summary"].get(sk)]
            cands = {}
            if ep_meds:
                cands["median_of_episode_medians"] = med(ep_meds)
            if folder_meds:
                cands["mean_of_folder_medians"] = mean(folder_meds)
                cands["median_of_folder_medians"] = med(folder_meds)
            if not cands:
                # CronusVLA: no latency recorded. Test elapsed/steps hypotheses.
                tot_ms = sum(r["episode_elapsed_ms"] for r in eps)
                tot_steps = sum(r["steps_executed"] for r in eps)
                tot_calls = sum(r["policy_calls"] for r in eps)
                cands["total_elapsed/total_steps"] = tot_ms / tot_steps
                cands["total_elapsed/total_calls"] = tot_ms / tot_calls
                cands["mean(elapsed/steps)"] = mean([r["episode_elapsed_ms"] / r["steps_executed"] for r in eps])
                cands["median(elapsed/steps)"] = med([r["episode_elapsed_ms"] / r["steps_executed"] for r in eps])
                cands["median(elapsed/calls)"] = med([r["episode_elapsed_ms"] / r["policy_calls"] for r in eps])
                ok = [k for k, v in cands.items() if abs(v - got) <= TOL_LAT]
                latency_method[(backbone.split("_")[0], col, "NOT CHECKABLE (no latency in episodes); closest=" + min(cands, key=lambda k: abs(cands[k] - got)))] += 1
                csv_notes.append((tag, col, got, {k: round(v, 3) for k, v in cands.items()}))
                continue
            ok = [k for k, v in cands.items() if abs(v - got) <= TOL_LAT]
            if ok:
                latency_method[(backbone.split("_")[0], col, ok[0])] += 1
            else:
                csv_mism.append((tag, col, got, {k: round(v, 3) for k, v in cands.items()}))
        # p95 (informational)
        for col, sk, ek in [("policy_p95_latency_ms", "policy_query_latency_ms", "query_latency_ms"),
                            ("cycle_p95_latency_ms", "inference_cycle_latency_ms", "inference_cycle_ms")]:
            got = fnum(row[col])
            if got is None:
                continue
            ep_p = [r[ek]["p95_ms"] for r in eps if r.get(ek)]
            folder_p = [e["summary"][sk]["p95_ms"] for e in fl if e["summary"].get(sk)]
            cands = {}
            if ep_p:
                cands["median_of_episode_p95"] = med(ep_p)
                cands["mean_of_episode_p95"] = mean(ep_p)
            if folder_p:
                cands["mean_of_folder_p95"] = mean(folder_p)
                cands["median_of_folder_p95"] = med(folder_p)
            if cands:
                ok = [k for k, v in cands.items() if abs(v - got) <= TOL_LAT]
                p95_method[(backbone.split("_")[0], col, ok[0] if ok else "NONE")] += 1
        # selected_depth_layers
        sel_sets = []
        for e in fl:
            dc = e["summary"].get("depth_calibration") or {}
            sl = dc.get("selected_layers")
            if sl is None:
                ep_sl = {tuple(r.get("selected_depth_layers") or []) for r in e["episodes"]}
                ep_sl.discard(())
                sl = list(ep_sl)[0] if len(ep_sl) == 1 else None
            sel_sets.append(tuple(sorted(sl)) if sl else ())
        csv_sel = row["selected_depth_layers"].strip()
        csv_sel_t = tuple(sorted(int(x) for x in re.split(r"[;,]", csv_sel))) if csv_sel not in ("", "N/A") else ()
        distinct = set(sel_sets)
        if len(distinct) > 1:
            csv_mism.append((tag, "selected_layers differ across task folders", sel_sets, csv_sel))
        elif cfg.startswith("depth_pruning") and (distinct and list(distinct)[0] != csv_sel_t):
            csv_mism.append((tag, "selected_depth_layers", csv_sel, sel_sets[0]))
        # fusion args
        if cfg.startswith("temporal_fusion"):
            ivals = {(e["summary"].get("arguments") or {}).get("fusion_keyframe_interval") for e in fl}
            fracs = {(e["summary"].get("arguments") or {}).get("fusion_max_reuse_fraction") for e in fl}
            gi, gf = fnum(row["fusion_keyframe_interval"]), fnum(row["fusion_max_reuse_fraction"])
            if ivals != {None} and gi not in ivals:
                csv_mism.append((tag, "fusion_keyframe_interval", gi, ivals))
            if fracs != {None} and gf not in fracs:
                csv_mism.append((tag, "fusion_max_reuse_fraction", gf, fracs))
            if ivals == {None}:
                csv_notes.append((tag, "fusion_keyframe_interval not in arguments; CSV says", gi, gf))
            # reusable tokens
            got = fnum(row["median_reusable_visual_tokens"])
            fm = [ (e["summary"].get("fusion") or {}).get("median_reusable_visual_tokens") for e in fl]
            fm = [x for x in fm if x is not None]
            cands = {}
            if fm:
                cands["mean_of_folder_medians"] = mean(fm); cands["median_of_folder_medians"] = med(fm)
            epm = [r["fusion_reusable_tokens_median"] for r in eps if "fusion_reusable_tokens_median" in r]
            if epm:
                cands["median_of_episode_medians"] = med(epm); cands["mean_of_episode_medians"] = mean(epm)
            tfp = [r for r in eps if "temporal_fused_patches" in r]
            if tfp:
                cands["mean(fused_patches/calls)"] = mean([r["temporal_fused_patches"] / r["policy_calls"] for r in tfp])
                cands["median(fused_patches/calls)"] = med([r["temporal_fused_patches"] / r["policy_calls"] for r in tfp])
                cands["sum(fused)/sum(calls)"] = sum(r["temporal_fused_patches"] for r in tfp) / sum(r["policy_calls"] for r in tfp)
            if got is not None and cands:
                ok = [k for k, v in cands.items() if abs(v - got) <= 0.01]
                reusable_method[(backbone.split("_")[0], ok[0] if ok else "NONE")] += 1
                if not ok:
                    csv_mism.append((tag, "median_reusable_visual_tokens", got, {k: round(v, 3) for k, v in cands.items()}))

print(f"CSV rows checked against folders: {rows_checked}")
print(f"CSV mismatches beyond tolerance: {len(csv_mism)}")
for m in csv_mism:
    print("  CSV MISMATCH", m)
print("latency-median derivation that matched the CSV (backbone, column, method): count")
for k, v in sorted(latency_method.items()):
    print("  ", k, v)
print("p95 derivation that matched (informational):")
for k, v in sorted(p95_method.items()):
    print("  ", k, v)
print("median_reusable_visual_tokens derivation that matched:")
for k, v in sorted(reusable_method.items()):
    print("  ", k, v)
print("notes:")
for n_ in csv_notes[:40]:
    print("  ", n_)

# summary.csv rows vs backbone csv rows
print("\n# summary.csv rows vs backbone CSV rows")
for env in ["simpler_widowx", "google_robot_fractal", "libero"]:
    with open(os.path.join(SUMM, env, "summary.csv")) as f:
        srows = list(csv.DictReader(f))
    bad = 0
    for sr in srows:
        found = False
        for relcsv, rows in csv_rows.items():
            if not relcsv.startswith(env + "/"):
                continue
            for r in rows:
                if r["model_name"] == sr["model_name"] and r["configuration"] == sr["configuration"] and \
                   (env != "libero" or True):
                    # libero: match suite via environment label
                    if env == "libero":
                        suite_map = {"Libero Long": "libero_10", "LIBERO-Goal": "libero_goal",
                                     "LIBERO-Object": "libero_object", "LIBERO-Spatial": "libero_spatial"}
                        if r["suite"] != suite_map.get(sr["environment"]):
                            continue
                    found = True
                    diffs = [k for k in sr if k in r and sr[k] != r[k] and k not in ("environment",)]
                    if diffs:
                        bad += 1
                        print(f"  {env}/summary.csv {sr['model_name']}/{sr['configuration']}: differs in {diffs}: "
                              f"{[(k, sr[k], r[k]) for k in diffs]}")
        if not found:
            bad += 1
            print(f"  {env}/summary.csv row not found in backbone CSVs: {sr['model_name']}/{sr['configuration']}/{sr['environment']}")
    print(f"  {env}/summary.csv: {len(srows)} rows, {bad} problems")

# ---------------------------------------------------------------- Lens B.1 depth pruning
print("\n# LENS B.1  depth pruning")
print("rule assumed: protected = indices < floor(0.25*L) and index L-1; |a-b|>=2; budget = number after depth_pruning")
depth_rows = []
depth_viol = []
sim_mismatch = []
nesting = defaultdict(dict)
for rel, e in folders.items():
    cfg = e["config"]
    if not cfg.startswith("depth_pruning"):
        continue
    budget = int(cfg.replace("depth_pruning", ""))
    s = e["summary"]
    L = nlayers(e["backbone"])
    dc = s.get("depth_calibration") or {}
    sel = dc.get("selected_layers")
    src = "summary"
    if sel is None:
        ep_sl = {tuple(r.get("selected_depth_layers") or []) for r in e["episodes"]}
        ep_sl.discard(())
        sel = sorted(list(ep_sl)[0]) if len(ep_sl) == 1 else (None if not ep_sl else sorted(list(ep_sl)))
        src = "episodes"
    infl = dc.get("influence")
    rules = dc.get("rules")
    nesting[(e["backbone"], e["leaf"])][budget] = tuple(sel) if sel else None
    if not sel:
        depth_viol.append((rel, "no selected_layers recorded", sel))
        continue
    sel = sorted(sel)
    lo = math.floor(0.25 * L)
    probs = []
    if len(sel) != budget:
        probs.append(f"count {len(sel)} != budget {budget}")
    if any(x >= L for x in sel):
        probs.append(f"index >= L={L}")
    if any(x < lo for x in sel):
        probs.append(f"inside protected leading quarter (< {lo})")
    if (L - 1) in sel:
        probs.append(f"final layer {L-1} removed")
    if any(b - a < 2 for a, b in zip(sel, sel[1:])):
        probs.append("adjacent removals")
    if infl is not None and len(infl) != L:
        probs.append(f"influence length {len(infl)} != L {L}")
    sim = None
    if infl:
        n = len(infl)
        start = math.floor(0.25 * n)
        cands = [i for i in range(start, n - 1)]
        ranked = sorted(cands, key=lambda i: (infl[i], i))
        chosen = []
        for i in ranked:
            if any(abs(i - p) <= 1 for p in chosen):
                continue
            chosen.append(i)
            if len(chosen) == budget:
                break
        sim = sorted(chosen)
        if sim != sel:
            # try alternative: protect last not applied / ceil
            alt = {}
            for name, st, protect_last in [("floor,last", start, True), ("floor,nolast", start, False),
                                            ("ceil,last", math.ceil(0.25 * n), True), ("half,nolast", n // 2, False)]:
                cands = [i for i in range(st, n - (1 if protect_last else 0))]
                ranked = sorted(cands, key=lambda i: (infl[i], i))
                ch = []
                for i in ranked:
                    if any(abs(i - p) <= 1 for p in ch):
                        continue
                    ch.append(i)
                    if len(ch) == budget:
                        break
                alt[name] = sorted(ch)
            matching = [k for k, v in alt.items() if v == sel]
            sim_mismatch.append((rel, sel, sim, matching))
    depth_rows.append((rel, L, sel, src, "influence" if infl else "no influence", rules, probs))
    if probs:
        depth_viol.append((rel, probs, sel))

# condense: per (backbone, config) distinct selections
print("\nselected layers per backbone/config (distinct across task folders):")
agg = defaultdict(lambda: defaultdict(list))
for rel, L, sel, src, has_infl, rules, probs in depth_rows:
    e = folders[rel]
    agg[(e["backbone"], e["config"])][(tuple(sel), src, has_infl, json.dumps(rules))].append(e["leaf"])
print("| backbone_env | config | L | selected | source | influence | rules | folders |")
print("|---|---|---|---|---|---|---|---|")
for (bb, cfg), d in sorted(agg.items()):
    for (sel, src, has_infl, rules), leaves in d.items():
        print(f"| {bb} | {cfg} | {nlayers(bb)} | {list(sel)} | {src} | {has_infl} | {rules} | {len(leaves)}: {','.join(str(x) for x in leaves)} |")
print(f"\nrule violations: {len(depth_viol)}")
seen = set()
for v in depth_viol:
    print("  VIOLATION", v)
print(f"\nselector simulation (ascending influence, protect floor(0.25L) & last, gap>=2) mismatches: {len(sim_mismatch)} of {sum(1 for r in depth_rows if r[4]=='influence')} folders with influence")
for rel, sel, sim, matching in sim_mismatch:
    print(f"  {rel}: recorded {sel}, simulated {sim}, alt rules matching: {matching}")
print("\nnesting dp1 ⊂ dp2 ⊂ dp4:")
for (bb, leaf), d in sorted(nesting.items()):
    s1, s2, s4 = d.get(1), d.get(2), d.get(4)
    ok = (s1 is None or s2 is None or set(s1) <= set(s2)) and (s2 is None or s4 is None or set(s2) <= set(s4))
    if not ok:
        print(f"  NOT NESTED {bb}/{leaf}: {s1} {s2} {s4}")
print("  (all others nested)")

# depth pruning latency effect (informational)
print("\ndepth pruning effect on policy median latency vs original (median of episode medians, ms):")
for bb in sorted(per_backbone):
    def medlat(cfg):
        eps = [r for e in folders.values() if e["backbone"] == bb and e["config"] == cfg for r in e["episodes"]]
        v = [r["query_latency_ms"]["median_ms"] for r in eps if r.get("query_latency_ms")]
        return round(med(v), 1) if v else None
    print(f"  {bb}: original={medlat('original')} dp1={medlat('depth_pruning1')} dp2={medlat('depth_pruning2')} dp4={medlat('depth_pruning4')}")

# ---------------------------------------------------------------- Lens B.2 temporal fusion
print("\n# LENS B.2  temporal fusion")
print("| backbone_env | config | folder | N | max_reuse_frac | event_thr | collect_rel | calls | keyframes | sum ceil(calls_i/N) | ratio kf/(calls/N) | median_reusable_tokens |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
fusion_zero = []
fusion_rows = []
for rel, e in folders.items():
    cfg = e["config"]
    if not cfg.startswith("temporal_fusion"):
        continue
    s = e["summary"]; a = s.get("arguments") or {}; fu = s.get("fusion") or {}
    N = a.get("fusion_keyframe_interval")
    frac = a.get("fusion_max_reuse_fraction")
    ev = a.get("fusion_event_motion_threshold")
    cr = a.get("fusion_collect_relevance", fu.get("collect_relevance"))
    calls = sum(r["policy_calls"] for r in e["episodes"])
    kf = fu.get("keyframes")
    mrt = fu.get("median_reusable_visual_tokens")
    if mrt is None:
        epm = [r["fusion_reusable_tokens_median"] for r in e["episodes"] if "fusion_reusable_tokens_median" in r]
        mrt = f"ep-median {med(epm)}" if epm else None
        tfp = [r["temporal_fused_patches"] for r in e["episodes"] if "temporal_fused_patches" in r]
        if tfp:
            mrt = f"fused_patches/calls mean {mean([r['temporal_fused_patches']/r['policy_calls'] for r in e['episodes']]):.1f}"
    exp_ceil = sum(math.ceil(r["policy_calls"] / N) for r in e["episodes"]) if N else None
    ratio = (kf / (calls / N)) if (kf is not None and N) else None
    fusion_rows.append((e["backbone"], cfg, e["leaf"], N, frac, ev, cr, calls, kf, exp_ceil, ratio, mrt))
    zero = (isinstance(mrt, (int, float)) and mrt == 0) or (isinstance(mrt, str) and (mrt.endswith(" 0") or mrt.endswith(" 0.0")))
    if zero:
        fusion_zero.append(rel)
for r in fusion_rows:
    bb, cfg, leaf, N, frac, ev, cr, calls, kf, exp_ceil, ratio, mrt = r
    print(f"| {bb} | {cfg} | {leaf} | {N} | {frac} | {ev} | {cr} | {calls} | {kf} | {exp_ceil} | {round(ratio,3) if ratio is not None else None} | {mrt} |")
print(f"\nfolders in fusion configs with zero reusable tokens ({len(fusion_zero)}):")
for z in fusion_zero:
    print("  ", z)

# ---------------------------------------------------------------- Lens B.3 guarded reuse
print("\n# LENS B.3  guarded reuse")
print("| backbone_env | setting | folders | frame_mae | patch_mae | cosine | transl_floor | cap | episodes | sum reuses | sum steps | reuse/step | ep with reuses>cap*calls | ep with reuses>calls*cap/(cap+1) | steps!=calls+reuses |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
gr = defaultdict(list)
for rel, e in folders.items():
    if e["config"].startswith("guarded_reuse"):
        gr[(e["backbone"], e["config"])].append(e)
for (bb, cfg), lst in sorted(gr.items()):
    args = [json.dumps({k: (e["summary"].get("arguments") or {}).get(k) for k in
             ["reuse_max_frame_mae", "reuse_max_local_patch_mae", "reuse_min_action_cosine",
              "reuse_min_translation_norm", "reuse_max_consecutive"]}, sort_keys=True) for e in lst]
    adist = Counter(args)
    a = json.loads(list(adist)[0])
    cap = a["reuse_max_consecutive"]
    eps = [r for e in lst for r in e["episodes"]]
    sr = sum(r.get("reuses", 0) for r in eps); ss = sum(r["steps_executed"] for r in eps)
    v1 = sum(1 for r in eps if cap is not None and r.get("reuses", 0) > cap * r["policy_calls"])
    v2 = sum(1 for r in eps if cap is not None and r.get("reuses", 0) > r["policy_calls"] * cap / (cap + 1))
    v3 = sum(1 for r in eps if r["steps_executed"] != r["policy_calls"] + r.get("reuses", 0))
    note = "" if len(adist) == 1 else f" ARGS DIFFER ACROSS FOLDERS {dict(adist)}"
    print(f"| {bb} | {cfg} | {len(lst)} | {a['reuse_max_frame_mae']} | {a['reuse_max_local_patch_mae']} | {a['reuse_min_action_cosine']} | {a['reuse_min_translation_norm']} | {cap} | {len(eps)} | {sr} | {ss} | {sr/ss:.4f} | {v1} | {v2} | {v3} |{note}")

# ---------------------------------------------------------------- Lens B.4 action repeat
print("\n# LENS B.4  action repeat")
print("| backbone_env | config | k | m (steps/calls in original: max ceil) | episodes | calls==ceil(steps/(m k)) | calls==floor | calls==steps/(mk) exact | steps==calls+reuses | reuses==steps-calls | sum reuses | mean steps/calls |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for bb in sorted(per_backbone):
    orig = [r for e in folders.values() if e["backbone"] == bb and e["config"] == "original" for r in e["episodes"]]
    m = max(math.ceil(r["steps_executed"] / r["policy_calls"]) for r in orig)
    m_min = min(math.ceil(r["steps_executed"] / r["policy_calls"]) for r in orig)
    for cfg in ["action_repeat2", "action_repeat4"]:
        lst = [e for e in folders.values() if e["backbone"] == bb and e["config"] == cfg]
        eps = [r for e in lst for r in e["episodes"]]
        ks = {(e["summary"].get("arguments") or {}).get("action_repeat") for e in lst}
        k = int(cfg.replace("action_repeat", ""))
        c_ceil = sum(1 for r in eps if r["policy_calls"] == math.ceil(r["steps_executed"] / (m * k)))
        c_floor = sum(1 for r in eps if r["policy_calls"] == math.floor(r["steps_executed"] / (m * k)))
        c_exact = sum(1 for r in eps if r["policy_calls"] * m * k == r["steps_executed"])
        c_sum = sum(1 for r in eps if r["steps_executed"] == r["policy_calls"] + r.get("reuses", 0))
        c_re = sum(1 for r in eps if r.get("reuses", 0) == r["steps_executed"] - r["policy_calls"])
        sr = sum(r.get("reuses", 0) for r in eps)
        ratio = mean([r["steps_executed"] / r["policy_calls"] for r in eps])
        print(f"| {bb} | {cfg} | {k} (args {ks}) | {m} (min {m_min}) | {len(eps)} | {c_ceil} | {c_floor} | {c_exact} | {c_sum} | {c_re} | {sr} | {ratio:.3f} |")
# also original relation
print("\noriginal config: steps vs calls relation per backbone")
for bb in sorted(per_backbone):
    orig = [r for e in folders.values() if e["backbone"] == bb and e["config"] == "original" for r in e["episodes"]]
    eq = sum(1 for r in orig if r["steps_executed"] == r["policy_calls"])
    man = {(e["summary"].get("checkpoint_manifest") or {}).get("native_action_chunk_size") for e in folders.values() if e["backbone"] == bb}
    print(f"  {bb}: {eq}/{len(orig)} episodes steps==calls; native_action_chunk_size in manifest: {man}; max ceil(steps/calls)={max(math.ceil(r['steps_executed']/r['policy_calls']) for r in orig)}")

# ---------------------------------------------------------------- Lens B.5 foveation
print("\n# LENS B.5  foveation keep ratio")
fov = defaultdict(Counter)
for rel, e in folders.items():
    a = e["summary"].get("arguments") or {}
    fov[(e["backbone"], e["config"])][a.get("fovea_keep_ratio")] += 1
bad = []
for (bb, cfg), c in sorted(fov.items()):
    if cfg == "fixed_foveation_keep20" and set(c) != {0.2}:
        bad.append((bb, cfg, dict(c)))
    if cfg == "fixed_foveation_keep50" and set(c) != {0.5}:
        bad.append((bb, cfg, dict(c)))
print("keep20/keep50 folders with fovea_keep_ratio != 0.2/0.5:", bad or "none")
print("fovea_keep_ratio recorded in non-foveation configs (default value):",
      {bb: dict(c) for (bb, cfg), c in fov.items() if cfg == "original"})
# arguments per config presence
print("\narguments.condition per config (distinct):")
cond = defaultdict(set)
for rel, e in folders.items():
    cond[e["config"]].add(e["summary"].get("condition"))
for k, v in sorted(cond.items()):
    print(f"  {k}: {v}")

# ---------------------------------------------------------------- reuse never / nearly never
print("\n# reuse rate per backbone across guarded_reuse settings (sum reuses / sum steps)")
for bb in sorted(per_backbone):
    out = []
    for cfg in ["guarded_reuse_strict", "guarded_reuse_moderate", "guarded_reuse_aggressive"]:
        eps = [r for e in folders.values() if e["backbone"] == bb and e["config"] == cfg for r in e["episodes"]]
        sr = sum(r.get("reuses", 0) for r in eps); ss = sum(r["steps_executed"] for r in eps)
        nz = sum(1 for r in eps if r.get("reuses", 0) > 0)
        out.append(f"{cfg.split('_')[-1]}: {sr}/{ss}={sr/ss:.4f} (episodes with any reuse {nz}/{len(eps)})")
    print(f"  {bb}: " + "; ".join(out))
