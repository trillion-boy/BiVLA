#!/usr/bin/env python3
"""Read-only audit of the 25 summarization CSVs."""
import csv, glob, io, os, re, sys
from collections import Counter, defaultdict, OrderedDict

ROOT = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11/summarization"
FILES = sorted(glob.glob(os.path.join(ROOT, "*", "*.csv")))
rel = lambda p: os.path.relpath(p, ROOT)

def load(path):
    raw = open(path, "rb").read()
    crlf = raw.count(b"\r\n"); lf_only = raw.count(b"\n") - crlf; cr_only = raw.count(b"\r") - crlf
    text = raw.decode("utf-8")
    lines = text.split("\r\n") if crlf else text.split("\n")
    if lines and lines[-1] == "": lines = lines[:-1]
    rdr = csv.reader(io.StringIO("\n".join(lines)))
    rows = list(rdr)
    hdr, body = rows[0], rows[1:]
    dicts = [OrderedDict(zip(hdr, r)) for r in body]
    return dict(path=path, raw_lines=lines, hdr=hdr, rows=dicts, crlf=crlf, lf=lf_only, cr=cr_only,
                bom=raw.startswith(b"\xef\xbb\xbf"), trailing=raw.endswith(b"\r\n"))

D = {rel(p): load(p) for p in FILES}
print("FILES OPENED:", len(D))

def num(s):
    try: return float(s)
    except: return None

def fam(cfg):
    for f in ("fixed_foveation","action_repeat","depth_pruning","guarded_reuse","temporal_fusion"):
        if cfg.startswith(f): return f
    return "original" if cfg == "original" else "?"

# ---------------- 1. structure ----------------
print("\n## 1. STRUCTURE")
print("| file | rows | cols | CRLF | LF-only | CR-only | BOM | trailing CRLF | has suite col | N/A cells | empty cells |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
allcols = set()
for k, d in D.items():
    na = sum(v == "N/A" for r in d["rows"] for v in r.values())
    emp = sum(v == "" for r in d["rows"] for v in r.values())
    allcols |= set(d["hdr"])
    print(f"| {k} | {len(d['rows'])} | {len(d['hdr'])} | {d['crlf']} | {d['lf']} | {d['cr']} | {d['bom']} | {d['trailing']} | {'suite' in d['hdr']} | {na} | {emp} |")
base = D["simpler_widowx/summary.csv"]["hdr"]
print("\nBase header (23 cols):", ",".join(base))
for k, d in D.items():
    extra = [c for c in d["hdr"] if c not in base]; missing = [c for c in base if c not in d["hdr"]]
    if extra or missing: print(f"  {k}: extra={extra} missing={missing} position_of_suite={d['hdr'].index('suite') if 'suite' in d['hdr'] else None}")
# header ordering check
for k, d in D.items():
    h = [c for c in d["hdr"] if c != "suite"]
    if h != base: print("  HEADER ORDER DIFFERS:", k)

# numeric representation per column
print("\n### Numeric representation per column (int-like vs float-like vs N/A vs other), aggregated over all files")
numcols = [c for c in base if c not in ("model_name","environment","checkpoint_model_id","configuration","selected_depth_layers")]
for c in numcols:
    cnt = Counter()
    for k, d in D.items():
        for r in d["rows"]:
            v = r[c]
            if v == "N/A": cnt["N/A"] += 1
            elif re.fullmatch(r"-?\d+", v): cnt["int"] += 1
            elif re.fullmatch(r"-?\d+\.\d+", v): cnt["float"] += 1
            else: cnt["other:"+v] += 1
    print(f"  {c}: {dict(cnt)}")
print("\n### Per-file representation of always-integer-valued columns (success_rate_pct / avg_reuses / median_reusable_visual_tokens / avg_steps)")
for c in ("success_rate_pct","avg_reuses","median_reusable_visual_tokens","avg_steps","avg_policy_calls","fusion_max_reuse_fraction"):
    per = {}
    for k, d in D.items():
        cnt = Counter()
        for r in d["rows"]:
            v = r[c]
            cnt["N/A" if v=="N/A" else "int" if re.fullmatch(r"-?\d+", v) else "float"] += 1
        per[k] = dict(cnt)
    print(f"  {c}:")
    for k,v in per.items(): print(f"     {k}: {v}")
print("\n### selected_depth_layers separators")
for k, d in D.items():
    seps = Counter()
    for r in d["rows"]:
        v = r["selected_depth_layers"]
        if v == "N/A": continue
        if ";" in v: seps[";"] += 1
        elif "," in v: seps[","] += 1
        else: seps["single"] += 1
    print(f"  {k}: {dict(seps)}  values={[r['selected_depth_layers'] for r in d['rows'] if r['selected_depth_layers']!='N/A']}")
print("\n### Quoted fields in raw lines")
for k, d in D.items():
    q = [l for l in d["raw_lines"] if '"' in l]
    if q: print(f"  {k}: {len(q)} quoted lines, e.g. {q[0][:160]}")

# ---------------- 2. column semantics ----------------
print("\n## 2. COLUMN SEMANTICS")
print("\n### 2a. cycle_median vs policy_median per backbone/env (non-action-repeat rows): ratio and difference")
print("| file | rows | cycle-policy diff min..max (ms) | cycle/policy min..max | equal? |")
print("|---|---|---|---|---|")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    rs = [r for r in d["rows"] if not r["configuration"].startswith("action_repeat")]
    diffs = [num(r["cycle_median_latency_ms"])-num(r["policy_median_latency_ms"]) for r in rs]
    rat = [num(r["cycle_median_latency_ms"])/num(r["policy_median_latency_ms"]) for r in rs]
    eq = sum(abs(x)<1e-9 for x in diffs)
    print(f"| {k} | {len(rs)} | {min(diffs):.3f}..{max(diffs):.3f} | {min(rat):.3f}..{max(rat):.3f} | {eq}/{len(rs)} exactly equal |")
print("\n### 2b. action_repeat rows: policy vs cycle latency; orig cycle; cycle/policy; steps, calls, reuses, steps-calls; freq candidates")
print("| file | cfg | policy_med | cycle_med | orig_cycle | cycle/policy | steps | calls | reuses | steps-calls | hz | 1000/cycle | steps/t | calls/t |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    orig = [r for r in d["rows"] if r["configuration"]=="original"][0]
    for r in d["rows"]:
        if not r["configuration"].startswith("action_repeat"): continue
        s, c, t = num(r["avg_steps"]), num(r["avg_policy_calls"]), num(r["avg_episode_time_s"])
        pm, cm = num(r["policy_median_latency_ms"]), num(r["cycle_median_latency_ms"])
        suite = r.get("suite","")
        print(f"| {k}{'/'+suite if suite else ''} | {r['configuration']} | {pm} | {cm} | {orig['cycle_median_latency_ms']} | {cm/pm:.3f} | {s} | {c} | {r['avg_reuses']} | {s-c:.3f} | {r['control_frequency_hz']} | {1000/cm:.3f} | {s/t:.3f} | {c/t:.3f} |")

print("\n### 2c. control_frequency_hz: which definition fits, per file (all rows). Reporting max abs relative error of each candidate")
print("| file | 1000/cycle_med maxrelerr | steps/t maxrelerr | calls/t maxrelerr | 1000/policy_med maxrelerr | best |")
print("|---|---|---|---|---|---|")
for k, d in D.items():
    errs = defaultdict(list)
    for r in d["rows"]:
        hz = num(r["control_frequency_hz"]); s,c,t = num(r["avg_steps"]), num(r["avg_policy_calls"]), num(r["avg_episode_time_s"])
        cm, pm = num(r["cycle_median_latency_ms"]), num(r["policy_median_latency_ms"])
        errs["1000/cycle"].append(abs(1000/cm-hz)/hz)
        errs["steps/t"].append(abs(s/t-hz)/hz)
        errs["calls/t"].append(abs(c/t-hz)/hz)
        errs["1000/policy"].append(abs(1000/pm-hz)/hz)
    mx = {kk: max(v) for kk,v in errs.items()}
    best = min(mx, key=mx.get)
    print(f"| {k} | {mx['1000/cycle']:.4f} | {mx['steps/t']:.4f} | {mx['calls/t']:.4f} | {mx['1000/policy']:.4f} | {best} |")
print("\n#### control_frequency_hz per-row detail where best-fit (steps/t) rel err > 2%")
for k, d in D.items():
    for r in d["rows"]:
        hz = num(r["control_frequency_hz"]); s,c,t = num(r["avg_steps"]), num(r["avg_policy_calls"]), num(r["avg_episode_time_s"])
        cm = num(r["cycle_median_latency_ms"])
        e1, e2, e3 = abs(1000/cm-hz)/hz, abs(s/t-hz)/hz, abs(c/t-hz)/hz
        if min(e1,e2,e3) > 0.02:
            print(f"  {k} {r.get('suite','')} {r['configuration']}: hz={hz} 1000/cycle={1000/cm:.3f} steps/t={s/t:.3f} calls/t={c/t:.3f}")

print("\n### 2d. avg_reuses accounting")
print("| file | cfg family | rows | reuses==0 | reuses==steps-calls (|d|<0.01) | reuses < steps-calls (guarded) | other |")
print("|---|---|---|---|---|---|---|")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    agg = defaultdict(Counter)
    for r in d["rows"]:
        f = fam(r["configuration"]); s,c,ru = num(r["avg_steps"]), num(r["avg_policy_calls"]), num(r["avg_reuses"])
        if abs(ru) < 1e-9 and abs(s-c) < 1e-9: agg[f]["zero&steps==calls"] += 1
        elif abs(ru) < 1e-9: agg[f]["zero_but_steps!=calls"] += 1
        elif abs(ru-(s-c)) < 0.011: agg[f]["==steps-calls"] += 1
        else: agg[f][f"other(ru={ru},s-c={s-c:.3f})"] += 1
    for f, cnt in agg.items():
        print(f"| {k} | {f} | {sum(cnt.values())} | {dict(cnt)} |")

print("\n### 2e. action_repeat / depth_layers columns in non-repeat / non-depth rows")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    ar = Counter(r["action_repeat"] for r in d["rows"] if not r["configuration"].startswith("action_repeat"))
    ar2 = {r["configuration"]: r["action_repeat"] for r in d["rows"] if r["configuration"].startswith("action_repeat")}
    dl = Counter(r["depth_layers"] for r in d["rows"] if not r["configuration"].startswith("depth_pruning"))
    dl2 = {r["configuration"]: (r["depth_layers"], r["selected_depth_layers"]) for r in d["rows"] if r["configuration"].startswith("depth_pruning")}
    print(f"  {k}: action_repeat in non-AR rows={dict(ar)}; AR rows={ar2}; depth_layers in non-DP rows={dict(dl)}; DP rows={dl2}")

print("\n### 2f. fusion columns in non-fusion rows / fusion rows")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    nf = Counter((r["fusion_keyframe_interval"], r["fusion_max_reuse_fraction"], r["median_reusable_visual_tokens"]) for r in d["rows"] if not r["configuration"].startswith("temporal_fusion"))
    fu = {r["configuration"].replace("temporal_fusion_",""): (r["fusion_keyframe_interval"], r["fusion_max_reuse_fraction"], r["median_reusable_visual_tokens"]) for r in d["rows"] if r["configuration"].startswith("temporal_fusion")}
    print(f"  {k}: non-fusion rows (kf,frac,tokens)={dict(nf)}; fusion rows={fu}")

# ---------------- 3. summary vs per-file ----------------
print("\n## 3. SUMMARY vs PER-FILE")
SUITE_MAP = {"libero_10":"Libero Long","libero_goal":"LIBERO-Goal","libero_object":"LIBERO-Object","libero_spatial":"LIBERO-Spatial"}
def strip_suite(line, hdr):
    # returns line with suite column removed and environment replaced -> compare
    vals = next(csv.reader([line]))
    return vals

def check_summary(summary_key, per_keys, is_libero):
    S = D[summary_key]
    per_rows = []  # (file, rowdict, rawline)
    for pk in per_keys:
        d = D[pk]
        for r, raw in zip(d["rows"], d["raw_lines"][1:]):
            per_rows.append((pk, r, raw))
    # build lookup by (model_name, suite/env, configuration)
    def key_of(r, is_summary):
        if is_libero:
            env = r["environment"] if is_summary else SUITE_MAP[r["suite"]]
        else:
            env = r["environment"]
        return (r["model_name"], env, r["configuration"])
    lookup = {key_of(r, False): (pk, r, raw) for pk, r, raw in per_rows}
    print(f"\n### {summary_key}: {len(S['rows'])} rows")
    n_byte_equal = 0; n_field_equal = 0; n_missing = 0
    for r, raw in zip(S["rows"], S["raw_lines"][1:]):
        kk = key_of(r, True)
        if kk not in lookup:
            n_missing += 1; print("  NOT FOUND in per-file:", kk); continue
        pk, pr, praw = lookup[kk]
        if raw == praw: n_byte_equal += 1
        # field compare excluding suite and environment
        fe = all(r[c] == pr[c] for c in S["hdr"] if c != "environment")
        env_same = r["environment"] == pr["environment"]
        if fe: n_field_equal += 1
        else:
            diffs = {c:(r[c],pr[c]) for c in S["hdr"] if c!="environment" and r[c]!=pr[c]}
            print(f"  FIELD DIFF {kk}: {diffs}")
        if not is_libero and raw != praw:
            print(f"  BYTE DIFF {kk} (file {pk})")
    print(f"  byte-equal raw lines: {n_byte_equal}/{len(S['rows'])}; field-equal (all cols except environment): {n_field_equal}/{len(S['rows'])}; not found: {n_missing}")
    # selection rule
    print("\n  Selection per (model, env/suite, family): summary pick vs rule 'max success, then min avg_steps' vs alt rules")
    print("  | model | env | family | picked | success | steps | rule(max succ,min steps) | rule(max succ,min policy_lat) | rule(max succ,min cycle_lat) | rule(max succ, first in file) | rule(max succ, min ep_time) | candidates (cfg:succ/steps/policy_lat/cycle_lat/t) |")
    print("  |---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for pk, r, raw in per_rows:
        env = SUITE_MAP[r["suite"]] if is_libero else r["environment"]
        groups[(r["model_name"], env, fam(r["configuration"]))].append(r)
    picks = {}
    for r in S["rows"]:
        picks[(r["model_name"], r["environment"], fam(r["configuration"]))] = r
    dev = []
    for g, rs in groups.items():
        if g[2] == "original": continue
        p = picks.get(g)
        if p is None: print("  NO PICK for", g); continue
        rule1 = min(rs, key=lambda r: (-num(r["success_rate_pct"]), num(r["avg_steps"])))["configuration"]
        rule2 = min(rs, key=lambda r: (-num(r["success_rate_pct"]), num(r["policy_median_latency_ms"])))["configuration"]
        rule3 = min(rs, key=lambda r: (-num(r["success_rate_pct"]), num(r["cycle_median_latency_ms"])))["configuration"]
        rule4 = max(rs, key=lambda r: num(r["success_rate_pct"]))["configuration"]  # first in file among ties
        rule5 = min(rs, key=lambda r: (-num(r["success_rate_pct"]), num(r["avg_episode_time_s"])))["configuration"]
        cands = "; ".join(f"{r['configuration'].split('_')[-1] if fam(r['configuration'])!='temporal_fusion' else r['configuration'].replace('temporal_fusion_','')}:{r['success_rate_pct']}/{r['avg_steps']}/{r['policy_median_latency_ms']}/{r['cycle_median_latency_ms']}/{r['avg_episode_time_s']}" for r in rs)
        flag = "" if p["configuration"]==rule1 else " **DEV**"
        print(f"  | {g[0]} | {g[1]} | {g[2]} | {p['configuration']}{flag} | {p['success_rate_pct']} | {p['avg_steps']} | {rule1} | {rule2} | {rule3} | {rule4} | {rule5} | {cands} |")
        if p["configuration"] != rule1: dev.append((g, p["configuration"], rule1, rule2, rule3, rule4, rule5))
    print(f"\n  Deviations from 'max success then min avg_steps': {len(dev)}")
    for x in dev: print("   ", x)
    # also: does summary include every family per group and only one per family?
    cnt = Counter((r["model_name"], r["environment"], fam(r["configuration"])) for r in S["rows"])
    multi = {k:v for k,v in cnt.items() if v>1}
    print("  families with >1 row in summary:", multi)
    fams_present = defaultdict(set)
    for r in S["rows"]: fams_present[(r["model_name"], r["environment"])].add(fam(r["configuration"]))
    for k2,v in fams_present.items():
        if len(v)!=6: print("  incomplete family set:", k2, sorted(v))

check_summary("simpler_widowx/summary.csv", [k for k in D if k.startswith("simpler_widowx/") and not k.endswith("summary.csv")], False)
check_summary("google_robot_fractal/summary.csv", [k for k in D if k.startswith("google_robot_fractal/") and not k.endswith("summary.csv")], False)
check_summary("libero/summary.csv", [k for k in D if k.startswith("libero/") and not k.endswith("summary.csv")], True)

# ---------------- 4. naming ----------------
print("\n## 4. NAMING")
for col in ("model_name","environment","checkpoint_model_id","suite"):
    vals = defaultdict(set)
    for k, d in D.items():
        for r in d["rows"]:
            if col in r: vals[r[col]].add(k)
    print(f"\n### distinct {col} ({len(vals)})")
    for v, fs in vals.items(): print(f"  {v!r}: {sorted(fs) if len(fs)<=3 else str(len(fs))+' files: '+', '.join(sorted(fs)[:2])+' ...'}")
print("\n### summary row order (model_name order of first appearance) per summary file")
for k in ("simpler_widowx/summary.csv","google_robot_fractal/summary.csv","libero/summary.csv"):
    order = []
    for r in D[k]["rows"]:
        key = (r["model_name"], r["environment"])
        if key not in order: order.append(key)
    print(f"  {k}: {order}")
print("### per-file configuration order")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    print(f"  {k}: {[r['configuration'] for r in d['rows']]}")

# ---------------- 5. latency per step ----------------
print("\n## 5. LATENCY PER STEP = 1000*avg_episode_time_s/avg_steps")
print("| file | suite | cfg | success | steps | calls | calls/steps | t (s) | lat/step ms | cycle_med | policy_med | t/calls ms |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
LPS = {}
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    for r in d["rows"]:
        s,c,t = num(r["avg_steps"]), num(r["avg_policy_calls"]), num(r["avg_episode_time_s"])
        lps = 1000*t/s; LPS[(k, r.get("suite",""), r["configuration"])] = lps
        print(f"| {k} | {r.get('suite','')} | {r['configuration']} | {r['success_rate_pct']} | {s} | {c} | {c/s:.3f} | {t} | {lps:.2f} | {r['cycle_median_latency_ms']} | {r['policy_median_latency_ms']} | {1000*t/c:.2f} |")
print("\n### per backbone/env: latency_per_step change vs original, per family (min..max over variants; suites listed separately for LIBERO)")
print("| file | suite | orig lat/step ms | family | variants | lat/step min..max ms | delta min..max ms | pct min..max |")
print("|---|---|---|---|---|---|---|---|")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    suite = d["rows"][0].get("suite","")
    o = LPS[(k, suite, "original")]
    byfam = defaultdict(list)
    for r in d["rows"]:
        if r["configuration"]=="original": continue
        byfam[fam(r["configuration"])].append((r["configuration"], LPS[(k, suite, r["configuration"])]))
    for f, lst in byfam.items():
        v = [x[1] for x in lst]
        print(f"| {k} | {suite} | {o:.2f} | {f} | {len(lst)} | {min(v):.2f}..{max(v):.2f} | {min(v)-o:+.2f}..{max(v)-o:+.2f} | {100*(min(v)-o)/o:+.1f}%..{100*(max(v)-o)/o:+.1f}% |")
print("\n### Rows where avg_policy_calls != avg_steps (ratio calls/steps)")
print("| file | suite | cfg | steps | calls | calls/steps | steps/calls | reuses |")
print("|---|---|---|---|---|---|---|---|")
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    for r in d["rows"]:
        s,c = num(r["avg_steps"]), num(r["avg_policy_calls"])
        if abs(s-c) > 1e-9:
            print(f"| {k} | {r.get('suite','')} | {r['configuration']} | {s} | {c} | {c/s:.4f} | {s/c:.3f} | {r['avg_reuses']} |")

# ---------------- 6. sanity ----------------
print("\n## 6. SANITY")
print("### success_rate_pct vs successes/episodes*100")
bad = 0
for k, d in D.items():
    for r in d["rows"]:
        exp = num(r["successes"])/num(r["episodes"])*100
        if abs(exp-num(r["success_rate_pct"]))>1e-6:
            bad += 1; print(f"  MISMATCH {k} {r.get('suite','')} {r['configuration']}: {r['success_rate_pct']} vs {exp}")
print(f"  mismatches: {bad}")
print("### tasks/episodes per environment/file")
for k, d in D.items():
    print(f"  {k}: tasks={set(r['tasks'] for r in d['rows'])} episodes={set(r['episodes'] for r in d['rows'])} episodes/task={set(str(num(r['episodes'])/num(r['tasks'])) for r in d['rows'])}")
print("### rows with avg_policy_calls > avg_steps")
for k, d in D.items():
    for r in d["rows"]:
        if num(r["avg_policy_calls"]) > num(r["avg_steps"]): print(f"  {k} {r['configuration']}: calls={r['avg_policy_calls']} steps={r['avg_steps']}")
print("  (end)")
print("### rows with avg_reuses > avg_steps - avg_policy_calls (+0.011)")
for k, d in D.items():
    for r in d["rows"]:
        if num(r["avg_reuses"]) > num(r["avg_steps"])-num(r["avg_policy_calls"])+0.011: print(f"  {k} {r['configuration']}: reuses={r['avg_reuses']} steps-calls={num(r['avg_steps'])-num(r['avg_policy_calls']):.3f}")
print("  (end)")
print("### duplicated configuration within a file (key = model_name, environment/suite, configuration)")
for k, d in D.items():
    cnt = Counter((r["model_name"], r.get("suite", r["environment"]), r["configuration"]) for r in d["rows"])
    dup = {kk:v for kk,v in cnt.items() if v>1}
    if dup: print(f"  {k}: {dup}")
    # duplicated full rows
    cnt2 = Counter(tuple(r.values()) for r in d["rows"])
    dup2 = [kk for kk,v in cnt2.items() if v>1]
    if dup2: print(f"  {k}: {len(dup2)} fully duplicated rows")
print("  (end)")
print("### configuration set per file vs canonical 14")
canon = ["original","fixed_foveation_keep20","fixed_foveation_keep50","action_repeat2","action_repeat4","depth_pruning1","depth_pruning2","depth_pruning4","guarded_reuse_strict","guarded_reuse_moderate","guarded_reuse_aggressive","temporal_fusion_motion_entropy","temporal_fusion_task_aware","temporal_fusion_conservative_adaptive"]
for k, d in D.items():
    if k.endswith("summary.csv"): continue
    have = [r["configuration"] for r in d["rows"]]
    miss = [c for c in canon if c not in have]; extra = [c for c in have if c not in canon]
    if miss or extra or len(have)!=14: print(f"  {k}: n={len(have)} missing={miss} extra={extra}")
print("  (end)")
print("### p95 < median anywhere?")
for k, d in D.items():
    for r in d["rows"]:
        if num(r["policy_p95_latency_ms"]) < num(r["policy_median_latency_ms"]): print(f"  {k} {r['configuration']}: policy p95 {r['policy_p95_latency_ms']} < med {r['policy_median_latency_ms']}")
        if num(r["cycle_p95_latency_ms"]) < num(r["cycle_median_latency_ms"]): print(f"  {k} {r.get('suite','')} {r['configuration']}: cycle p95 {r['cycle_p95_latency_ms']} < med {r['cycle_median_latency_ms']}")
print("  (end)")
print("### cycle_median < policy_median anywhere?")
for k, d in D.items():
    for r in d["rows"]:
        if num(r["cycle_median_latency_ms"]) < num(r["policy_median_latency_ms"]): print(f"  {k} {r.get('suite','')} {r['configuration']}: cycle {r['cycle_median_latency_ms']} < policy {r['policy_median_latency_ms']} (ratio {num(r['cycle_median_latency_ms'])/num(r['policy_median_latency_ms']):.3f})")
print("  (end)")
print("### avg_steps == max horizon (all episodes timed out)?")
for k, d in D.items():
    for r in d["rows"]:
        if re.fullmatch(r"\d+", r["avg_steps"]) or r["successes"]=="0": print(f"  {k} {r.get('suite','')} {r['configuration']}: steps={r['avg_steps']} successes={r['successes']} calls={r['avg_policy_calls']}")
print("  (end)")
