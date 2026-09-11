import json, os, math
from collections import Counter, defaultdict
RES = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11/results"
F = {}
for dp, dn, fn in os.walk(RES):
    dn.sort()
    if "summary.json" in fn:
        rel = os.path.relpath(dp, RES); p = rel.split("/")
        F[rel] = dict(bb=p[0], cfg=p[1], s=json.load(open(dp + "/summary.json")),
                      eps=[json.loads(l) for l in open(dp + "/episodes.jsonl") if l.strip()])
def sig(bb, cfg):
    return sorted((r["task"], r["episode_index"], r["success"], r["steps_executed"], r["policy_calls"]) for e in F.values() if e["bb"] == bb and e["cfg"] == cfg for r in e["eps"])
print("# (a) conservative_adaptive vs original, episode-level (task, idx, success, steps, calls)")
for bb in sorted({e["bb"] for e in F.values()}):
    a, b = sig(bb, "original"), sig(bb, "temporal_fusion_conservative_adaptive")
    if bb == "openvla_libero":
        a = [x for x in a if not x[0].startswith("libero_10")]
    diff = sum(1 for x, y in zip(a, b) if x != y)
    print(f"  {bb}: identical={a == b} (episodes differing: {diff}/{len(b)})")
print("\n# (b) depth: would greedy (floor(0.25L) protected, gap>=2) have removed the last layer without protect_last?")
seen = set()
for rel, e in F.items():
    dc = e["s"].get("depth_calibration") or {}
    infl, sel = dc.get("influence"), dc.get("selected_layers")
    if not infl or not sel: continue
    k = (e["bb"], e["cfg"], tuple(sel))
    if k in seen: continue
    seen.add(k)
    n = len(infl); st = math.floor(0.25 * n); b = len(sel)
    ranked = sorted(range(st, n), key=lambda i: (infl[i], i)); ch = []
    for i in ranked:
        if any(abs(i - p) <= 1 for p in ch): continue
        ch.append(i)
        if len(ch) == b: break
    if sorted(ch) != sorted(sel):
        print(f"  {e['bb']}/{e['cfg']}: without protect_last -> {sorted(ch)}, recorded {sorted(sel)} (last-layer protection binding)")
print("  (all other folders: last-layer protection not binding)")
print("\n# (c) episode_errors field presence per backbone")
c = defaultdict(Counter)
for rel, e in F.items():
    c[e["bb"]][repr(e["s"].get("episode_errors"))] += 1
for bb, v in sorted(c.items()):
    print(f"  {bb}: {dict(v)}")
print("\n# (d) univla bridge fusion task_aware: OOM folders and where the OOM run sits in time (recorded fields)")
for t in ["widowx_put_eggplant_in_basket", "widowx_stack_cube"]:
    e = F[f"univla_simplerenv_bridge/temporal_fusion_task_aware/{t}"]
    errs = [r["episode_index"] for r in e["eps"] if r.get("error")]
    print(f"  {t}: error episode indices {errs[:12]}... total {len(errs)}; ok indices {[r['episode_index'] for r in e['eps'] if not r.get('error')]}")
