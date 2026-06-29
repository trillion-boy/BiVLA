# Non-Uniform Representation for VLAs — progress & findings

**Project.** *Non-Uniform Visual Representation for Vision-Language-Action Models*
(Junseo + Soumyaratna). Training-free, inference-time control of **frozen** VLAs
(UniVLA/Emu3, SpatialVLA/PaliGemma2) on SimplerEnv / WidowX-Bridge.

---

## 1. The thesis (re-framed)

The north star is **not "foveation" per se** — it is **non-uniform allocation of
compute/representation**: *spend the model's budget where the task needs it, save
it where it doesn't.* Foveation (sharp center, coarse periphery) is just the
**spatial** instance of this idea. There are (at least) three axes:

| Axis | "focus where it matters" means | Our lever | Where it works |
|---|---|---|---|
| **Spatial** | object sharp, background coarse | foveation / visual-token reduction | needs a real ViT (SpatialVLA) |
| **Depth** | hard decisions go deep, easy ones stay shallow | adaptive layer count | UniVLA (FFN-bound decode) |
| **Temporal** | recompute only what changed between frames | feature / KV caching | both (modest on decode-bound models) |

**Hard constraints (the "constitution").** Training-free · no external modules
(no GroundingDINO) · frozen backbone · **reduce latency while maintaining/raising
success.**

The mentor's method-overview flow already encodes the controller idea:
`Instruction+Image → Understand task type → Check phase (grasp/place) → Need extra
focus? → (Apply focus | Normal model) → Predict action → Move`. Our work tests
**which "focus" actually buys latency** on each backbone, and refines the
controller accordingly.

---

## 2. What we have tried (by axis)

### 2.1 Spatial — visual-token reduction

**(a) FastV-style token pruning on UniVLA/Emu3.** Prune low-attention visual
tokens inside the LLM after an early layer (importance = the model's own
attention; no external module). Training-free, input grid untouched (OOD-safe).
- **Result (eggplant, N=8):** baseline 100% / 1305 ms·infer⁻¹; keep 0.7 → **75%** /
  1261 ms (**1.03×**); keep 0.5 → **38%** / 1293 ms (1.01×).
- **Verdict: does not work on UniVLA.** Success collapses while latency barely
  moves. **Why:** Emu3's per-token *decode* is **FFN/MoE-bound**; attention over
  visual tokens is a small slice of the cost, so cutting tokens saves ~3% but
  removes spatial information the policy needs → fails. *(A clean negative result:
  on autoregressive, decode-bound VLAs, spatial token-pruning is the wrong lever.)*

**(b) ToMe (token merging) on SpatialVLA/SigLIP.** Merge similar (background)
tokens *between* SigLIP ViT layers, unmerge to full grid at the end → cuts ViT
compute, identical layout downstream (no OOD). AutoGaze/center can protect the
important region. Training-free, merge-not-drop.
- **Status:** implemented + CPU unit-tested (token count preserved, protected
  tokens bit-exact, redundant tokens collapse). **Not yet benchmarked on-policy.**
- **Why SpatialVLA, not UniVLA:** UniVLA encodes images as a **discrete VQ grid**
  declared in the text prefix — you cannot average discrete IDs and dropping any
  breaks the grid, so token reduction there is structurally OOD. SpatialVLA has a
  real continuous ViT, the natural home for spatial token methods.

**Earlier (foveation-by-blur), pre-this-work.** Blurring the periphery keeps the
**same token count**, so the policy processes the same sequence → **no latency
gain** (matches the mentor's note: *"overhead was a lot, not much gain"*). This is
the core lesson that pushed us to real token reduction / other axes.

### 2.2 Depth — adaptive layer count (UniVLA) ✅ works

Bypass the most redundant decoder layers (redundancy = low input→output cosine
change, calibrated once; cuts attention **and** FFN proportionally — the real
latency lever on FFN-bound decode). The repo's mechanism, fixed for transformers
4.51 (bypassed layers must still write a placeholder KV or DynamicCache crashes).
- **Result (eggplant, N=8):**

  | prune | success | ms·infer⁻¹ | speedup |
  |---|---|---|---|
  | 0 (base) | **100%** | 1305 | 1.00× |
  | **4** | **100%** | 1190 | **1.10×** |
  | 8 | 62% | 983 | 1.33× |
  | 12 | 62% | 852 | 1.53× |

- **Verdict: works.** Pruning 4 layers is a **free 10%** (success unchanged), and
  latency keeps dropping with more pruning. Unlike spatial token pruning, depth
  reduction *actually* moves UniVLA latency.
- **Key diagnostic:** the failures at prune 8/12 are episodes that **never grasp**
  (`ever_grasped_src=false`) — i.e. the damage concentrates at the **precise grasp
  moment**, while free-space transit tolerates aggressive pruning. → motivates a
  **phase-adaptive depth controller** (next).

### 2.3 Temporal — caching (planned)
Reuse static visual tokens'/features' KV across adjacent frames, recompute only
what changed. Orthogonal, stackable on the above. Expected to be modest on UniVLA
(decode-bound) but worth measuring; not yet started.

---

## 3. Lessons that shape the direction

1. **Blur ≠ latency.** Latency only moves if the token/compute count actually
   drops (drop / merge / cache / skip), not if pixels are blurred.
2. **The backbone decides which axis pays off.** UniVLA = discrete VQ + FFN-bound
   decode → *spatial token reduction can't lower latency* (structurally OOD +
   wrong cost center); **depth** is the lever. SpatialVLA = real ViT → the spatial
   axis can pay off (to be measured).
3. **OOD-safety: merge/cache/skip-late > drop.** Dropping tokens or re-indexing
   positions breaks a frozen model (we observed a hard 0% collapse from position
   re-indexing); keeping original positions / merging / pruning only *late* avoids
   it.
4. **Failures are localized, not uniform.** Aggressive depth pruning fails
   specifically at the grasp phase — so a *non-uniform* (phase-aware) policy can
   recover success while keeping most of the speedup. This is the project thesis
   in action.

---

## 4. Current direction: phase-adaptive depth (the foveation principle, in depth)

Calibrate the layer-redundancy ranking once, then let the controller pick **how
many** of the ranked layers to bypass **per phase**:
- precise phases (grasp / fine alignment) → keep depth (prune ~0–2),
- coarse phases (reach / transit) → prune aggressively (~8).

Toggling pre-ranked layers is cheap (module swap, no re-calibration). Expected:
**average latency near the aggressive setting, success near the full-depth
setting** — non-uniform compute over time, gated by the same controller the
mentor's flow already describes ("Need extra focus?" → now also "need full
depth?").

---

## 5. Status summary

| Method | Axis | Model | Training-free | Status | Headline |
|---|---|---|---|---|---|
| Foveation-by-blur | spatial | both | ✅ | done (neg.) | no latency gain |
| FastV token prune | spatial | UniVLA | ✅ | measured | **bad trade** (succ↓, lat flat) |
| ToMe merge | spatial | SpatialVLA | ✅ | built, untested | OOD-safe, needs benchmark |
| Layer pruning | depth | UniVLA | ✅ | measured | **+** free 10% @100%, scales |
| Phase-adaptive depth | depth | UniVLA | ✅ | next | non-uniform compute (our contribution) |
| Temporal caching | temporal | both | ✅ | planned | stackable, modest expected |

*Artifacts in repo: `docs/SURVEY.md` (literature), `adaptive_sparse_vla/fastv_emu3.py`
+ `experiments/FastV_univla.md` (spatial/UniVLA), `SpatialVLA/experiments/tome/`
(spatial/SpatialVLA), layer-pruning via `LLM_PRUNE_COUNT` env in `eval.py`.*
