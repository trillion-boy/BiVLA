# Survey: Training-Free, Latency-Reducing Methods for Frozen VLAs

**Scope.** For *frozen* Vision-Language-Action models (e.g. UniVLA/Emu3, SpatialVLA/PaliGemma2)
on robot manipulation (SimplerEnv / WidowX Bridge), which **training-free, inference-time**
methods **reduce latency** while **maintaining or improving task success**?

**Thesis (our project).** A lightweight controller that applies **AutoGaze-style visual
token reduction** selectively (per task / per phase) so that focus **subtracts compute**
instead of adding it.

> **Verification status.** Produced by a deep-research harness (search → fetch → adversarial
> verify → synthesize). The web session limit interrupted verification, so:
> - ✅ **3/3 verified** this run: FastV, FasterVLM, LLaVA-PruMerge core claims.
> - ▣ **sourced (direct quote from primary arXiv) but verification incomplete**: SparseVLM,
>   HiRED, FiCoCo, VLA-Cache, EfficientVLA, DeeR-VLA. Numbers are quoted from the papers —
>   reliable, but re-run verification (after limit reset) to fully confirm.

---

## 1. The core finding for us

Our own experiments showed **foveation-by-blur does NOT reduce latency** (we cut sharp
patches 16.7 → 4.9 but wall-clock stayed flat) because a frozen VLA processes the **same
number of tokens** regardless of how many are sharp. The literature confirms the only lever
that moves latency is **actual visual-token reduction** (drop / merge / cache), and—crucially—
**two training-free VLA-specific methods already do this on SIMPLER-style benchmarks**
(EfficientVLA, VLA-Cache). They are both our closest prior art and our best templates.

---

## 2. Comparison table

| Method | Train-free? | Where it cuts tokens | Latency / FLOPs | Accuracy effect | OOD-safe on frozen? | VLA? | Ref |
|---|---|---|---|---|---|---|---|
| **FastV** ✅3/3 | ✅ | LLM, after layer 2 (attention score) | **45% FLOPs↓** | maintained | ✅ late-layer pruning | VLM (portable) | arXiv:2403.06764 (ECCV'24 Oral) |
| **FasterVLM** ✅3/3 | ✅ | **pre-LLM** (after visual encoder, [CLS]-attn) | **>95% FLOPs↓** | 95% pruned → ~89.4% retained | medium (drop) | VLM | arXiv:2412.01818 |
| **LLaVA-PruMerge** ✅3/3 | ✅ / opt. FT | CLIP encoder (pre-LLM), **prune + merge** | 4–10× FLOPs (prefill) | ~5.5% tokens, competitive (≈18×, 14× MME/TextVQA) | ✅ merge | VLM | arXiv:2403.15388 |
| **SparseVLM** ▣ | ✅ | LLM decoding, **text-guided** | 54% FLOPs↓, **wall-clock 37%↓** | 97% retained | medium | VLM | arXiv:2410.04417 |
| **HiRED** ▣ | ✅ | ViT encoding (CLS budget) | **latency 78%↓, throughput 4.7×** @20% budget | — | medium (drop) | VLM | arXiv:2408.10945 |
| **FiCoCo** ▣ | ✅ | ViT (-V) or LLM (-L); filter-correlate-compress | 5.7× / 14.7× FLOPs | 92.8% / 93.6% retained; beats SparseVLM ~6% | ✅ info-recovery (merge) | VLM | arXiv:2411.17686 |
| **ToMe** | ✅ | between ViT blocks, **merge** | ~2× throughput | small drop | ✅ merge | ViT | ICLR'23, facebookresearch/ToMe |
| **🤖 VLA-Cache** ▣ | ✅ | **temporal token caching** | 1.63×, 27.3% FLOPs↓ (LIBERO); 1.7× + ctrl-freq +15% | **−0.3% success** | ✅ reuse | **✅ VLA** | arXiv:2502.02175 (NeurIPS'25) |
| **🤖 EfficientVLA** ▣ | ✅ | LLM layers + **task-aware token select** + action-head cache | **1.93×, FLOPs→28.9% (SIMPLER)** | **−0.6% success** | ✅ | **✅ VLA** | arXiv:2506.10100 (NeurIPS'25) |
| **🤖 DeeR-VLA** ▣ | ❌ (trains exits) | dynamic depth / early-exit | situational compute ↓ | maintained | — | **✅ VLA** | arXiv:2411.02359 (NeurIPS'24) |
| **AutoGaze** (ours) | ✅ (pretrained) | **pre-ViT** patch removal | 4–100× tokens (ViT **and** LLM) | maintained on video QA | ⚠️ multi-scale-into-ViT needs training | video → we apply to VLA | arXiv:2603.12254 (CVPR'26) |

---

## 3. Where each method cuts tokens (the key axis)

- **Inside the LLM (late layers):** FastV, SparseVLM. ViT still encodes all patches → ViT
  cost unchanged, but the (often dominant) LLM cost drops. **Most OOD-safe** because early
  layers already aggregated visual info before pruning.
- **At the ViT→LLM boundary (pre-LLM):** FasterVLM, LLaVA-PruMerge, HiRED. Tokens never enter
  the LLM. ViT still runs fully.
- **Inside / before the ViT:** ToMe (between blocks), **AutoGaze (before the ViT)** — only
  these cut the **ViT itself**. This is AutoGaze's distinctive advantage.
- **Across time (caching):** VLA-Cache reuses static tokens between frames.

---

## 4. VLA-specific prior art (read these first)

### EfficientVLA — arXiv:2506.10100 (NeurIPS 2025) — our closest competitor & template
Training-free, **on SIMPLER**. Combines three redundancy cuts: (1) prune inconsequential LLM
layers, (2) **task-aware visual-token selection** (compact, diverse subset), (3) cache action-
head (diffusion) features. On CogACT: **1.93× speedup, FLOPs → 28.9%, success −0.6%.** Claims
to outperform FastV and VLA-Cache. → Heavily overlaps our intended "controller + token
reduction + frozen + SIMPLER" plan; we must differentiate sharply (see §6).

### VLA-Cache — arXiv:2502.02175 (NeurIPS 2025)
Training-free temporal token caching: reuse static visual tokens across adjacent frames,
recompute only task-relevant/changed ones. **1.7× CUDA speedup, +15% control frequency**;
on LIBERO **1.63×, 27.3% FLOPs↓, −0.3% success.** Exactly the "temporal redundancy" lever.

### DeeR-VLA — arXiv:2411.02359 (NeurIPS 2024)
Dynamic multi-exit MLLM: easy situations exit early, hard ones use full depth. **Requires
training the exit heads** (not fully training-free), but it is the canonical "compute per
situation" design our controller resembles.

---

## 5. OOD safety on frozen backbones (supports our intuition)

- **Merge > drop.** FiCoCo integrates discarded tokens' info into retained tokens
  (correlate+compress) and keeps >92% accuracy under extreme compression, beating drop-based
  SparseVLM by ~6%. LLaVA-PruMerge and ToMe likewise **merge**, not hard-drop. → The OOD
  collapse we hit earlier from token *dropping* is mitigated by *merging*.
- **Late-layer pruning is safe.** FastV's premise: visual-token attention is highly
  inefficient in deep layers, so pruning *after* early layers barely hurts a frozen model.

---

## 6. Recommended directions for our project

| Priority | Direction | Rationale |
|---|---|---|
| **1** | **AutoGaze pre-ViT reduction + per-phase controller** | EfficientVLA/FastV cut LLM tokens but the **ViT still encodes all patches**; AutoGaze cuts the **ViT itself** — a genuine gap. Per-phase (grasp aggressive / place conservative) is finer than EfficientVLA's task-aware selection. |
| **2** | **Merge, don't drop** (FiCoCo / ToMe style) | Directly fixes the frozen-VLA OOD collapse; literature-backed. |
| **3** | **FastV-style late-LLM token pruning, controller-gated** | Easiest, OOD-safe latency win; complements the repo's existing decoder-layer pruning. |
| **4** | **Temporal caching** (VLA-Cache style) | Extra control-loop speedup; orthogonal, stackable. |

**Differentiation (sharpened, given EfficientVLA exists):**
1. **Cut the ViT, not just the LLM** — AutoGaze-style pre-ViT reduction; competitors stop at
   LLM/token-selection.
2. **Phase-level non-uniform budget** (grasp vs place) — "foveation feel" at the *token* level.
3. **Explicit OOD-safe merging** for frozen UniVLA/SpatialVLA (not CogACT).
4. **Honest negative result as motivation:** show foveation-by-blur fails to cut latency
   (our experiment), then beat that ceiling with token-level reduction.

**Pitfalls to avoid:**
- Token *dropping* → OOD collapse on frozen ViT (use merge).
- **Blur ≠ token reduction** → no latency gain (we proved this; do not repeat).
- SpatialVLA's visual tokens encode spatial position → naive removal breaks spatial reasoning.
- Control-loop **temporal jitter** if the focus/compression flickers between frames.
- Heavy add-ons (e.g., GroundingDINO per step) → *increase* latency, defeating the goal.

---

## References

- FastV — *An Image is Worth 1/2 Tokens After Layer 2*, ECCV 2024 (Oral). arXiv:2403.06764
- FasterVLM — arXiv:2412.01818
- LLaVA-PruMerge — arXiv:2403.15388
- SparseVLM — arXiv:2410.04417
- HiRED — arXiv:2408.10945
- FiCoCo (*filter-correlate-compress*) — arXiv:2411.17686
- ToMe — *Token Merging: Your ViT But Faster*, ICLR 2023. github.com/facebookresearch/ToMe
- VLA-Cache — arXiv:2502.02175 (NeurIPS 2025)
- EfficientVLA — arXiv:2506.10100 (NeurIPS 2025)
- DeeR-VLA — arXiv:2411.02359 (NeurIPS 2024)
- AutoGaze — *Attend Before Attention*, CVPR 2026. arXiv:2603.12254

*Generated via deep-research harness (23 sources, 94 claims, 10 verified 3/3 this session;
remaining claims sourced from primary arXiv with direct quotes, verification interrupted by
web session limit — re-run to fully confirm).*
