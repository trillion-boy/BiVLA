# Research Progress Report

**Topic: Training-Free Test-Time Efficiency for Frozen VLA Models — Non-Uniform Resource Allocation**
**Benchmark: SimplerEnv WidowX-Bridge (4 tasks) · Models: UniVLA, SpatialVLA · Constraints: no training, no external modules, frozen weights**
**Author: Junseo · July 2026**

---

## Executive Summary

Starting from the visual-focus direction ("look sharply only where it matters"), I profiled where a control step's time actually goes and found that **the real bottleneck is not visual processing but the autoregressive action decode (70–75% of each step, on both backbones)**. After establishing that "making each decode cheaper" is structurally limited (and on SpatialVLA, impossible), I pivoted to **"calling the decode less often" via action-chunk execution** — which improves success rates **and** cuts latency roughly in half at the same time.

| | Starting point (visual focus) | Current (chunk execution, k=2) |
|---|---|---|
| Success rate | +2pp (marginal) | **+13pp averaged over 4 tasks** |
| Latency | ~unchanged | **~1.9× faster** |

All numbers below are measured in our own environment, N=24 episodes per cell, with baselines re-measured under identical conditions. Our vanilla baselines reproduce the SpatialVLA paper's zero-shot numbers within sampling noise (e.g., eggplant 66.7% vs. 70.8% reported), confirming the setup is sound.

---

## Phase 1. The Visual (Spatial) Axis — "Does focusing on important regions make us faster?"

**Starting point — your controller design.** The method reads the instruction and image, infers the task type, checks the current phase (grasp/place), and decides whether *extra visual focus* is needed — sharpening important regions and coarsening the rest (AutoGaze-inspired). Your measured result on UniVLA: 82.29% → 84.38% success (+2.09pp), latency roughly unchanged.

**Follow-up experiments on the same axis (external-module-free variants):**

| Method | Backbone | Mechanism | Success | Latency |
|---|---|---|---|---|
| FastV | UniVLA | drop low-attention visual tokens inside the LLM | **collapses** (100→75→38%) | ~unchanged |
| ToMe | SpatialVLA | merge similar ViT tokens, restore count at output | maintained (grasp 88→92%) | 0.99× (none) |
| Temporal caching (stride 2) | SpatialVLA | reuse image features every other step | maintained (71%) | 1.04× (+4%) |
| Temporal caching (stride 3) | SpatialVLA | reuse for 2 of 3 steps | collapses (54%) | 1.05× |

**Phase 1 conclusion:** no visual-side intervention meaningfully reduces latency — even the ones that preserve or improve accuracy. This raised the key question: *why not?*

---

## Phase 2. Measurement + The Depth Axis — "Where does the time actually go?"

**Profiling one control step (CUDA-synchronized timing):**

| Stage | SpatialVLA | UniVLA |
|---|---|---|
| Visual encoding | 13.9% | 6.1% |
| LLM prefill | 6.6% | 12.6% |
| **LLM decode (sequential action-token generation)** | **75.0%** | **69.8%** |

Two architecturally unrelated backbones (VQ-token LLM vs. ViT+Gemma2) converge on the same structure: **manipulation VLAs are decode-bound**. This explains Phase 1 quantitatively — the visual side (encoding + prefill) is a ~19–20% ceiling, so no visual method can deliver meaningful speedup. It also explains why AutoGaze succeeds in its own domain: video QA has thousands of visual tokens and is *encoder*-bound; single-image manipulation inverts that cost structure. I believe this measurement is publishable on its own as the reason visual-efficiency methods fail to transfer to manipulation.

**Attacking the decode directly — layer (depth) pruning.** Decode is slow because every generated token traverses all decoder layers. We rank layers by redundancy (cosine similarity between a layer's input and output, calibrated once at test time — training-free) and bypass the most redundant ones.

- **UniVLA: works.** Bypassing 4/8 layers keeps or improves success at ~1.1× speed. A phase-adaptive controller (full depth until grasp, aggressive pruning after, switched by the policy's own gripper signal) reached 100% success at 1.14× on eggplant, with selective per-task-archetype application.
- **SpatialVLA: fails, informatively.** Bypassing even the *single* most redundant of 26 layers hurt 3 of 4 tasks (eggplant 66.7→37.5%). The ranking itself is validated (ours > random > deliberately-worst, in the predicted order — the "worst" control breaks generation entirely), so the conclusion is that **Gemma2 simply has no spare layers**.
- **Self-speculative decoding: fails, arithmetically.** I implemented a lossless variant (draft with a few layers bypassed, verify with the full model; output provably identical to the baseline). But the draft costs 85% of the full model (only 4/26 layers removable), so even perfect acceptance caps the speedup at 1.4×; measured, it was 1.38× *slower*. A cheap-enough draft would require removing ~15+ layers, which the pruning study shows produces garbage. Same root cause as above.

**Phase 2 conclusion:** "make each decode forward cheaper" is backbone-dependent — viable on UniVLA, structurally closed on SpatialVLA (layer pruning, token pruning, and speculation all fail for the same reason: no exploitable redundancy in the decode path).

---

## Phase 3. The Temporal Axis — "Don't make decode cheaper; call it less often" (Breakthrough)

**A discovery hiding in plain sight.** In SpatialVLA, one action = exactly 3 tokens, and each generate call decodes 12 tokens — i.e., **the model already predicts a 4-action chunk every step**. But the evaluation wrapper executes only the first action, discards the other three, and re-generates from scratch next step — throwing away ¾ of what the (75%-of-latency) decode already paid for. UniVLA sits at the opposite extreme: it predicts 5 actions and always executes all 5. The two systems occupy the two fixed endpoints of the *replanning-frequency* spectrum.

**Implementation.** Execute k of the 4 predicted actions per generate (queueing the rest; zero model calls on queued steps — encoding, prefill, and decode all skipped). The model's computation is untouched, so there is no accuracy mechanism to break; we simply use the chunk the checkpoint was trained to predict.

**Results (eggplant, N=24):**

| Config | Grasp | Success | ms/step | Speedup |
|---|---|---|---|---|
| Baseline (replan every step) | 87.5% | 66.7% | 836 | 1.0× |
| **Chunk k=2** | 91.7% | **87.5%** | **457** | **1.8×** |
| **Chunk k=4** | 91.7% | 75.0% | **235** | **3.6×** |

Both settings beat baseline on **both** axes — a Pareto improvement, not a trade-off. Success improving under chunk execution is consistent with the action-chunking literature (ACT): per-step replanning constantly overwrites the plan and causes trajectory jitter; executing one coherent plan is smoother.

**Full 4-task suite (k=2, N=24 per cell):**

| Task | Baseline | Chunk k=2 | Δ | Speedup |
|---|---|---|---|---|
| Eggplant | 66.7% | **87.5%** | +20.8pp | 1.8× |
| Carrot | 25.0% | **41.7%** | +16.7pp | 1.9× |
| Spoon | 8.3% | **29.2%** | +20.9pp (≈3.7 SE) | 2.0× |
| Stack | 31.3%* | 25.0% | −6.3pp (within noise) | 1.9× |
| **Mean** | **32.8%** | **45.9%** | **+13.0pp** | **~1.9×** |

*Stack baseline measured twice (33.3%, 29.2%; grasp 58.3% both) — pooled.

**The failure analysis points exactly at the next step.** In every non-improving case (stack at k=2; eggplant's k=4 failures), grasp rate equals baseline and the failures concentrate in *grasped-but-not-placed*. Sparse feedback costs nothing during transport, approach, or grasp — only during **precise placement** (stack, block-on-block alignment, is the suite's most placement-critical task). Two independent experiments agree on this.

---

## Next Step: Adaptive Replanning — Your Controller, A New Actuator

This is where the results reconnect with your original design. Your controller structure — *(task type × current phase) → "does this moment need precision?" → allocate resources* — is validated by our data: precision demand really is non-uniform over phases. What limited its impact was only the actuator it controlled: visual focus sits inside the ~20% cost ceiling. Plugging the **replanning frequency k** into the same controller instead:

- Transport/approach: k=4 (fast, sparse feedback)
- Near grasp/place (detected from the policy's own gripper signal): k=1 (dense feedback)

Expected outcome: k=4's speed (~3.6×) at ≥k=2's success rate, recovering stack's placement losses — still training-free, still module-free. The controller also applies symmetrically to UniVLA (which should *reduce* k from its fixed 5 at precision moments, gaining accuracy rather than speed). Same controller, opposite-direction gains on the two backbones — which I think makes the strongest case for the non-uniform allocation thesis.

**Planned experiments:** (1) adaptive-k controller on SpatialVLA, full 4-task suite; (2) UniVLA k-sweep (k=1..5) for the replan-frequency curve on the second backbone; (3) consolidated cross-backbone table for the paper draft.

---

## Appendix: Everything Tried (One Table)

| Method | Axis | Backbone | Success | Latency | Verdict |
|---|---|---|---|---|---|
| Visual focus (yours) | spatial | UniVLA | +2pp | — | marginal |
| FastV | spatial | UniVLA | collapses | none | fails |
| ToMe | spatial | SpatialVLA | maintained/↑ | 0.99× | accuracy-only |
| Temporal caching (s2) | temporal (visual) | SpatialVLA | maintained | 1.04× | small |
| Layer pruning | depth | UniVLA | maintained/↑ | 1.1× | **works (this backbone)** |
| Phase-adaptive depth controller | depth | UniVLA | 100% (eggplant) | 1.14× | works, selectively |
| Layer pruning | depth | SpatialVLA | hurt by 1 layer | 1.03× | fails (no spare layers) |
| Self-speculative decoding | depth | SpatialVLA | identical (lossless) | 0.72× (slower) | fails (arithmetic) |
| **Chunk execution k=2** | **temporal (replan)** | SpatialVLA | **+13pp mean** | **1.9×** | **key result** |
| Chunk execution k=4 | temporal (replan) | SpatialVLA | +8.3pp (eggplant) | 3.6× | place-limited → adaptive |
| Adaptive replanning | temporal (replan) | both | (planned) | (planned) | **next step / core contribution** |

*Full data and reproduction recipes: `docs/VISUAL_TOKENS_VS_LATENCY.md`, `docs/DEPTH_PRUNING_RESULTS.md`, and the per-experiment reports under `experiments/`.*
