# Visual-token reduction does not reduce latency on frozen manipulation VLAs

**Central finding.** On frozen, single-image manipulation VLAs (UniVLA/Emu3,
SpatialVLA/PaliGemma2) evaluated on SimplerEnv WidowX-Bridge, **reducing the
number of visual tokens does not reduce wall-clock latency** — even when it
preserves (or improves) task success. The per-step cost is dominated by the
**autoregressive action decode through the large LLM**, not by the vision
encoder or the visual prefix. This holds across two backbones and two distinct
token-reduction mechanisms.

## Evidence (measured, N=24, pure model-inference ms)

| Method | Backbone | What it does to the visual path | Success | Latency |
|---|---|---|---|---|
| **FastV** | UniVLA/Emu3 | drop low-attention visual tokens in the LLM | **collapses** (100→75→38%) | ~1.0× (flat) |
| **ToMe** | SpatialVLA/SigLIP | merge ViT tokens + restore count | **maintained** (67% = baseline; grasp 88→92%) | **0.99×** (none) |
| **Temporal cache (stride 2)** | SpatialVLA | reuse SigLIP features every other step | **maintained** (71% ≈ 67%; grasp 83%) | **1.04×** (~4%) |
| **Temporal cache (stride 3)** | SpatialVLA | reuse SigLIP features 2 of 3 steps | **collapses** (54%; grasp 67%) | 1.05× |

Temporal caching (stride 2) is the **only** visual-side method that moved latency
at all — and only ~4%, realizing a small slice of the 20.5% prefill-side ceiling
(stride 3's extra latency gain is negligible while success collapses from stale
features). This confirms the ceiling empirically: visual-side methods top out at a
few percent; the decode is untouched.

- SpatialVLA baseline reproduced the prior report exactly (eggplant 87.5% grasp /
  66.7% success), confirming a correct setup; ToMe held success at 66.7% with
  **no** latency gain (836 → 844 ms).
- UniVLA FastV barely moved latency while destroying success.

## Measured per-step breakdown (SpatialVLA, profiler)

Splitting one control step (CUDA-synced timing, avg over 25 steps):

| stage | ms | share |
|---|---|---|
| encoding (SigLIP ViT) | 125 | 13.9% |
| LLM prefill (read prompt once) | 60 | 6.6% |
| **LLM decode (12 action tokens × 56 ms)** | **677** | **75.0%** |
| env / python | 41 | 4.5% |
| **total** | **903** | 100% |

→ **Prefill-side ceiling (encoding + prefill) = 20.5%.** Any token-reduction or
temporal-caching method can save *at most* ~20%, and realistically less. The
decode is **75%** and only **LLM-depth reduction** shrinks it (it makes every one
of the 12 sequential token-forwards cheaper). This is the quantitative reason
ToMe netted 0.99×: it targeted the 14% ViT and the merge overhead cancelled it.

## Measured per-step breakdown (UniVLA/Emu3, profiler)

Same profiler on the *other* backbone (VQ tokenizer + Emu3 decoder, avg over 6
steps, eggplant):

| stage | ms | share |
|---|---|---|
| encoding (VQ tokenizer) | 83 | 6.1% |
| LLM prefill (read prompt once) | 172 | 12.6% |
| **LLM decode (26 action tokens)** | **951** | **69.8%** |
| env / python | 157 | 11.5% |
| **total** | **1362** | 100% |

→ **Prefill-side ceiling = 18.7%**, decode = **69.8%**. Two architecturally
distinct backbones — Emu3 (VQ-token LLM) and PaliGemma2 (SigLIP ViT + Gemma2) —
**converge** on the same structure: decode ≈ 70–75%, prefill-side ceiling ≈
19–20%. The decode-bound property is therefore not a quirk of one model but a
**structural property of single-image autoregressive manipulation VLAs**: every
control step decodes a short action sequence one token at a time through the full
LLM, and that sequential decode dominates wall-clock regardless of how the visual
front-end is built.

## Why — the cost is decode-bound, not vision-bound

A manipulation step = encode one image (≈256 visual tokens) → **prefill** the
prompt once → **autoregressively decode** the action (many sequential single-token
forwards through the full LLM, every control step). The dominant wall-clock cost
is that sequential decode (memory-bandwidth / FFN bound). The 256 visual tokens
are a small, fixed prefix:

- Cutting them shrinks the **prefill** and the visual **KV** the decode attends
  to — but the decode's per-token **FFN** cost is unchanged, so wall-clock barely
  moves.
- Merging inside the ViT and **restoring** the count (our safe ToMe) speeds only
  the SigLIP ViT's middle layers — a small fraction — and the merge overhead
  cancels it (net 0.99×).

## Reconciling with AutoGaze (why it *does* cut latency)

AutoGaze (CVPR'26) reports 4–100× token reduction and real latency gains — in
**video understanding**, a fundamentally different cost structure:

| | AutoGaze (Video QA) | Ours (single-image VLA) |
|---|---|---|
| visual tokens | **thousands–tens of thousands** (the bottleneck) | ~256 (small fixed prefix) |
| cost concentrated in | **encoder** (encoding huge visual input) | **decoder** (autoregressive action gen) |
| token reduction effect | latency ↓↓ | latency ≈ unchanged |

In video QA the visual tokens *are* the sequence and the work is encode-heavy, so
4–100× fewer tokens cuts real time. In single-image manipulation the visual
tokens are a small prefix and the work is decode-heavy, so the same idea doesn't
transfer. The mentor's plan to adapt AutoGaze to VLAs was reasonable; our
experiments empirically reveal that the **cost structure inverts** for
manipulation, which the video setting hides.

**Also: FLOPs ≠ wall-clock.** Token-reduction papers often report FLOPs. An
autoregressive decoder's wall-clock is governed by sequential token generation,
not FLOPs, so cutting visual FLOPs need not cut measured latency. We measured
wall-clock (ms/infer), which exposes this gap.

## What this implies

1. **Latency lever = the LLM, not the visual tokens.** The only intervention that
   actually reduced wall-clock here was **LLM layer (depth) pruning** on UniVLA
   (free ~10% at 100% success; bigger with more pruning, task-dependent). The
   analog for SpatialVLA is Gemma2 depth pruning.
2. **Visual-token reduction is still valuable — for accuracy, not latency.** ToMe
   held/raised grasp (88→92%) OOD-safely; training-free layer pruning *raised*
   success on 3/4 tasks. So "non-uniform visual representation" is best framed as
   an **accuracy-preserving / sometimes-improving** intervention, applied
   **selectively per archetype**, with latency coming from the depth axis.
3. **Where visual reduction *would* help latency:** regimes where visual tokens
   dominate — long video context, many frames, multi-view, or very high-resolution
   inputs. Single-image manipulation is not that regime.

## Status of methods tried

| Method | Axis | Backbone | Success | Latency | Verdict |
|---|---|---|---|---|---|
| foveation-by-blur | spatial | both | — | none | token count unchanged |
| FastV | spatial (LLM tokens) | UniVLA | ↓↓ | none | wrong lever |
| ToMe (merge+restore) | spatial (ViT tokens) | SpatialVLA | maintained | none | OOD-safe, no latency |
| layer pruning | **depth (LLM)** | UniVLA | maintained/↑ | **modest ↓** | works; selective per archetype |
| Gemma2 depth pruning | depth (LLM) | SpatialVLA | (next) | (expected ↓) | planned |
| temporal caching (stride 2) | temporal | SpatialVLA | maintained (71%) | **~4% ↓** (1.04×) | best visual-side latency, still modest |

*Artifacts: `docs/DEPTH_PRUNING_RESULTS.md`, `experiments/FastV_univla.md`,
`experiments/DepthController_univla.md`, `SpatialVLA/experiments/tome/`,
`docs/NON_UNIFORM_REPRESENTATION.md`.*
