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

## Gemma2 depth pruning on SpatialVLA (measured, N=24 per cell)

We repeated the depth-pruning experiment (validated on UniVLA/Emu3) on
SpatialVLA's Gemma2 decoder (26 layers), using the same training-free
mechanism: bypass the `count` layers with highest `cos(layer_input,
layer_output)` (= most redundant), calibrated once via forward hooks on the
first real step. Implementation: `SpatialVLA/experiments/tome/depth_prune_gemma2.py`,
wired into `SpatialVLA/experiments/tome/tome_spatialvla_eval.py` via
`--depth-prune N`.

**Sanity-checking the ranking (eggplant, count=2).** Because Gemma2's per-layer
redundancy values were unexpectedly flat across the middle of the network
(`L8..L23` all in a narrow 0.88–0.93 band vs. UniVLA's presumably wider spread —
see raw values below), we ran two control conditions to check whether the
ranking carries real signal or is picking essentially at random:

| mode | layers bypassed | success | note |
|---|---|---|---|
| **redundant** (our method) | [8, 10] | **41.7%** | best of the three |
| random | [11, 22] | 12.5% | much worse than redundant |
| least (deliberately "important") | [2, 4] | **generation never terminates** — hits the `max_new_tokens=256` cap almost every step (~865 ms → 12–13 s/step) instead of emitting EOS after ~12 tokens | catastrophic |

Ranking: **redundant > random > least (broken)**, in the predicted order — this
confirms the cosine-redundancy ranking carries real signal on Gemma2 despite
the flat middle band, it just discriminates less sharply than on Emu3. The
"least" collapse has a clean mechanistic explanation: layers 2 and 4 have low
cosine similarity (0.60, 0.75) because they perform large, foundational
transformations near the input; removing them corrupts the representation
every later layer depends on, unlike removing a near-identity middle layer.

**Per-layer redundancy (cos(in, out), eggplant calibration):**
```
L0=0.105  L1=0.634  L2=0.603  L3=0.771  L4=0.748  L5=0.900  L6=0.833  L7=0.885
L8=0.922  L9=0.921  L10=0.925 L11=0.901 L12=0.917 L13=0.916 L14=0.900 L15=0.885
L16=0.915 L17=0.918 L18=0.905 L19=0.919 L20=0.920 L21=0.913 L22=0.902 L23=0.914
L24=0.860 L25=0.676
```
Only the front (`L0-L4`) and back (`L25`) discriminate sharply; 16 consecutive
middle layers sit within a 0.045 band — likely a Gemma2-specific
"massive-activation" effect where the residual stream's dominant direction
persists across layers regardless of how much useful work each layer does.

**Minimum viable cut (count=1, the single top-ranked layer) across 4 tasks:**

| task | baseline success | prune=1 success | Δ (N=24, |Δ|/SE) | latency speedup |
|---|---|---|---|---|
| eggplant | 66.7% | 37.5% | **-29.2pt (3.04 SE — real)** | 1.00× |
| carrot | 25.0% | 29.2% | +4.2pt (0.48 SE — noise) | 1.04× |
| stack | 33.3% | 25.0% | -8.3pt (0.86 SE — borderline) | 1.03× |
| spoon | 8.3% | 0.0% | -8.3pt (1.47 SE — collapses to 0) | 1.03× |

**Verdict: unlike UniVLA/Emu3 (which tolerated bypassing 4/8 = 50% of layers
with maintained/improved success on most tasks), SpatialVLA/Gemma2 does not
have a broadly-safe static depth-pruning operating point** — even the single
most-redundant layer (1/26 = 3.8%) hurt 3 of 4 tasks, and only carrot was
unaffected. This is a genuine backbone-dependent finding, not a bug (confirmed
via the redundant/random/least triangulation above): Gemma2's exploitable
layer-level redundancy is narrower and more task-specific than Emu3's.

**Practical implication:** SpatialVLA's primary latency lever is **temporal
caching** (stride 2, ~4% real latency reduction, maintained/improved success —
see below), not depth pruning. Depth pruning remains UniVLA's primary lever.
Gemma2 depth pruning may still be usable, but only selectively for
carrot-like (planar-placement) tasks — reinforcing, on a second backbone, that
static/uniform application of any single technique is unsafe and the right
frame is *which axis, on which backbone, for which archetype*.

## Status of methods tried

| Method | Axis | Backbone | Success | Latency | Verdict |
|---|---|---|---|---|---|
| foveation-by-blur | spatial | both | — | none | token count unchanged |
| FastV | spatial (LLM tokens) | UniVLA | ↓↓ | none | wrong lever |
| ToMe (merge+restore) | spatial (ViT tokens) | SpatialVLA | maintained | none | OOD-safe, no latency |
| layer pruning | **depth (LLM)** | UniVLA | maintained/↑ | **modest ↓** | works; selective per archetype |
| Gemma2 depth pruning | depth (LLM) | SpatialVLA | maintained on 1/4 tasks only | modest ↓ (~3-4%) when it works | narrow/task-specific; not a broad lever on this backbone |
| temporal caching (stride 2) | temporal | SpatialVLA | maintained (71%) | **~4% ↓** (1.04×) | best broad SpatialVLA latency lever |

*Artifacts: `docs/DEPTH_PRUNING_RESULTS.md`, `experiments/FastV_univla.md`,
`experiments/DepthController_univla.md`, `SpatialVLA/experiments/tome/`,
`docs/NON_UNIFORM_REPRESENTATION.md`.*
