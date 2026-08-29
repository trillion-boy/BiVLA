# Literature audit and position of the six interventions

Audit date: 27 August 2026. This review separates the six implemented or
evaluated intervention families into positive candidates, negative controls,
and a recent baseline. Here, *valid* means that an intervention is
training-free and its mechanism is implemented coherently. It does not mean
that the intervention has demonstrated statistically reliable robot-task
success in this workspace.

## Inventory and claim status

| # | intervention | role in the paper | present verdict |
|---:|---|---|---|
| 1 | fixed foveation | spatial negative/robustness control | valid image transform, but no model-speed mechanism at fixed resolution |
| 2 | unconditional action repeat | temporal negative control | reduces model calls but introduces an unsafe open-loop trade-off |
| 3 | calibrated decoder-layer removal | positive candidate | small local speedup; task-success non-inferiority is untested |
| 4 | guarded action reuse | positive candidate | large conditional saving; safe trigger rate is untested |
| 5 | VLA-Cache | required recent baseline | literature-backed speed mechanism, but not locally Pareto-positive |
| 6 | interaction-aware temporal fusion and shared cache mask | primary positive candidate | exact identical-frame action; rollout and selective-cache gains are untested |

Accordingly, the repository contains six intervention families, but only
three should be presented as candidate contributions: calibrated depth,
guarded action reuse, and interaction-aware fusion/cache reuse. Fixed
foveation and unconditional action repeat are controls. VLA-Cache is a
baseline. Motion, entropy, task relevance, spatial dilation, keyframes, and
gripper guards are components of these families rather than additional
standalone tricks.

## 1. Fixed foveation — spatial negative control

The geometry-preserving foveation transform keeps a sharp central disc and
progressively blurs the image periphery. It is mechanically valid and useful
for testing whether OpenVLA is robust to peripheral clutter. At unchanged
output resolution, however, OpenVLA still produces 256 projected visual
tokens and executes the same vision encoder, projector, and language-model
computation. The CPU blur adds preprocessing work instead of removing model
work. The earlier notebook results also showed task-dependent success changes.

**Decision:** retain fixed foveation as a robustness and negative-control
ablation. Do not describe it as a speed trick unless a future implementation
also changes resolution, token count, or model computation. The untraceable
“RetinaBased” claim in the notebook should not be cited until its source is
identified.

## 2. Unconditional action repeat — temporal negative control

Repeating each predicted delta action genuinely reduces model calls per
environment step, but increases open-loop displacement and delays visual
feedback. The earlier results report an accuracy decrease, which conflicts
with the paper's target of lower latency without lost success. It also stops
being a matched intervention when policies emit action chunks of different
lengths.

Action-reuse work nevertheless supports studying temporal redundancy.
FlashVLA uses training-free action reuse as part of its acceleration pipeline,
although the full method reports a 0.7-point average success decrease.
SpecPrune-VLA similarly distinguishes coarse movement from precision-sensitive
manipulation phases.

**Decision:** use fixed repeat only to demonstrate why unconditional temporal
shortcuts are insufficient. Guarded action reuse is the candidate replacement.

## 3. Calibrated decoder-layer removal — positive candidate

The block-influence score based on `1 - cosine(input, output)` originates in
ShortGPT and is used by EfficientVLA. This supports the hypothesis that some
decoder blocks contribute little on representative VLA inputs and can be
removed without retraining. However, ShortGPT's broad language-model results
and EfficientVLA's CogACT experiments do not establish non-inferiority for
base OpenVLA on LIBERO.

The first notebook protocol had four problems: it ranked layers on a single
episode's first frame, did not protect the final block, could silently select
adjacent layers, and tested an artificial cache. The maintained implementation
aggregates influence over a disjoint calibration set, protects early and final
layers, enforces non-adjacency, structurally removes selected blocks, and
reindexes the generation cache. A local two-layer test still flipped the
binary gripper decision despite small aggregate action L2.

**Decision:** evaluate conservative operating points, beginning with one of
32 blocks, and promote the method only if paired LIBERO evaluation establishes
success non-inferiority. Larger removal budgets are stress tests or ablations.

## 4. Guarded action reuse — positive control-level candidate

Guarded reuse observes the current image before deciding whether to skip a
model call. It requires two recent densely inferred actions to agree, stable
global and local image signatures, no gripper transition, motion outside the
fine/contact regime, and no preceding reuse beyond a fixed cap. The local
patch guard is important: a gripper or object change can be hidden by a small
whole-frame mean error.

This mechanism differs from fixed repeat because every reuse is conditional
and immediately falls back to dense inference when any safeguard fails. Its
value depends on two quantities that static diagnostics cannot determine:
how often the guard fires during real rollouts and whether those reused steps
preserve contact-sensitive behavior.

**Decision:** retain it as a candidate, report its trigger and fallback rates
by task phase, and calibrate all thresholds on a held-out split. It is not a
perception-only contribution, so the paper should label it as a complementary
control-level trick.

## 5. VLA-Cache — required recent perception baseline

VLA-Cache is a NeurIPS 2025 training-free method with released OpenVLA code.
It compares consecutive frames, identifies static patches, protects patches
that receive strong task-conditioned attention, and reuses selected KV
computation. Unlike token dropping, historical information remains available
for reused patches.

On base OpenVLA across LIBERO, the paper reports average success of 75.0% for
dense inference and 74.7% for VLA-Cache, with latency changing from 51.91 ms
to 31.83 ms. This is a strong efficiency result with nearly maintained, not
improved, average accuracy. A 50-token spatial-reuse ablation reports 84.4%
to 85.4%, and the OpenVLA-OFT experiment reports 96.8% to 97.4%, but neither
number replaces paired reproduction on the target base-OpenVLA protocol.

The audited local reproduction corrected a dimensionality bug in the released
attention-ranking helper. On identical synthetic frames, exact-action settings
were slower, while settings with a small steady-state speedup flipped the
gripper decision. This diagnostic does not contradict the published rollout
average; it shows that action safety cannot be inferred from aggregate L2 or
from visual similarity alone.

**Decision:** retain VLA-Cache as the primary recent baseline and an enabling
mechanism, not as a confirmed positive trick in this workspace.

## 6. Interaction-aware temporal fusion and shared cache mask — primary candidate

TTF-VLA (AAAI 2026) reports that training-free temporal token fusion improves
base OpenVLA LIBERO success from 68.4% to 72.4% with less than 2% runtime
overhead. Its evaluated fusion is a temporal-denoising intervention rather
than a speed mechanism; direct KQV reuse is presented as future work. This
provides accuracy-side motivation for combining temporal fusion with
selective cache reuse.

The first-party candidate creates one auditable patch mask from local motion,
image entropy, optional text-to-vision relevance, explicit protected regions,
spatial dilation, a reuse budget, and periodic dense keyframes. The projected
representations of only the safe patches are fused from the previous frame.
The same patch indices form the interface for selective KV/KQV reuse, so the
accuracy and speed mechanisms make a consistent decision rather than using
two independently tuned selectors.

VLA-InfoEntropy independently supports entropy-aware cache budgeting. Its
April 2026 preprint reports OpenVLA average LIBERO success of 75.0% to 76.4%
and CUDA latency of 51.91 ms to 31.25 ms, but no official implementation was
found during this audit. IAprune similarly reports an interaction-aware
success/speed gain for OpenVLA-OFT, but it is a token-pruning method and its
main acceleration result is not on vanilla OpenVLA.

A recent anonymous submission also studies coordinated cross-modal token
reuse. Therefore, the paper should not claim that a shared mask alone is
novel. The defensible differentiators are contact-aware fallbacks, one mask
jointly governing temporal denoising and computation reuse, an optimized
dense-attention baseline, and paired base-OpenVLA evaluation.

The local identical-frame diagnostic reused 128 of 256 projected visual
tokens and preserved the action exactly. Fusion latency was 158.36 ms versus
157.63 ms for dense SDPA. This is compatibility evidence only: projected-token
fusion does not reduce FLOPs, and selective KV/KQV execution is not yet
connected to this mask.

**Decision:** make the combined fusion/cache formulation the primary
perception candidate. Treat temporal fusion alone as the accuracy-side
ablation and shared-mask selective cache reuse as the required speed-side
extension.

## Exact inference optimization — required baseline, not a seventh trick

OpenVLA supports FlashAttention 2 and PyTorch SDPA. On the RTX 5090, SDPA gives
fused CUDA attention without an additional compiled extension. This changes
the inference implementation rather than the policy definition, although
action-token parity must still be measured because BF16 kernels may differ
numerically.

Every intervention must be compared with optimized dense SDPA or
FlashAttention in inference mode. Comparing only with an artificially slow
eager implementation would inflate acceleration claims. Exact kernel choice
is therefore a systems control, not a seventh trick or a paper contribution.

## Revised paper position

The six interventions support a coherent, falsifiable argument rather than a
collection of uniformly positive perturbations:

> Training-free VLA acceleration succeeds only when computation is removed
> conditionally along redundancies aligned with robot interaction. Static
> spatial degradation and unconditional temporal reuse expose the failure of
> generic shortcuts, while task-aware visual reuse, calibrated depth
> redundancy, and safeguarded action reuse are candidates for moving the
> speed-success Pareto frontier.

The central empirical question is not whether all six tricks work. It is
whether the three guarded candidates pass a common paired evaluation after
the two negative controls establish why their safeguards are necessary and
VLA-Cache anchors comparison with recent work. Synthetic action agreement is
only a compatibility diagnostic. No candidate should be called positive until
it passes the gates in `docs/experiment_protocol.md`.

## Primary sources

- [OpenVLA (2024)](https://arxiv.org/abs/2406.09246)
- [ShortGPT (2024)](https://arxiv.org/abs/2403.03853)
- [VLA-Cache, NeurIPS 2025](https://arxiv.org/abs/2502.02175)
- [EfficientVLA (2025 preprint)](https://arxiv.org/abs/2506.10100)
- [FlashVLA (2025 preprint)](https://arxiv.org/abs/2505.21200)
- [TEAM-VLA (2025 preprint)](https://arxiv.org/abs/2512.09927)
- [TTF-VLA (AAAI 2026)](https://arxiv.org/abs/2508.19257)
- [Coordinated cross-modal token reuse (anonymous submission)](https://openreview.net/pdf?id=R6d86jMO74)
- [SpecPrune-VLA (2026 preprint)](https://arxiv.org/abs/2509.05614)
- [VLA-InfoEntropy (2026 preprint)](https://arxiv.org/abs/2604.05323)
- [VLA-IAP / IAprune (2026 preprint)](https://arxiv.org/abs/2603.22991)
- [ActionCache (2026 preprint; flow-based VLAs, not OpenVLA)](https://arxiv.org/abs/2607.06370)
