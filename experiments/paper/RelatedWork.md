# Related Work

*Reading draft of `relatedwork.tex`, citations spelled out, same content.
1044 words of prose. This is the LONG six-family draft (v3, 2026-09-03). The
final target is 0.75 page, about 800 words in ieeeconf, so roughly 240 words
come out in polishing. Cut candidates are listed under "Notes for the
co-authors". Provenance for every claim: `RelatedWork_Sources.md`, the new
claims in its last section.*

**Regenerate this file whenever `relatedwork.tex` changes.** It has fallen
behind three times, once carrying a sentence whose meaning had been inverted in
the `.tex` and fixed there but not here. `check_reading_copies.py` now tests
for that.

---

We cover the cost of VLA inference, the six intervention families we evaluate,
grouped by the role each plays in our argument, and how the literature tests
such claims.

### Inference cost in VLA policies

VLA policies adapt pretrained vision-language models to output robot actions,
inheriting their size and latency, and the resulting efficiency literature
already has surveys of its own (the CAS systematic survey, the Yu et al.
survey). We divide that literature by the resource each method spends.

1. **When the policy runs.** One decision is executed over several control
   steps, a lineage running from frame skip in reinforcement learning to
   action chunking (DQN, ACT, Diffusion Policy).
2. **What it is shown.** Visual tokens are reduced, reweighted, or reused.
3. **How much network each call uses.** Only part of the decoder runs.

A claim about efficiency is a claim about one of these resources being spent
differently, so we treat them as axes rather than competing methods, and each
of the six families below acts on one of them.

### Generic shortcuts, used as negative controls

The simplest way to spend less is to degrade the input or to skip feedback
unconditionally. Both fail in ways that motivate the safeguards the candidates
carry. On the spatial axis, foveation descends from Schwartz's log-polar
mapping, taken up in robot vision to cut data while preserving central
resolution (Traver and Bernardino) and applied to VLAs by gaze-conditioned
policies (Gaze-Reg, Look Focus Act). Every encoder we run splits the image
into a uniform grid, so at a fixed output resolution a pixel-space edit leaves
the visual token count and the model computation unchanged. Foveation before
the encoder therefore tests whether the policy survives losing its periphery,
not whether it runs faster. Methods that foveate inside the encoder do shed
tokens, but they give up pretrained weights fitted to a uniform grid (Look
Focus Act). On the temporal axis, executing one action over several control
steps reduces model calls in proportion. But it acts open loop through
contacts. FlashVLA reports a 0.7-point average success decrease with
training-free action reuse, and SpecPrune-VLA separates coarse movement from
precision-sensitive phases for the same reason. We run fixed foveation and
fixed repeat as controls that establish when the candidate methods need gates.

### The recent baseline

VLA-Cache reuses the key-value computation of visually static patches while
recomputing dynamic and task-relevant ones, the latter selected by
language-model attention, so history is retained unlike under token dropping.
On base OpenVLA across LIBERO it reports average success of 74.7% against
75.0% dense at 31.8 ms against 51.9 ms, a speed result with nearly maintained
rather than improved success. That makes it the anchor against which we
measure candidates, not a first-party contribution. It also explains a pattern
the token-pruning literature reports. Methods developed for vision-language
models (FastV, SparseVLM, ToMe) transfer poorly to VLAs, which VLA-Cache
attributes to VLAs' short action sequences and VLA-Pruner reproduces on
another setting. Whatever is removed from the visual stream must be removed by
a signal that knows what the robot is doing.

### Conditional candidates

Three families remove computation only where a signal says it is safe.

**Depth.** Decoder-layer redundancy is well established in language models.
But the recipes built on it disagree on which layers may go. ShortGPT ranks
every layer by Block Influence, the criterion we adopt, and constrains nothing
further, while others keep the final layer and remove a contiguous deep block
(Gromov et al.). EfficientVLA carries the unconstrained form into VLAs without
training, and MoLe-VLA needs a learned router and distillation, which places
it outside a training-free study. Several recent compact VLAs build such
reductions in by design rather than applying them at inference (FLOWER,
SmolVLA, TurboVLA), so whether removing layers helps depends on what the
architecture already leaves out. We add protected regions and a non-adjacency
rule to the ShortGPT score and calibrate on a disjoint split.

**Guarded reuse.** Where fixed repeat skips feedback blindly, our guarded
reuse skips a model call only when the current image and the recent
trajectory are both stable, and falls back to dense inference the moment
either gate fails. Recent work gates reuse on action similarity and
visual-token stability (FlashVLA) or on the manipulation phase (SpecPrune-VLA).
Ours differs in where the gate sits and how far it may go. It reads subsampled
pixels before any network call, at whole-frame and local scale, requires the
last two dense actions to agree in direction and gripper state, and permits
one reused step before the next dense call.

**Temporal fusion with a shared cache mask.** TTF-VLA fuses visual tokens
across frames without training and raises base OpenVLA on LIBERO by four
points at under two percent overhead, a denoising result rather than a speed
one. VLA-InfoEntropy selects which tokens to reuse by image and attention
entropy, and VLA-IAP prunes tokens by interaction alignment. Each selects
patches for one purpose with its own signal. We build one mask from motion,
entropy and task relevance that drives both the denoising and the cache path,
so the cache reuses only what the fusion also treats as stable, add
contact-aware fallbacks, and test it against optimized dense inference on
paired episodes.

### How these claims are evaluated

The field reports results on SimplerEnv and LIBERO, and recent work addresses
the infrastructure around them. The vla-eval harness unifies fourteen
benchmarks and documents evaluation pitfalls earlier work had left unrecorded,
and StarVLA describes the field as fragmented across incompatible codebases.
But infrastructure cannot supply the comparison itself. Papers that use
several backbones also change the benchmark (VLA-Cache, SpecPrune-VLA,
VLA-IAP), and those with both factors present do not report the crossing cell
(Gaze-Reg, VLA-Pruner), so the effect of a backbone cannot be separated from
the effect of a benchmark. The tables we cite report mean success rates, which
cannot say on which episodes an intervention helped, and speedups against
dense baselines that differ in attention backend, which cannot be compared
across papers. We hold both fixed. We exclude quantisation,
which changes none of the three resources, and learned early exit, which
needs training. Isolating the effect of choices that appear only in source
code is an established practice (Bag of Tricks for CNNs, Bag of Tricks for
LLMs). We evaluate the six families under one protocol across six backbones
and two benchmarks, on matched episodes against optimized dense inference, so
that a candidate is called positive only when its speed and its success both
clear a preregistered gate.

---

## Notes for the co-authors

**What changed from the three-method version (2026-09-03).**

- Order now matches III.B: controls, then the VLA-Cache baseline, then the
  three candidates. VLA-Cache moved ahead of the candidates because the
  temporal-fusion cache path is VLA-Cache-compatible and reads better once
  VLA-Cache has been introduced.
- Six families are covered, each with its lineage and its role. Foveation and
  unconditional repeat are named as controls in the text, VLA-Cache as the
  baseline, depth / guarded reuse / temporal fusion as candidates.
- Four claims were corrected against `docs/literature_review.md`:
  VLA-IAP is token pruning, not cache reuse (was lumped with VLA-InfoEntropy);
  SpecPrune-VLA "distinguishes" phases, it does not "avoid the decrease";
  VLA-Pruner "reproduces" the poor transfer, it does not "attribute" it;
  "Many also compare against an eager baseline" was an unverified count and
  is now a statement about measurement, not about papers.
- Numbers added, all from the mentor's audit: VLA-Cache 74.7 vs 75.0, 31.8 vs
  51.9 ms; FlashVLA 0.7-point decrease; TTF-VLA four points at under 2%.
- Dropped from v1: the "KV-cache compression" exclusion (VLA-Cache IS KV
  reuse, so that sentence would now contradict the section) and the
  OpenVLA-OFT citation in the chunking lineage (covered by Preliminaries).

**What the mentor's CSVs changed (2026-09-03).** Two words. "why the
candidate methods need gates" became "when": fixed repeat 2 collapses four of
six backbones on WidowX (CronusVLA 34 to 3.5, CogACT 50 to 12) but harms none
of four on Fractal, so the gate is a contact-task requirement and Results
carries the split. "across six backbones and two benchmarks" was added to the
closing sentence, because the paragraph faults others for the empty crossing
cell and four of our backbones fill it. The compact-VLA sentence is no longer
a cut candidate: MiniVLA drops to 0% at two removed blocks, CronusVLA to 0% at
four, SpatialVLA to 14%.

**Review-response edits (2026-09-03).** A simulated review raised four
points. Three changed text. (1) The "not new" admission on the shared mask is
gone; the design is stated positively without a "first" claim, which
literature_review.md bars, and without a collapse-when-combined claim, for
which there is no experiment. (2) Guarded reuse now names its differences
from FlashVLA: pixel gates before any network call, whole-frame and local
scale, two-action agreement in direction and gripper, one reused step.
FlashVLA's gate is token-aware, so the review's suggested "broad visual
stability" was not used. (3) "leave the crossing cell empty" and "overstates
the gain" were replaced by statements about what the tables cannot show,
with no verdict on any paper. (4) The review asked for "four LIBERO suites"
in place of "six backbones and two benchmarks"; that reading came from
experiment_protocol.md, but setup.tex and the CSVs are SimplerEnv WidowX and
Fractal, so the sentence stays. What does need fixing is setup.tex, which
still describes the three-backbone grid.

**Pending the mentor.** "calibrate on a disjoint split": the harness
calibrates on the same tasks and seed as the test run. Confirm or reword.
The VLA-Cache paragraph stays until the rollout question is answered.

**Cut candidates for the polish to 0.75 page, cheapest first.**

1. "It also explains a pattern the token-pruning literature reports." plus
   the FastV / SparseVLM / ToMe sentence, about 40 words. Cut if the paper
   does not run any VLM token-pruning method.
2. The vla-eval / StarVLA infrastructure sentence, about 35 words.
3. The quantisation / early-exit exclusion sentence, about 20 words.

**Citations are complete.** `flashvla`, `ttfvla` and `vlainfoentropy` were
added to `main.bib` on 2026-09-03 from the PDFs. Reading them changed two
sentences, so the earlier second-hand versions were not accurate enough to
ship.

**The anonymous OpenReview submission (R6d86jMO74) is not cited.** It is
non-archival, still under review, and has no public version, so a reviewer
cannot verify it and IEEE has no reference format for it. Dropping it costs
nothing, because TTF-VLA and VLA-InfoEntropy already establish that a reuse
mask is not novel on its own. The concurrent-work clause stays in prose
without a reference. Restore the citation only if the paper appears publicly
before camera-ready.
