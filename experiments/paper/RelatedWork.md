# Related Work

*Reading draft of `relatedwork.tex`, citations spelled out, same content.
1169 words of prose. This is the LONG six-family draft (v3, 2026-09-03). The
final target is 0.75 page, about 800 words in ieeeconf, so roughly 370 words
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
steps, a lineage running from frame skip in reinforcement learning to action
chunking (DQN, ACT, Diffusion Policy, OpenVLA-OFT). 2. **What it is shown.**
Visual tokens are reduced, reweighted, or reused. 3. **How much network each
call uses.** Only part of the decoder runs.

A claim about efficiency is a claim about one of these resources being spent
differently, so we treat them as axes rather than competing methods, and each
of the six families below acts on one of them.

### Generic shortcuts, used as negative controls

The simplest way to spend less is to degrade the input or to skip feedback
unconditionally. Both fail in ways that motivate the safeguards the candidates
carry. On the spatial axis, foveation descends from Schwartz's log-polar
mapping, taken up in robot vision to cut data while preserving central
resolution (Traver and Bernardino) and brought to VLAs through human gaze, as a
training-time regularizer (Gaze-Reg) or as a foveation signal (Look Focus Act).
Every encoder we run splits the image into a uniform grid, so at a fixed output
resolution an edit in pixel space leaves the visual token count and the model
computation unchanged. Foveation before the encoder therefore tests whether the
policy survives losing its periphery, not whether it runs faster. Methods that
foveate inside the encoder do shed tokens, but they give up pretrained weights
fitted to a uniform grid (Look Focus Act). On the temporal axis, executing one
action over several control steps reduces model calls in proportion. But it
acts open loop through contacts. FlashVLA gates its action reuse and still
reports a success decrease, and SpecPrune-VLA separates coarse movement from
phases that need precision for the same reason. We run fixed foveation and
fixed repeat as controls that establish when the candidate methods need gates.

### The recent baseline

VLA-Cache reuses the key-value computation of visually static patches while
recomputing patches that move or that the language model attends to, so history
is retained unlike under token dropping. On base OpenVLA across LIBERO it
reports average success of 74.7% against 75.0% dense at 39% lower latency, a
speed result at nearly unchanged success. It is the closest published point of
comparison for our candidates, not a contribution of ours. The same paper
accounts for a pattern reported for token pruning. Methods developed for
vision-language models (FastV, SparseVLM, ToMe) transfer poorly to VLAs, which
it attributes to VLAs' short action sequences and VLA-Pruner reproduces on
another setting. Whatever is dropped from the token stream must be chosen by a
signal that knows what the robot is doing.

### Conditional candidates

Three families remove computation only where a signal says it is safe.

**Depth.** Redundancy among decoder layers is well established in language
models. But the recipes built on it disagree on which layers may go. ShortGPT
ranks every layer by Block Influence, the criterion we adopt, and constrains
nothing further, while others keep the final layer and remove a contiguous deep
block (Gromov et al.). EfficientVLA carries the unconstrained form into VLAs
without training, and MoLe-VLA needs a learned router and distillation, which
places it outside a training-free study. Several recent compact VLAs build the
reduction in by design, keeping only part of the language model's layers
(FLOWER, SmolVLA) or dropping the language-model pathway altogether (TurboVLA),
so whether removing layers at inference helps depends on what the architecture
already leaves out. We add protected regions and a ban on adjacent removals to
the ShortGPT selector, and choose the removed set on a disjoint split with no
weight update.

**Guarded reuse.** Where fixed repeat skips feedback blindly, our guarded reuse
skips a model call only when the current image and the recent trajectory are
both stable, and falls back to dense inference the moment either gate fails.
Recent work gates reuse on action similarity and visual token stability
(FlashVLA), and SpecPrune-VLA conditions its token pruning on the manipulation
phase. Ours differs in where the gate sits and how far it may go. It reads
subsampled pixels before any network call, at frame and patch scale, requires
the last two dense actions to agree in direction and gripper state, and permits
one reused step before the next dense call.

**Temporal fusion.** TTF-VLA fuses visual tokens across frames without
training. It keeps the current token for patches flagged by grayscale pixel
difference or attention relevance, reuses the previous token elsewhere, and
anchors a keyframe to bound drift, which raises base OpenVLA on LIBERO by four
points at under two percent overhead, a denoising result rather than a speed
one. VLA-InfoEntropy selects tokens for VLA-Cache's key-value reuse by image
and attention entropy, and VLA-IAP prunes tokens by interaction alignment. Our
fusion keeps TTF-VLA's hard fusion and keyframe and changes the selection.
Patches are scored by motion, image entropy and task relevance, every flagged
patch protects a one-patch ring around it, and the reusable fraction is capped.
We test it on six backbones against optimized dense inference on paired
episodes.

### How these claims are evaluated

The field reports results on SimplerEnv and LIBERO, and recent work addresses
the infrastructure around them. The vla-eval harness unifies fourteen
benchmarks and documents evaluation pitfalls earlier work had left unrecorded,
and StarVLA describes the field as fragmented across incompatible codebases.
But infrastructure cannot supply the comparison itself. Papers that use several
backbones also change the benchmark (VLA-Cache, SpecPrune-VLA, VLA-IAP), and
those with both factors present do not report the crossing cell (Gaze-Reg,
VLA-Pruner), so the effect of a backbone cannot be separated from the effect of
a benchmark. The tables we cite report mean success rates, which cannot say on
which episodes an intervention helped. A speedup also depends on the dense
baseline it is measured against, and an eager attention baseline inflates it
relative to a fused one. We remove both, by pairing episodes and by measuring
every speedup against fused attention. We exclude quantization, which lowers
numerical precision rather than any of the three quantities above, and learned
early exit, which needs training. Isolating the effect of choices that appear
only in source code is an established practice (Bag of Tricks for CNNs, Bag of
Tricks for LLMs). We evaluate the six families under one protocol on three
benchmarks, the WidowX and Fractal suites of SimplerEnv and the four suites of
LIBERO, and run each backbone on every benchmark for which a checkpoint at the
size we evaluate is released, as listed in Section IV-A. Every comparison is on
matched episodes against optimized dense inference. A candidate is called
positive only when its end-to-end latency, with the cost of its own signals
included, and its success both clear a preregistered gate.
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

**Second review round (2026-09-03), four points, all acted on.**
(1) FlashVLA's 0.7-point decrease belongs to its gated reuse, so it was
evidence for "even gated reuse costs something", not for "unconditional
repeat fails". The number is gone and the sentence now says FlashVLA gates
its reuse and still reports a decrease; the failure of unconditional repeat
is our own control result. (2) Our grid is 10 of 12 cells, MiniVLA and
UniVLA have no Fractal checkpoint, so the closing sentence says "six
backbones, four of them on both benchmarks" before anyone counts. If Results
ends up with fewer filled cells, the "do not report the crossing cell" clause
must go. (3) "with no weight update" added to the depth calibration clause;
what is chosen is the removed set, not a threshold. (4) The gate is now on
end-to-end latency with the cost of the candidate's own signals included,
which experiment_protocol.md already requires and the CSVs already measure
(fusion task-aware costs OpenVLA 12 ms per call).

**Third review round (2026-09-03), four wording points, all acted on.**
"We hold both fixed" named what is fixed (paired episodes, fused attention
baseline), because "both" was reading as the nouns before it. Quantization
now "lowers numerical precision rather than any of the three quantities
above" instead of "changes none of the three resources". The closing
sentence gives the reason for the 10-cell grid, checkpoints exist for both
benchmarks on four backbones, instead of a bare "four of them". VLA-Cache is
"a speed result at nearly unchanged success"; the earlier "rather than
improved" came from literature_review.md but reads as a dig in the paper.

**Grid sentence is count-free (2026-09-03).** Table II adds LIBERO (four
suites) and SmolVLA, and OpenVLA on LIBERO is still running, so "six
backbones" and "two benchmarks" were dropped. The sentence names the three
benchmarks and the rule (every backbone on each benchmark with a released
checkpoint). Restore counts once the grid is frozen.

**Fifth pass, reviewer self-check (2026-09-04).** Three edits. VLA-Cache's
latency is given as a ratio, 39% lower, because Table II will carry our own
OpenVLA dense latency on LIBERO on different hardware. The token-pruning
paragraph closes on "dropped from the token stream", since foveation drops no
tokens and Table I shows it helping SpatialVLA. The grid sentence says "a
checkpoint at the size we evaluate": CronusVLA releases LIBERO checkpoints at
7B only and we run 0.5B, and SpatialVLA releases none. Still open and not
fixable in prose: six families here versus five in Methods and the tables,
the cache path of temporal fusion, "Both fail" versus repeat 2 on Fractal,
the disjoint split, and "preregistered".

**Shared cache mask removed (2026-09-04).** The temporal-fusion heading
is now "Temporal fusion" and the closing sentence no longer says the mask
drives a cache path. The CSVs show no latency change under temporal fusion
and no separate fusion-plus-cache condition, so the speed claim had nothing
in the tables behind it. Methods III.E says the same. Restore only if a
fusion-plus-cache condition arrives with its own rows.

**Sixth pass (2026-09-04), TTF-VLA read from the PDF.** TTF-VLA already
does per-patch hard fusion from pixel difference plus attention relevance,
with keyframe anchoring. Without the cache path our method is that recipe
with a different selector, so the paragraph now states the relationship and
lists only what we change (entropy term, one-patch ring, reuse cap, six
backbones). Also: VLA-Cache is "the closest published point of comparison",
not "the anchor against which we measure", since no rollout exists; and
OpenVLA-OFT joins the chunking lineage.

**Seventh pass (2026-09-04), five PDFs read.** ShortGPT, SmolVLA and
vla-eval match the text. Two did not. TurboVLA drops the LLM pathway rather
than removing layers, so the compact-VLA sentence now separates FLOWER and
SmolVLA (keep part of the layers) from TurboVLA (no language-model
pathway). Gaze-Reg regularizes attention at training time and uses no gaze
at inference, so "policies conditioned on gaze" became "as a training-time
regularizer (Gaze-Reg) or as a foveation signal (Look Focus Act)". Its
crossing-cell gap is confirmed: pi0 on LIBERO and ALOHA-Sim, OpenVLA on
LIBERO only. Sources rows 34 to 38.

**Handoff to the mentor's Setup and Protocol (2026-09-03).** The mentor
writes those two sections. RW commits them to: six families (two controls,
one baseline, three candidates); six backbones on two SimplerEnv suites,
WidowX and Fractal; matched episodes with a paired test; a dense baseline
under fused attention; a preregistered gate on both speed and success; depth
calibration on a disjoint split; and, from the mentor's own description, one
GPU and one run per configuration with episodes paired by seed. If any of
these is not what Setup says, RW changes to match, not the other way round.

**Pending the mentor.** "calibrate on a disjoint split": the harness
calibrates on the same tasks and seed as the test run. Confirm or reword.
The VLA-Cache paragraph stays until the rollout question is answered.

**Cut candidates for the polish to 0.75 page, cheapest first.**

1. "It also explains a pattern reported for token pruning." plus
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
