# Related Work

*Reading draft of `relatedwork.tex`, citations spelled out, same content.
1453 words of prose. This is the LONG six-family draft (v3, 2026-09-03). The
final target is 0.75 page, about 800 words in ieeeconf, so roughly 390 words
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

Most VLA policies adapt pretrained vision-language models to output robot
actions, inheriting their size and latency, and the resulting efficiency
literature already has surveys of its own (the CAS systematic survey, the Yu et
al. survey). We divide that literature by the resource each method spends.

1. **When the policy runs.** One decision is executed over several environment
steps, a lineage running from the frame skip of Atari agents (DQN) to action
chunking (ACT, Diffusion Policy, OpenVLA-OFT). 2. **What it is shown.** The
image is degraded before the encoder, or its visual tokens are reduced or
reused. 3. **How much of the decoder each call runs.** Layers are removed.

Each efficiency claim we consider is a claim about one of these resources being
spent differently, so we treat them as axes rather than competing methods, and
each of the six families below acts on one of them.

### Generic shortcuts, used as controls

The simplest shortcuts degrade the input or skip feedback unconditionally. Both
remove detail or feedback without a signal, which is the case the candidates'
safeguards exist to avoid. On the second axis, what the policy is shown,
foveation keeps resolution at the center and sheds it outward, a pattern
Schwartz modeled as a log-polar mapping, taken up in robot vision to cut data
while preserving central resolution (Traver and Bernardino) and brought into
robot learning as a gaze-centered foveated tokenizer in a ViT policy (Look
Focus Act). Gaze also enters a VLA with the input left unchanged, as a
training-time attention regularizer (Gaze-Reg). Every encoder we run splits the
image into a uniform grid, so at a fixed output resolution an edit in pixel
space leaves the visual token count and the model computation unchanged.
Foveation before the encoder therefore tests whether the policy survives losing
its periphery, not whether it runs faster. Methods that foveate inside the
encoder do shed tokens, but they give up pretrained weights fitted to a uniform
grid (Look Focus Act). On the first axis, when the policy runs, executing one
action over several environment steps reduces model calls per step in
proportion. But it acts open loop through contacts. FlashVLA gates its action
reuse, and its LIBERO-Spatial ablation shows the gated reuse alone lowering
success at every reduced token budget, and SpecPrune-VLA, which prunes tokens
rather than calls, keeps more of them in the contact phases where failures
under pruning cluster. We run fixed foveation and action repeat as controls.
Action repeat is guarded reuse with every gate removed and the cap alone
setting the interval, and that pair establishes what the reuse gates buy.
Foveation degrades the input with no signal, and establishes whether perceptual
simplification alone costs success.

### The recent baseline

VLA-Cache reuses the cached key-value entries of visually static patches and
recomputes those that move or that the decoder's attention marks task-relevant.
On OpenVLA's LIBERO checkpoints it reports average success of 74.7% against
75.0% dense at 39% lower CUDA latency, a speed result at nearly unchanged
success. It is the training-free VLA acceleration method that later work adopts
as its baseline (VLA-Pruner, EfficientVLA, SpecPrune-VLA), not a contribution
of ours. The same paper also reports that two token pruning methods developed
for vision-language models (FastV, SparseVLM) transfer poorly to VLAs. It
attributes the success loss to their working within a single frame, which
disrupts spatial fidelity, and the missing speedup to their targeting long
output sequences, whereas a VLA emits a few action tokens. VLA-Pruner
reproduces the loss on the same OpenVLA setting and attributes it instead to a
mismatch between the attention patterns of the prefill and the action-decode
stage. Both accounts fault the selection signal, one for reading a single frame
and one for reading the prefill alone.

### Conditional candidates

Three families act only where a signal says it is safe, once at calibration for
depth and at every step for the other two.

**Depth.** Redundancy among decoder layers is well established in language
models. But the recipes built on it disagree on which layers can go. ShortGPT
ranks every layer by Block Influence, the criterion we adopt, and constrains
nothing further, while others remove one contiguous block chosen by the
similarity between its input and output and offer finetuning as an optional
step to heal the cut (Gromov et al.). EfficientVLA applies the same
unconstrained ranking to a VLA without training, under the name non-contiguous
pruning, and MoLe-VLA trains a layer router end to end with self-distillation,
which places it outside a training-free study. Several recent compact VLAs
build the reduction in by design, keeping only part of the language model's
layers (FLOWER, SmolVLA) or dropping the large language model from the action
path altogether (TurboVLA), so whether removing layers at inference helps
depends on what the architecture already leaves out. We add protected regions
and a ban on adjacent removals to the ShortGPT selector, and choose the removed
set on a disjoint split with no weight update.

**Guarded reuse.** Where action repeat skips feedback blindly, our guarded
reuse skips a model call only when the current image is stable and the recent
actions agree and still command translation, and falls back to dense inference
the moment any gate fails. Recent work gates reuse on action similarity and
visual token stability (FlashVLA), and SpecPrune-VLA sets its pruning budget by
the end-effector speed read from the actions it has already emitted, the
translational part of which our floor also reads. Against FlashVLA, ours
differs in the signal. FlashVLA compares the visual token sets its previous two
calls selected, whereas ours reads subsampled pixels of the current frame,
which FlashVLA's gate never sees, at whole-frame and local scale, and adds a
gripper-state check and a translation floor. Both read the angle between their
two preceding actions and both cap consecutive reuse.

**Temporal fusion.** TTF-VLA fuses visual tokens across frames without
training. It keeps the current token for patches flagged by grayscale pixel
difference or attention relevance, reuses the previous token elsewhere, and
anchors a keyframe to bound drift. It reports that this raises average success
on OpenVLA's task-finetuned LIBERO checkpoints by four points at under two
percent overhead, a denoising result rather than a speed one. VLA-InfoEntropy
selects tokens for VLA-Cache's key-value reuse by image and attention entropy,
and VLA-IAP prunes tokens by interaction alignment. Our fusion shares TTF-VLA's
hard fusion, keyframe and two signals, pixel motion and text-to-vision
attention, and uses the second in our task-aware setting only. It adds an
image-entropy term and a protective ring around every flagged patch, and caps
the reusable fraction, which TTF-VLA fixes outright in its VLA-Cache setting
and leaves to the mask on OpenVLA. We test it on every backbone we evaluate
against dense inference under fused attention on paired episodes.

### How these claims are evaluated

The field reports results on SimplerEnv and LIBERO, and recent work addresses
the infrastructure around them. The vla-eval harness unifies fourteen
benchmarks and documents evaluation pitfalls earlier work had left unrecorded,
and StarVLA describes the field as fragmented across incompatible codebases.
But infrastructure cannot supply the comparison itself. Among the papers we
cite that test an intervention, those that use several backbones run at least
one of them on a benchmark the others do not see (VLA-Cache, SpecPrune-VLA),
and where one backbone does appear on two simulation benchmarks the others
appear on one (VLA-Pruner, Gaze-Reg, VLA-IAP), so no cited grid shows whether
an intervention's response to a change of benchmark holds across backbones. The
tables we cite report mean success rates, which cannot say on which episodes an
intervention helped. A speedup also depends on the dense baseline it is
measured against, and an eager attention baseline inflates it relative to a
fused one. We remove both, by pairing episodes and by measuring every speedup
against fused attention. We exclude quantization, which lowers numerical
precision rather than any of the three resources above, and learned early exit,
which trains the exits into the policy (DeeR-VLA). Evaluating a set of tricks
under one protocol, rather than one per paper, is an established practice (Bag
of Tricks for CNNs, Bag of Tricks for LLMs). We evaluate the six families under
one protocol on three evaluation environments, the WidowX Bridge and Google
Robot (Fractal) environments of SimplerEnv and LIBERO with its four suites, and
run each backbone on every environment for which a checkpoint at the size we
evaluate is released, as listed in Section IV-A. Every comparison is on matched
episodes against dense inference under fused attention. A candidate is called
positive only when it lowers end-to-end latency, with the cost of its own
signals included, or raises success, while the other stays within a
preregistered margin of dense.

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

**Eighth pass (2026-09-04), VLA-Cache and VLA-Pruner PDFs.** Three
corrections. "History is retained unlike under token dropping" is not in
VLA-Cache and was removed. VLA-Cache attributes the poor transfer of VLM
pruning to ignoring the temporal structure of closed-loop control, not to
short action sequences; VLA-Pruner reproduces it on the same OpenVLA setting
and blames a prefill versus action-decode attention mismatch. VLA-Pruner
does run OpenVLA on both LIBERO and SIMPLER, so the crossing-cell sentence
now says that where one backbone appears on two benchmarks the others
appear on one. Every citation with a concrete claim has now been read from
its PDF (sources rows 27 to 45).

**Handoff to the mentor's Setup and Protocol (refreshed 2026-09-04).** The
mentor writes those two sections. RW commits them to: six families (two
controls, one baseline, three candidates), five in Methods and Table I until
the VLA-Cache rollout is decided; three evaluation settings, WidowX Bridge,
Google Robot (Fractal) and LIBERO with four suites, with every backbone run
on each setting that releases a checkpoint at the evaluated size; matched
episodes paired by seed, with a paired test once the per-episode files
arrive; a dense baseline under fused attention, backend per backbone to be
confirmed; a candidate is positive when it lowers end-to-end latency with its
own signal cost included or raises success, while the other stays within a
preregistered margin of dense; depth calibration on a disjoint split; and,
from the mentor's own description, one GPU and one run per configuration. If
any of these is not what Setup says, RW changes to match, not the other way
round.

**Pending the mentor.** "calibrate on a disjoint split": the harness
calibrates on the same tasks and seed as the test run. Confirm or reword.
The VLA-Cache paragraph stays until the rollout question is answered.

**Cut candidates for the polish to 0.75 page, cheapest first.**

1. The FastV / SparseVLM sentence and the VLA-Pruner sentence in the
   baseline paragraph, about 75 words, with the closing "Whatever is dropped
   from the token stream" sentence that depends on them. Cut if the paper
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

### Ninth pass, 2026-09-04 (adversarial logic read after all citations were verified)

Eight sentences changed, none of them a citation claim. "Both fail" is gone
because the CSVs contradict it (repeat 2 harms no backbone on Fractal,
foveation helps three on WidowX); the controls are now defined as a
candidate's axis without its signal. "remove computation" became "act"
because temporal fusion removes none. "permits one reused step" became "caps
consecutive reused steps" because the aggressive preset caps at two. "keeps
TTF-VLA's" became "shares TTF-VLA's". "six backbones" became "every backbone
in our grid". "three quantities" became "three resources". The bag-of-tricks
sentence now describes what those papers did (one protocol over many tricks)
rather than what they studied. The closing gate is now "lowers latency or
raises success, while the other stays within a preregistered margin", so an
accuracy-only candidate can be positive. Setup must state the same gate.

### Tenth pass, 2026-09-04 (88-agent audit against the PDFs, the CSVs and the tricks-branch code)

Fourteen body sentences changed. The ones that were wrong, not merely
attackable: Look, Focus, Act is not a VLA; VLA-Cache and TTF-VLA run
OpenVLA's LIBERO-finetuned checkpoints, not "base OpenVLA"; VLA-Cache
compared FastV and SparseVLM only, never ToMe, and gives two reasons
(single-frame pruning breaks spatial fidelity, and VLAs emit few output
tokens); VLA-IAP puts three backbones on LIBERO, so it belongs to the other
crossing-cell group. The attackable ones: the FlashVLA "success decrease" is
now its reuse-alone ablation; SpecPrune-VLA prunes tokens, it does not skip
calls; two of our three "differences" from FlashVLA were shared and are now
stated as shared; TTF-VLA already uses attention relevance, so our fusion
"shares ... two signals" and adds entropy, the ring and the cap; Gromov et
al. heal by finetuning; TurboVLA keeps a text encoder; the CNN Bag of
Tricks is about training; "negative controls" is now "controls"; the
benchmark sentence counts consistently. Full report in the session
scratchpad, audit_report.md.

### Eleventh pass, 2026-09-04 (recheck of the tenth-pass sentences)

Nine body sentences changed. Gaze-Reg is no longer filed under foveation
(it argues against foveated input); the FlashVLA ablation is named as
LIBERO-Spatial and "every reduced token budget"; VLA-Cache's two reasons
go to their two outcomes (success loss, missing speedup); "CUDA latency"
replaces "GPU latency per call"; the fusion cap is credited to TTF-VLA's
VLA-Cache setting; the controls sentence pairs fixed repeat with the reuse
gates and gives foveation its own purpose; guarded reuse requires the arm
to be moving; the crossing-cell sentence no longer claims every
multi-backbone paper changes the benchmark; "Google Robot (Fractal)" and
"three evaluation settings". The handoff note above was refreshed.

### Twelfth pass, 2026-09-05 (final audit, one agent per cited paper plus claim ledgers and three reviewer personas)

Twenty-two body sentences changed; see the relatedwork.tex header for the
reason behind each. The ones that mattered most: Gromov et al. described by
their principal algorithm rather than their ablation heuristic; VLA-Cache
called the recent baseline rather than "closest point of comparison" (each
candidate has a closer relative); the crossing-cell conclusion narrowed to
what the cited grids actually leave unshown; "fixed repeat" renamed "action
repeat"; the guarded-reuse contrast with FlashVLA corrected (both decide
before the call, FlashVLA never sees the current frame); TTF-VLA's number
attributed; DeeR-VLA cited for learned early exit; axis glosses fixed so
foveation fits axis 2.
