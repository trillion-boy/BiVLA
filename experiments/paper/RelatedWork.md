# Related Work

*Reading draft of `relatedwork.tex`, citations spelled out, same content.
1657 words of prose. This is the LONG draft (v3, 2026-09-03; five families since 2026-09-08). The
final target is 0.75 page, about 800 words in ieeeconf, so roughly 390 words
come out in polishing. Cut candidates are listed under "Notes for the
co-authors". Provenance for every claim: `RelatedWork_Sources.md`, the new
claims in its last section.*

**Regenerate this file whenever `relatedwork.tex` changes.** It has fallen
behind three times, once carrying a sentence whose meaning had been inverted in
the `.tex` and fixed there but not here. `check_reading_copies.py` now tests
for that.

We cover the cost of VLA inference, the five intervention families we evaluate, grouped by the role each plays in our argument, the published method the field measures against, and how the literature tests efficiency claims.

**Inference cost in VLA policies.** Most VLA policies adapt pretrained vision-language models to output robot actions and inherit the size and latency of those models, and the resulting efficiency literature already has surveys of its own [effvlasurvey1, effvlasurvey2]. We divide that literature by what each method changes at inference. 1) The first axis is *when the policy runs*. One decision is executed over several environment steps, a practice whose lineage runs from the frame skip of Atari agents [dqn] to action chunking [act, diffusionpolicy, openvlaoft]. 2) The second is *what the policy is shown*. The image is degraded before the encoder, or its visual tokens are reduced or reused. 3) The third is *how much of the decoder each call runs*. Layers are removed. Each method we consider changes one of these three things, so we treat the three as axes rather than as competing methods, and each of the five families below acts on one axis.

**Generic shortcuts, used as controls.** The simplest shortcuts degrade the input or skip feedback unconditionally, with no signal to say when. On the second axis, what the policy is shown, foveation keeps resolution at the center and sheds it outward. Schwartz modeled this pattern as a log-polar mapping [schwartz], robot vision took it up to cut data while preserving central resolution [traver], and robot learning built it into a gaze-centered foveated tokenizer for a Vision Transformer (ViT) policy [lookfocusact]. Gaze also enters a VLA with the input left unchanged, as a training-time attention regularizer [gazereg]. Every encoder we run splits the image into a uniform grid, so at a fixed image resolution an edit in pixel space leaves the visual token count and the model computation unchanged. Foveation before the encoder therefore tests whether the policy survives losing peripheral detail, not whether it runs faster. Methods that foveate inside the encoder do shed tokens, but they give up pretrained weights fitted to a uniform grid [lookfocusact]. On the first axis, when the policy runs, executing one action over several environment steps reduces model calls per step in proportion to the number of steps. But the repeated action is executed open-loop through contact. FlashVLA gates its action reuse, yet its LIBERO-Spatial ablation with pruned inference disabled shows the gated reuse lowering success below dense at every token budget it tests [flashvla]. SpecPrune-VLA, which prunes tokens rather than calls, keeps more tokens in the contact phases, where failures cluster under pruning [specprune]. We run foveation and action repeat as controls. Action repeat holds each action for a fixed number of steps and is the ungated limit of guarded reuse, so the pair shows what ungated skipping costs and how much skipping the gates admit. Foveation degrades the input with no signal and establishes what perceptual simplification alone does to success.

**The published point of comparison.** VLA-Cache reuses the cached key-value entries of visually static patches and recomputes those that move or that the decoder's attention marks as task-relevant [vlacache]. On OpenVLA's task-finetuned LIBERO checkpoints, it reports an average success of 74.7% against 75.0% dense at 39% lower CUDA latency, a speed result at nearly unchanged success. Later work on training-free VLA acceleration adopts it as a baseline [vlapruner, efficientvla, specprune]. We do not run it, and we cite it as the point the field measures against. The same paper reports that two visual token reduction methods developed for vision-language models [fastv, sparsevlm] transfer poorly to VLAs. It attributes their loss of success to working within a single frame and disrupting spatial fidelity, and their lack of speedup to targeting long output sequences, whereas a VLA emits only a few action tokens [vlacache]. VLA-Pruner reproduces the loss on the same OpenVLA setup and attributes it to a mismatch between the attention patterns of the vision-language prefill stage and the action-decode stage [vlapruner]. Both accounts fault how the tokens are chosen, one for working within a single frame and one for reading the prefill alone.

**Conditional candidates.** Three families act only where a signal says it is safe, once at calibration for depth pruning, at every step for guarded reuse, and at every call for temporal fusion. Each is a recipe assembled from published parts, and our contribution is the protocol under which they are compared. Fig. 1 places the three above the control loop and the two controls below it. *Depth pruning.* Decoder layers in language models are redundant [shortgpt, gromov], and the recipes built on that redundancy disagree on which layers to remove. ShortGPT ranks every layer by Block Influence, the criterion we adopt, and imposes no further constraint [shortgpt]. Gromov et al. instead remove one contiguous block, chosen by the similarity between its input and output, and offer finetuning as an optional step to heal the cut [gromov]. EfficientVLA applies the same unconstrained ranking, which it calls non-contiguous pruning, to a VLA without training [efficientvla], and MoLe-VLA trains a layer router end to end with self-distillation [molevla], so it falls outside a training-free study. Several recent compact VLAs build the reduction in, keeping only part of the language model's layers [flower, smolvla] or dropping the language model from the action path altogether [turbovla]. Whether removing layers at inference helps is therefore a question about what the architecture already leaves out, and we report the depth response of every backbone we evaluate. We add protected regions and a ban on adjacent removals to the ShortGPT selector and choose the removed set on calibration frames ahead of the test episodes, with no weight update. *Guarded reuse.* Where action repeat skips feedback blindly, our guarded reuse skips a model call only when the current image is stable, the recent actions agree, and the latest of them commands a translation above a floor. It falls back to dense inference the moment any gate fails. FlashVLA gates reuse on action similarity and visual token stability [flashvla], and SpecPrune-VLA sets its pruning budget by the end-effector speed that it infers from the actions it has already emitted [specprune], a signal whose translational part our translation floor also reads. Our guarded reuse differs from FlashVLA in the signal. FlashVLA compares the visual token sets that its previous two calls selected, whereas ours reads subsampled pixels of the current frame, which FlashVLA's gate never sees, at whole-frame and local scale, and adds a gripper-state check and a translation floor. Both cap consecutive reuse, and both read the angle between the two most recent actions they inferred, an angle ours computes over the pose dimensions alone. *Temporal fusion.* TTF-VLA fuses visual tokens across frames without training. It keeps the current token for patches flagged by grayscale pixel difference or attention relevance, reuses the previous token elsewhere, and inserts a dense keyframe to bound drift. It reports that this fusion raises average success on OpenVLA's task-finetuned LIBERO checkpoints by 4 points at under 2% overhead, a denoising result rather than a speed one [ttfvla]. VLA-InfoEntropy selects tokens for VLA-Cache's key-value reuse by image entropy and attention entropy [vlainfoentropy], and VLA-IAP prunes tokens by interaction alignment [vlaiap]. Our fusion shares TTF-VLA's hard fusion, its keyframe, and its two signals of pixel motion and text-to-vision attention, using the second in our task-aware setting only. It adds an image-entropy term of the kind VLA-InfoEntropy uses and a protective ring around every flagged patch, and it caps the reusable fraction, which TTF-VLA fixes per suite in its VLA-Cache experiments and leaves to the mask in its OpenVLA experiments. We test it against dense inference on matched episodes for every backbone we evaluate.

**How these claims are evaluated.** The field reports results on SimplerEnv [simplerenv] and LIBERO [libero], and recent work addresses the infrastructure around them. The vla-eval harness unifies 14 benchmarks and documents evaluation pitfalls that earlier work had left unrecorded [vlaeval], and StarVLA describes the field as fragmented across incompatible codebases [starvla]. But infrastructure cannot supply the comparison itself. Among the papers we cite that test an intervention on a robot policy, those that use several backbones either confine all of them to one simulation benchmark [molevla] or run at least one backbone on a benchmark the other backbones never see [vlacache, specprune]. Where one backbone does appear on two simulation benchmarks, the other backbones appear on one [vlapruner, gazereg, vlaiap]. No cited grid therefore shows whether an intervention's response to a change of benchmark holds across backbones. The tables we cite report mean success rates, which cannot say on which episodes an intervention helped. A speedup also depends on the dense baseline it is measured against, and an eager, unfused attention baseline inflates it relative to a fused one, as SpecPrune-VLA's own FlashAttention comparison shows [specprune]. We remove the first confound by pairing episodes and disclose the second by measuring every speedup against the dense implementation each backbone ships with, under the attention backend that Section IV-A records for it. We exclude quantization, which lowers numerical precision rather than changing any of the three things above, and learned early exit, which trains the exits into the policy [deervla]. Evaluating a set of tricks under one protocol, rather than under a different protocol per paper, is an established practice [bagoftricks_cnn, bagoftricks_llm]. We evaluate the five families under one protocol in three environments, namely the WidowX Bridge and Google Robot (Fractal) environments of SimplerEnv and LIBERO with its four suites. Each backbone runs in every environment for which a checkpoint at the evaluated size is released, as listed in Section IV-A. Every comparison is on matched episodes against dense inference with the implementation each backbone ships with. A candidate is called positive only when it lowers latency, with the cost of any run-time signal included, or significantly raises success under the paired test that Section IV-A records. A candidate that improves one measure at a significant cost to the other is not counted.

## Notes for the co-authors

**What changed from the three-method version (2026-09-03).**

- Order: controls, then the published point of comparison (VLA-Cache, cited
  and not run since 2026-09-08), then the three candidates.
- Five families are covered, each with its lineage and its role. Foveation and
  action repeat are named as controls in the text, depth / guarded reuse /
  temporal fusion as candidates.
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
fixable in prose (as of 2026-09-04; the family count was settled at five on
2026-09-08): the cache path of temporal fusion, "Both fail" versus repeat 2 on Fractal,
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
mentor writes those two sections. RW commits them to: five families (two
controls, three candidates; VLA-Cache cited, not run, per the mentor on
2026-09-08); three evaluation settings, WidowX Bridge,
Google Robot (Fractal) and LIBERO with four suites, with every backbone run
on each setting that releases a checkpoint at the evaluated size; matched
episodes paired by seed, with a paired test once the per-episode files
arrive; a dense baseline under the attention backend each harness resolves to, printed
per backbone with torch and transformers versions (explicit SDPA: OpenVLA,
UniVLA, SmolVLA; implicit SDPA under transformers 4.47.0: CogACT, CronusVLA,
MiniVLA; eager Gemma2 decoder: SpatialVLA; OpenVLA LIBERO to confirm); a candidate is positive when it lowers end-to-end latency with its
own signal cost included or significantly raises success under the paired
test, with no significant cost to the other measure; depth calibration on frames ahead of the test episodes; and,
from the mentor's own description, one GPU and one run per configuration. If
any of these is not what Setup says, RW changes to match, not the other way
round.

**Calibration wording (resolved internally, not asked of the mentor).** The
harness calibrates on the same tasks as the test run, with a different seed
where recorded (CronusVLA seed 10000; the other backbones record no
calibration source). Rather than assert a holdout we cannot verify, both RW
and Setup dropped "disjoint from the test episodes" and now read "on
calibration frames ahead of the test episodes." Whether to also confirm the
calibration source with the mentor is the user's open call.
The VLA-Cache paragraph stays as literature; the rollout question was answered on 2026-09-08 (skipped).

**Cut candidates for the polish to 0.75 page, cheapest first.**

1. The FastV / SparseVLM sentence and the VLA-Pruner sentence in the
   baseline paragraph, about 75 words, with the closing "Whatever is dropped
   from the token stream" sentence that depends on them. The paper runs no VLM token-pruning method, so this cut is available; it is
   kept for now because Section II's FastV sentence is what the Introduction
   leans on.
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

### Eighteenth pass, 2026-09-06

Body regenerated from `relatedwork.tex` after the six-reviewer attack. The changes are logged in the `.tex` header (eighteenth pass). Citations now appear as bracketed bib keys.

Body regenerated again on 2026-09-08 (twentieth pass): five families, VLA-Cache as the published point of comparison, and the three fused-attention sentences replaced by the per-backbone backend wording after the mentor's attention_backend.md. Details in the `.tex` header.
