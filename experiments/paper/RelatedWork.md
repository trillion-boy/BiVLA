# Related Work

*Reading draft of `relatedwork.tex`, citations spelled out, same content.
661 words of prose. *Bag of Tricks*' Related Work is 350 on the same basis,
so ours is 1.89x its length. (This file previously said 429, which was a
miscount, see `RelatedWork_Plan.md` §0.)
Provenance for every claim: `RelatedWork_Sources.md`.*

**Regenerate this file whenever `relatedwork.tex` changes.** It has fallen
behind three times, once carrying a sentence whose meaning had been inverted in
the `.tex` and fixed there but not here. `check_reading_copies.py` now tests
for that.

---

We cover three strands, namely the cost of VLA inference, the training-free
interventions we re-measure, and how such interventions are evaluated.

### Inference cost in VLA policies

VLA policies adapt pretrained vision-language models to output robot actions,
inheriting their size and latency, and the resulting efficiency literature
already has surveys of its own. We divide it by the resource each method
spends, which recovers a recent survey's categories of dynamic computation,
perception and action generation (the CAS systematic survey).

1. **When the policy runs.** A predicted chunk is executed over several control
   steps, a lineage running from frame skip in RL to parallel decoding (DQN,
   ACT, Diffusion Policy, OpenVLA-OFT).
2. **What it is shown.** Visual tokens are reduced or reweighted.
3. **How much network each call uses.** Decoder layers are skipped.

We treat these as axes rather than competing methods, because a claim about
efficiency is a claim about one of these resources being spent differently.

### The interventions we re-measure

Layer redundancy in language models is well established, **but its recipes
disagree on which layers may go.** ShortGPT ranks every layer by Block
Influence, the criterion we adopt, and constrains nothing further, while Gromov
et al. remove a contiguous block of the deepest layers and find the final layer
must be kept. EfficientVLA carries the unconstrained form into VLAs without
training, and MoLe-VLA needs a learned router and distillation, placing it
outside what we test. **These rules have not been compared on a robot policy,
and our depth results turn on which one holds.** Notably, several recent compact
VLAs **build such reductions in by design** rather than applying them at
inference (FLOWER, SmolVLA, TurboVLA), which makes whether a reduction transfers
a question about architecture and not only about a switch.

On the visual axis, methods developed for VLMs (FastV, SparseVLM, ToMe) have
been reported to transfer poorly to VLAs. VLA-Cache attributes this to their
short action sequences and measures FastV leaving FLOPs unchanged while latency
rises, and VLA-Pruner reproduces the degradation on a third setting.

Our own visual intervention, foveation, comes from the other direction. It
descends from Schwartz's log-polar mapping, taken up in robot vision to cut
data while preserving central resolution (Traver & Bernardino) and applied to
VLAs by gaze-conditioned policies (Gaze-Reg, Look-Focus-Act). We include it
because it edits the observation *before* the encoder, which is what makes it
comparable across backbones. Methods that foveate *inside* the encoder give distant
patches a coarser resolution and so shed tokens, but they give up the
pretrained weights, which were fitted to a uniform patch grid
(Look-Focus-Act). A training-free study therefore edits pixels and keeps the
token count its backbone already has. **No matter how the encoders differ**, each splits
the image into a **uniform** grid, so empty background gets the same budget as
the region where the gripper meets the object.

### How these claims are evaluated

Results are reported on SimplerEnv and LIBERO, and recent work addresses the
infrastructure around them. The vla-eval harness unifies fourteen benchmarks
and documents previously undocumented evaluation pitfalls, and StarVLA
describes the field as fragmented across incompatible codebases and protocols.

What infrastructure cannot supply is **the comparison itself.** Papers using
several backbones change the benchmark at the same time (VLA-Cache,
SpecPrune-VLA, VLA-IAP), those with both axes present leave the crossing cell
empty (Gaze-Reg, VLA-Pruner), and the tables we cite report mean success rates
over independent runs, without matched-episode outcomes. Whether an
intervention keeps its *direction* when the backbone or the benchmark changes
therefore cannot be read off these tables, nor can the per-task disagreements
they average over.

---

We do not cover quantisation, KV-cache compression or learned early exit, which
spend resources our axes do not. Measuring the settings papers leave to their
source code is an established form (the image-classification and the
inference-time-computation *bag of tricks* studies). Unique to this paper, we
measure the same interventions on both axes of a backbone × benchmark grid, and
we test each on matched episodes rather than aggregate rates. The direction of
an effect, and not only its size, then becomes something the evidence can decide.

---

## Notes for the co-authors (not part of the section)

**Structure copied from *Bag of Tricks*, measured:** one roadmap sentence,
three bold run-in paragraphs, each ending on our position rather than on a
summary, and a closing "Unique to this paper" sentence. No tables — theirs is
in the Introduction, and so is ours.

**Punctuation.** Both `.tex` files use no em-dashes, no semicolons and no
prose colons. That was measured against the 23 papers we hold: prose semicolons
run at a median of **1.0 per 1000 words** and **11 of 23 use none**, so zero is
normal. Colons are a different story — median **5.2 per 1000**, and only 2 of
23 use none — so dropping them is a house choice rather than a convention, and
restoring three or four would be unremarkable. (First count was wrong: it
included author-year citation separators like `Ma et al., 2024; Firoozi et al.,
2025`, which inflated semicolons to a median of 8.3. IEEEtran numeric style has
none of those.)

**Length — 661 words of prose, which is 1.89x the model paper's 350.** That
ratio is larger than this file used to claim, because the 429 it compared
against was a miscount (`RelatedWork_Plan.md` §0). The 2026-08-22 clause on
foveated tokenization added 45 of those words. The page budget still rests
on an *unverified* rule of thumb (≈500–550 words per IEEEtran column), so
**compile one real column before cutting anything.** Only if that overruns does
the following apply.

Paragraph 2 is 260 words against *Bag of Tricks*' 80 for the equivalent
paragraph, because it carries four topics: the layer-rule disagreement, the
VLM→VLA transfer failure, foveation's lineage and rationale, and the compact
VLAs. Everything that could be compressed without losing an argument already
has been. (Earlier notes here said 586, then 589, then 614. The first two were
miscounts; 614 counts citation markers as tokens and 616 is the prose count
used for the comparison above. `check_reading_copies.py` asserts the header
figure so it cannot drift again.)

If a real compile says it must shrink, the two candidates and their costs:

| cut | saves | what it costs |
|---|---|---|
| the compact-VLA clause | ~25 | the only in-paper motivation for the five-model expansion the mentor is running |
| the Schwartz/Traver lineage | ~24 | the "why this axis" that was specifically asked for; the uniform-grid argument survives, the provenance does not |

Neither is a free cut. Prefer taking the overrun out of a section with slack.

**Three things deliberately left out, and one that changed.**

1. **The per-task split** (`RelatedWork.md` §2.5 of the long version) gets one
   clause, not a paragraph. It is a finding, its evidence is in
   `PerTaskRows.md`, and its statement is in Results. (Table I was dropped on
   2026-08-22, see `tableI.tex`.)
2. ~~**StarVLA** is not cited — we have a search snippet, not the PDF.~~
   **Now cited.** The PDF was read 2026-08-22; the snippet we had been
   carrying turned out not to appear in the paper, so the caution was right.
3. **The ten training-free papers** in `Survey_2026-08.md` §4 are not cited.
   Abstract-level only.
4. **The grid's size** belongs in **Setup**, not here. Related Work claims a
   *method* — crossing both axes plus matched episodes — not a coverage
   record, because vla-eval already publishes 14 benchmarks × 6 model servers.
   The closing sentence said "a **complete** backbone × benchmark grid" until
   2026-08-22, which contradicted the Introduction's "five of the six cells"
   and would have broken outright on the expanded grid, since depth pruning
   applies to two of the five new models. It now claims the crossing.

**Two sentences a reviewer will test.** The matched-episode claim is scoped to
"the tables we cite," not to the field — keep it that way. And the
crossing-cell claim was already too strong once, which is why
`Survey_2026-08.md` exists; re-run that sweep close to submission.
