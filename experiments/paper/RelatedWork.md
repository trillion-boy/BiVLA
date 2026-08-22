# Related Work

*Reading draft of `relatedwork.tex` — citations spelled out, same content.
590 rendered words in the LaTeX, against a 450–560 target and *Bag of Tricks*'
429. Provenance for every claim: `RelatedWork_Sources.md`.*

---

We cover three strands: the cost of VLA inference, the training-free
interventions we re-measure, and how such interventions are evaluated.

### Inference cost in VLA policies

VLA policies adapt pretrained vision-language models to output robot actions,
inheriting their size and latency; the resulting efficiency literature already
has surveys of its own. We divide it by the resource each method spends,
recovering the categories a recent systematic survey (CAS, arXiv 2510.17111)
draws between dynamic computation, perception and action generation:

1. **when the policy runs** — executing a predicted chunk over several control
   steps, from frame skip in RL through action chunking (ACT, Diffusion Policy)
   to OpenVLA-OFT's parallel decoding;
2. **what it is shown** — reducing or reweighting visual tokens;
3. **how much network each call uses** — skipping decoder layers.

We treat these as axes rather than competing methods: a claim about efficiency
is a claim about one of these resources being spent differently.

### The interventions we re-measure

Layer redundancy in language models is well established (ShortGPT, Gromov et
al.); the Block Influence criterion we adopt is ShortGPT's. EfficientVLA
carries it into VLAs without training, while MoLe-VLA pursues layer skipping
**with** a learned router and distillation, placing it outside what we test.

On the visual axis, methods developed for VLMs (FastV, SparseVLM, ToMe) have
been reported to transfer poorly to VLAs: VLA-Cache attributes this to VLAs
emitting short action sequences and measures FastV leaving FLOPs unchanged
while latency rises, and VLA-Pruner reproduces the degradation on a third
setting.

Our own visual intervention, foveation, comes from the other direction: the
log-polar mapping of Schwartz, taken up in robot vision to cut data while
preserving central resolution (Traver & Bernardino) and applied to VLAs by
gaze-conditioned policies (Gaze-Reg, Look-Focus-Act). We include it because
it edits the observation *before* the encoder, which is what makes it
comparable across backbones: however the encoders differ, each splits the image
into a **uniform** grid, so empty background receives the same budget as the
region where the gripper meets the object.

Notably, recent compact VLAs **build these reductions in** rather than applying
them at inference — FLOWER prunes 30–50% of its VLM layers by design, SmolVLA
keeps only the first sixteen, TurboVLA drops the language model from the action
path — which makes whether such a reduction transfers a
question about architecture, not only about a switch.

### How these claims are evaluated

Results are reported on SimplerEnv and LIBERO, and recent work addresses the
infrastructure: vla-eval unifies fourteen benchmarks and documents
previously undocumented evaluation pitfalls, and StarVLA describes the field as
fragmented across incompatible codebases and protocols.

What infrastructure cannot supply is **the comparison itself.** Papers that use
several backbones change the benchmark at the same time (VLA-Cache,
SpecPrune-VLA, VLA-IAP), and those with both axes present leave the crossing
cell empty (Gaze-Reg, VLA-Pruner); the tables we cite report mean success rates
over independent runs, without matched-episode outcomes. So whether an
intervention keeps its *direction* when the backbone or the benchmark changes
cannot be read off these tables, nor can the per-task disagreements they
average over (Table I).

---

We do not cover quantisation, KV-cache compression or learned early exit, which
spend resources our three axes do not. Measuring the settings that papers leave
to their source code is an established form (the image-classification and the
inference-time-computation *bag of tricks* studies). Unique to this paper, we
measure the same interventions over a complete backbone × benchmark grid and
test each result on matched episodes rather than on aggregate rates, so that
the direction of an effect — not only its size — is something the evidence can
decide.

---

## Notes for the co-authors (not part of the section)

**Structure copied from *Bag of Tricks*, measured:** one roadmap sentence,
three bold run-in paragraphs, each ending on our position rather than on a
summary, and a closing "Unique to this paper" sentence. No tables — theirs is
in the Introduction, and so is ours.

**Length.** 590 rendered words against the 450–560 target set in
`RelatedWork_Plan.md`. The target itself rests on an *unverified* rule of thumb
(≈500–550 words per IEEEtran column); compile one real column before treating
either number as binding. If it runs long, the first sentence to cut is the
Schwartz/Traver lineage clause in paragraph two — the foveation rationale
survives without it, since the load-bearing part is "edits the observation
before the encoder," not who invented log-polar mapping.

**Three things deliberately left out, and one that changed.**

1. **The per-task split** (`RelatedWork.md` §2.5 of the long version) gets one
   clause, not a paragraph. It is a finding; its evidence is Table I in the
   Introduction and its statement is in Results.
2. ~~**StarVLA** is not cited — we have a search snippet, not the PDF.~~
   **Now cited.** The PDF was read 2026-08-22; the snippet we had been
   carrying turned out not to appear in the paper, so the caution was right.
3. **The ten training-free papers** in `Survey_2026-08.md` §4 are not cited.
   Abstract-level only.
4. **The 8 × 3 scale statement** (eight backbones once the mentor's five runs
   land, three benchmarks) belongs in **Setup**, not here. Related Work claims
   a *method* — complete grid plus matched episodes — not a coverage record,
   because vla-eval already publishes 14 benchmarks × 6 model servers.

**Two sentences a reviewer will test.** The matched-episode claim is scoped to
"the tables we cite," not to the field — keep it that way. And the
crossing-cell claim was already too strong once, which is why
`Survey_2026-08.md` exists; re-run that sweep close to submission.
