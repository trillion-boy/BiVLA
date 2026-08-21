# Related Work

*Reading draft of `relatedwork.tex` — citations spelled out, same content.
483 words in the LaTeX, against *Bag of Tricks*' 429. Provenance for every
claim: `RelatedWork_Sources.md`.*

---

We cover three strands: efficient VLA inference, the training-free
interventions we re-measure, and how such interventions are evaluated.

### Efficient VLA inference

VLA policies adapt pretrained vision-language models to output robot actions,
inheriting their size and latency; surveys of the resulting efficiency
literature are now appearing. Work on reducing inference cost falls into three
families, by the resource each spends:

1. **when the policy runs** — executing a predicted chunk over several control
   steps, a lineage running from frame skip in RL through action chunking
   (ACT, Diffusion Policy) to the parallel decoding of OpenVLA-OFT;
2. **what it is shown** — reducing or reweighting visual tokens;
3. **how much network each call uses** — skipping decoder layers.

We treat these as three axes rather than as competing methods, because a claim
about efficiency is a claim about one of these resources being spent
differently.

### The interventions we re-measure

Layer redundancy in language models is well established, and the Block
Influence criterion we adopt is ShortGPT's. EfficientVLA carries it into VLAs
without training; MoLe-VLA pursues layer skipping **with** a learned router and
distillation, which places it outside what we test.

On the visual axis, methods developed for VLMs (FastV, SparseVLM, ToMe) have
been reported to transfer poorly to VLAs: VLA-Cache attributes this to VLAs
emitting short action sequences and measures FastV leaving FLOPs unchanged
while latency rises, and VLA-Pruner reproduces the degradation on a third
setting. Foveation, our own visual intervention, is a vision technique of the
same kind, tested for the same transfer.

Notably, recent compact VLAs no longer apply these reductions at inference but
**build them in** — FLOWER prunes 30–50% of its VLM layers by design, SmolVLA
retains only the first sixteen, and TurboVLA removes the language model from
the action path entirely — which makes whether such a reduction transfers a
question about architecture, not only about a switch.

### How these claims are evaluated

Efficiency results are reported on SimplerEnv and LIBERO, and recent work has
begun to address the surrounding infrastructure: vla-eval unifies fourteen
benchmarks behind one harness and documents previously undocumented evaluation
pitfalls.

What that infrastructure cannot supply is **the comparison itself.** Papers
that use several backbones change the benchmark at the same time (VLA-Cache,
SpecPrune-VLA, VLA-IAP), and those with both axes present leave the crossing
cell empty (Gaze-Reg, VLA-Pruner); all report mean success rates over
independent runs. Consequently, whether an intervention keeps its *direction*
when the backbone or the benchmark changes cannot be read off the published
tables, and neither can the per-task disagreements those tables average over
(Table I).

---

Unique to this paper, we measure the same interventions over a complete
backbone × benchmark grid and test each result on matched episodes rather than
on aggregate rates, so that the direction of an effect — not only its size —
is something the evidence can decide.

---

## Notes for the co-authors (not part of the section)

**Structure copied from *Bag of Tricks*, measured:** one roadmap sentence,
three bold run-in paragraphs, each ending on our position rather than on a
summary, and a closing "Unique to this paper" sentence. No tables — theirs is
in the Introduction, and so is ours.

**Three things deliberately left out.**

1. **The per-task split** (`RelatedWork.md` §2.5) gets one clause, not a
   paragraph. It is a finding; its evidence is Table I in the Introduction and
   its statement is in Results.
2. **StarVLA**, whose stated problem supports our motivation, is not cited —
   we have a search snippet, not the PDF.
3. **The ten training-free papers** in `Survey_2026-08.md` §4 are not cited.
   Abstract-level only.

**Two sentences a reviewer will test.** *"None pairs episodes"* is a claim
about the tables we cite, not about the field — keep it that way. And the
crossing-cell claim was already too strong once, which is why
`Survey_2026-08.md` exists; re-run that sweep close to submission.
