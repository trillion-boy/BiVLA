# Related Work — provenance audit

*Every factual claim in `relatedwork.tex`, with how it was established. The
rule this session settled on: a claim about what a paper says needs that
paper's PDF, not its abstract.*

---

## Verified from the PDF

| # | claim in the draft | where verified |
|---|---|---|
| 1 | the Block Influence criterion we adopt is ShortGPT's | `RelatedWork.md` A.3, made against the PDF |
| 2 | EfficientVLA carries layer pruning into VLAs without training | A.1, from the PDF — *"이 셋을 학습 없이 동시에 줄이는 것이 목표"* |
| 3 | MoLe-VLA uses a learned router and distillation, so it is outside training-free | read this session: §3.5 *Optimization Objective*, Eq. 18–19, Gumbel-Softmax router, CogKD |
| 4 | VLA-Cache attributes poor transfer to short action sequences, and measures FastV leaving FLOPs unchanged while latency rises | A.2, from the PDF: 1.864 T unchanged, 51.91 → 53.28 ms |
| 5 | VLA-Pruner reproduces the degradation on a third setting | read this session: its Table 2, FastV retains 73.1% of baseline on SIMPLER at 75% |
| 6 | FLOWER prunes 30–50% of its VLM layers by design | read this session: *"we prune between 30% and 50% of the pretrained VLM's layers"* |
| 7 | SmolVLA retains only the first sixteen | read this session: *"We use only the first 16 layers of the LLM within the VLM"* |
| 8 | TurboVLA removes the language model from the action path | read this session: *"avoiding … processing multimodal inputs through a billion-parameter language model"*, chunks emitted *"without autoregressive action-token generation"* |
| 9 | vla-eval unifies fourteen benchmarks and documents previously undocumented pitfalls | read this session: 14 simulation benchmarks, six model servers, *"documenting previously undocumented pitfalls"* |
| 10 | papers using several backbones change the benchmark at the same time | VLA-Cache (A.2: LIBERO→OpenVLA, SIMPLER→CogACT); SpecPrune-VLA (Table 1: LIBERO→OpenVLA-OFT, SimplerEnv→DB-OFT); VLA-IAP (Table 1 caption: DreamVLA, π0 LIBERO, π0.5 VLABench) |
| 11 | those with both axes leave the crossing cell empty | Gaze-Reg (A.5); VLA-Pruner (LIBERO has OpenVLA + OpenVLA-OFT, SIMPLER has OpenVLA only) |
| 12 | the tables we cite report mean success rates over independent runs, without matched-episode outcomes | grep over every PDF read: *paired* 0, *McNemar* 0, and in vla-eval also *significance* 0, *statistic* 0, *confidence* 0, *error bar* 0. The draft says **"the tables we cite,"** not "no prior work" — see the re-check note below |
| 13 | FastV, SparseVLM and ToMe were developed for VLMs | all three PDFs read this session |
| 14 | OpenVLA-OFT uses parallel decoding and action chunking | stated in the VLA-Pruner PDF describing its own baseline |
| 15 | foveation has been applied to VLAs by gaze-conditioned policies | `RelatedWork.md` A.5 (Gaze-Reg, arXiv 2603.23202) and A.6 (Look Focus Act, arXiv 2507.15833), both marked *원문 확인* — the two PDFs were opened and their tables cross-checked |
| 17 | log-polar was adopted in robot vision to cut data while preserving central resolution | **PDF read 2026-08-22**, Traver & Bernardino p3: *"Elegant trade-off solution between these three mutually opposing criteria: wide field of view, high visual resolution and little data to process … the reduced size of log-polar images (as much as 30 times smaller than uniformly-sampled Cartesian images have been reported) hugely facilitates real-time visual data processing. On the other hand, the radially logarithmic sampling entails that a higher resolution is devoted to the center of the scene (fovea area)"* — both halves of our clause, verbatim, with a number |
| 18 | layer redundancy in language models is well established (the `gromov` half) | **PDF read 2026-08-22**: *"we can remove a substantial fraction of the deepest layers from models with minimal degradation … for Llama-2-70B we can eliminate up to roughly half of the layers before the performance collapses."* **ICLR 2025**, not an arXiv preprint — fix the `.bib` |
| 19 | re-measuring methods one did not invent is an established form | **PDF read 2026-08-22**, Bag of Tricks (CNN) abstract: *"most refinements are either briefly mentioned as implementation details or only visible in source code. In this paper, we will examine a collection of such refinements and empirically evaluate their impact on the final model accuracy through ablation study."* Stronger than a genre claim — see the note below |
| 16 | every encoder splits the image into a uniform grid, so foveation is the one axis whose meaning does not change with the backbone | property of the patch tokenisers we run, not a claim about a cited paper; the corresponding statement in `RelatedWork.md` §2.3 (d) is what it compresses |

## Cited but **not** read — lineage only, no specific claim attached

These carry no number, quote or characterisation in the draft. They are cited
for provenance of an idea, which is the lowest-risk use, but they are unread
and that should not be forgotten before the bibliography is finalised.

| citation | what the draft asserts about it | risk |
|---|---|---|
| the two efficient-VLA surveys (arXiv 2510.17111, 2510.24795) | only that such surveys *"are now appearing"* | low — an existence claim |
| Gromov et al., *The Unreasonable Ineffectiveness of the Deeper Layers* | grouped with ShortGPT as establishing layer redundancy | low, but it is a substantive grouping; read before submission |
| DQN (frame skip) | the origin of executing one decision over several steps | low — textbook lineage |
| ACT, Diffusion Policy | action chunking in imitation learning | low — textbook lineage |
| SimplerEnv, LIBERO | the benchmarks results are reported on | none — we run them |
| `schwartz` — log-polar retinotopic mapping | that log-polar is his | **settled 2026-08-22 without needing his PDF.** Traver's bibliography gives the citable entry: *Schwartz, "Computational anatomy and functional architecture of the striate cortex," Vision Research 20 (1980) 645–669.* Cite **1980**; drop the 1977/1980 conflation. Traver Fig. 1 also confirms the model form — *"The retinal plane is mapped onto the cortical plane via w = log(z)"*, with a central circle omitted to handle the singularity, which is where the `+a` variant comes from |
| `bagoftricks_llm` — *Bag of Tricks for Inference-time Computation of LLM Reasoning* | that re-measuring is an established form | none — PDF read and measured (`RelatedWork_Plan.md` §5) |

**All four of the remaining unread citations were read on 2026-08-22**
(`gromov`, `traver`, `bagoftricks_cnn`, plus `starvla` which had been left
uncited). Nothing in the section is now cited unread except the two
efficient-VLA surveys, whose only claim is that they exist.

**`bagoftricks_cnn` was undervalued and the assessment above was wrong.** It
was filed as *"a genre claim, no number attached — low priority, do not bother
reading."* Its abstract states our thesis directly, eleven years early:
*"most refinements are either briefly mentioned as implementation details or
only visible in source code."* That is precisely what we found about the
candidate-window choice. It is a **precedent for the specific claim**, not
just for the format, and the Related Work sentence citing it can say so.

**~~Deliberately not cited:~~ Now citable — StarVLA.** It was left out because
we had only a search snippet. The PDF was read on 2026-08-22 and it is now
cited alongside vla-eval. Verbatim, from the abstract:

> *"VLA methods remain fragmented across incompatible architectures,
> codebases, and evaluation protocols, hindering principled comparison and
> reproducibility."*

and from §1: *"results are reported on disjoint subsets of benchmarks with
inconsistent protocols, making fair comparison infeasible,"* creating a
*"Tower of Babel"* for VLA research.

⚠️ **The caution was justified.** The snippet we had been carrying —
*"most existing VLA methods are evaluated on limited environments with
substantially different preprocessing pipelines, policy interfaces, and
evaluation protocols"* — **does not appear in the paper.**
`Survey_2026-08.md` §3 had it inside quotation marks. The substance was right
and the quotation was not, which is the exact failure mode refusing to cite
from a snippet was meant to prevent. Corrected there.

**Deliberately not cited:** the ten training-free papers in
`Survey_2026-08.md` §4. Abstract-level only. The MoLe-VLA lesson applies.

---

## Two claims worth re-checking before submission

1. **The matched-episode claim.** This is a negative claim over the papers we
   read, not over the field. The draft states it as a property of the tables
   we cite, which is what we can defend. If it is ever widened to "no prior
   work does this," it needs the systematic sweep again.

2. **The crossing-cell claim.** Verified across five papers, but it is the
   sentence a reviewer is most likely to test, and the field is moving fast —
   `Survey_2026-08.md` exists because this claim was already too strong once.
   Re-run the sweep close to submission.

---

## What is deliberately absent from the section

- **No table.** *Bag of Tricks* puts its configuration table in the
  Introduction, before Related Work; ours goes in the same place
  (`Table~\ref{tab:reporting}`), and Related Work only points at it.
- **The per-task split (§2.5 of `RelatedWork.md`)** appears as one clause, not
  a paragraph. It is a finding, and findings belong in Results and in the
  Introduction, not in background.
- **Our own numbers.** Related Work makes no measurement claim of ours except
  the forward pointer in the closing sentence.
- **The 8 × 3 scale statement.** Eight backbones (three ours, five the
  mentor's) across three benchmarks is a fact about our Setup, and it goes
  there. Related Work claims a *method* — a complete grid plus matched
  episodes — deliberately **not** a coverage record, because vla-eval already
  publishes 14 benchmarks × 6 model servers and 657 results, and 8 × 3 will
  not fill completely (no public SpatialVLA/LIBERO or UniVLA/Fractal weights;
  depth pruning is unavailable or already applied on three of the eight).
- **Scope disclaimer, present on purpose.** The closing paragraph names
  quantisation, KV-cache compression and learned early exit as out of scope.
  That is not a citation claim; it forecloses the obvious "why not X" without
  spending a paragraph on each.
