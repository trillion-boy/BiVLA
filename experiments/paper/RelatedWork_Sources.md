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
| 14 | OpenVLA-OFT uses parallel decoding and action chunking | **upgraded from second-hand to primary, 2026-08-22.** Was verified only from VLA-Pruner's description of its own baseline; now from the paper itself (Kim, Finn & Liang, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*), abstract p1: *"an Optimized Fine-Tuning (OFT) recipe that integrates **parallel decoding, action chunking**, a continuous action representation, and a simple L1 regression-based learning objective"* |
| 20 | our three axes are the field's own categories, not our invention | **PDF read 2026-08-22.** The CAS systematic survey (Guan et al., Institute of Automation, arXiv 2510.17111, confirmed from the listing page) divides efficiency work into §3.2 *Dynamic Computation Pathways*, §4 *Efficient Perception Feature*, §5 *Efficient Action Generation* — our axes 3, 2 and 1 — and populates them with the same papers we cite (§3.2 names SmolVLA and FLOWER; §4.1 names FastV and EfficientVLA; §4.2 names VLA-Cache) |
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
| `effvlasurvey2` — arXiv 2510.24795, *A Survey on Efficient Vision-Language-Action Models* | only that such surveys exist | low — an existence claim. **The only remaining unread citation in the section.** Full bibliographic data verified, see below |
| DQN (frame skip) | the origin of executing one decision over several steps | low — textbook lineage |
| ACT, Diffusion Policy | action chunking in imitation learning | low — textbook lineage |
| SimplerEnv | the benchmark results are reported on | none — we run it, all 24 result directories are SimplerEnv |
| LIBERO | named alongside SimplerEnv as where the field reports | low — a statement about the field's practice, not about us. **We do not run LIBERO in this paper.** The five-model expansion would, and `notebooks/05` supports it, but no LIBERO run exists in `results/`. Do not let this citation drift into implying we evaluated on it |

**Read since this table was first written**, and therefore no longer on it:
`gromov` (see claim 18 and the Table I reframing in `TableI_Cells.md` §4d),
`traver` (claim 17), `bagoftricks_cnn` (claim 19), `openvlaoft` (claim 14),
`starvla`, `effvlasurvey1` (claim 20), and `bagoftricks_llm`
(`RelatedWork_Plan.md` §5). `schwartz` is settled from the published abstract,
below.

> **arXiv identity, resolved 2026-08-22 from the arXiv listing page.**
>
> `effvlasurvey2` = **arXiv 2510.24795**, *A Survey on Efficient
> Vision-Language-Action Models*, Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan
> Zhang, Ji Zhang, Zheng Wang, Lianli Gao, Jingkuan Song, Nicu Sebe, Heng Tao
> Shen. Submitted 27 Oct 2025 (v1), revised 2 Feb 2026 (v2). 28 pages, 8
> figures. `doi:10.48550/arXiv.2510.24795`. Use **v2** if the number is pinned.
>
> `effvlasurvey1` = **arXiv 2510.17111**, *Efficient Vision-Language-Action
> Models for Embodied Manipulation: A Systematic Survey*, Weifan Guan, Qinghao
> Hu, Aosheng Li, Jian Cheng (Institute of Automation, CAS). Submitted 20 Oct
> 2025 (v1), revised 23 Oct 2025 (v3). cs.RO.
> `doi:10.48550/arXiv.2510.17111`. **Confirmed 2026-08-22 from the arXiv
> listing page**, which also matches its abstract's four dimensions, namely
> model architecture, perception feature, action generation and
> training/inference strategies. Both survey numbers are now settled.
>
> **And the two taxonomies differ, which confirms we cited the right one.**
> 2510.24795 organises the field by pipeline stage, namely efficient model
> design, efficient training and efficient data collection. That is not our
> axis structure. The CAS survey organises by resource spent, giving dynamic
> computation, perception and action generation, which is what claim 20 rests
> on. `effvlasurvey2` therefore stays an existence-only citation, and its
> abstract-level remarks about the field lacking a unified framework are
> **not** cited, per the MoLe-VLA rule.

### Schwartz — closed, and the paywalled PDF is not needed

The draft's only claim is an **attribution of the idea**: *"the log-polar
retinotopic mapping of Schwartz."* The published abstract states exactly that:

> *"The logarithm of retinal eccentricity provides a good fit to the integrated
> cortical magnification factor. Under the assumption that the cortical map is
> analytic (conformal), this implies that a **complex logarithmic function of
> retinal coordinates** describes the two-dimensional structure of the cortical
> representation of a visual stimulus."*

An abstract is weak evidence for what a paper *did* (the MoLe-VLA lesson), but
this is not that kind of claim — no number, table or characterisation of the
method hangs on it. Buying the PDF would buy nothing.

**Bibliographic entry, authoritative version.** The DOI page gives a longer
title than Traver's bibliography does; use the DOI page.

| field | value |
|---|---|
| author | Eric L. Schwartz |
| title | Computational anatomy and functional architecture of striate cortex: A spatial mapping approach to perceptual coding |
| journal | Vision Research **20** (1980) 645–669 |
| doi | `10.1016/0042-6989(80)90090-5` |

⚠️ Traver ref [24] renders it as *"…functional architecture of **the** striate
cortex"* and drops the subtitle. Do not copy the entry out of Traver.

**And `w = log(z + a)` really is Schwartz's — checked, because it looked like
it might not be.** The abstract says only "complex logarithmic function,"
i.e. `log(z)`, so the `+a` offset in `RelatedWork.md` §2.3 (a) looked like a
later engineering fix wrongly attributed to him. It is not. Traver §2.3:

> *"either using a different mapping for the fovea … or applying the
> **log(z + a)** model. This other model was proposed in **[24]** as a better
> approximation to the retino-topic mapping of monkeys and cats."*

and Traver's `[24]` is Schwartz 1980. Our note stands as written.

### Gromov — nothing further needed either

The PDF supplied **is** the camera-ready: *"Published as a conference paper at
ICLR 2025"* is printed on every page. *".bib needs fixing"* meant our own
bibliography entry should read **ICLR 2025**, not an arXiv preprint — a
one-line edit in a `.bib` we have not written yet, not a request for another
copy of the paper. Phrasing it as "fix the .bib" without saying who fixes it
was ambiguous; recording the resolution here so it does not resurface.

---

**All four of the remaining unread citations were read on 2026-08-22**
(`gromov`, `traver`, `bagoftricks_cnn`, plus `starvla` which had been left
uncited). Nothing in the section is now cited unread except the two
efficient-VLA surveys, whose only claim is that they exist.

### Three times in a row, a paper was under-rated before being read

Worth recording as a pattern rather than three separate slips, because the
next "low priority, existence claim only" judgement should be distrusted.

| paper | rated before reading | what it turned out to be |
|---|---|---|
| Gromov | *"low risk, a substantive grouping — read before submission"* | the mechanism that explains our worst cell (−30.4), and the paper that makes Table I's constraint column **contested** rather than empty |
| Bag of Tricks (CNN) | *"a genre claim, no number attached"* | states our thesis eleven years early — *"only visible in source code"* |
| CAS survey (2510.17111) | *"an existence claim"* | supplies our three axes as **its own** section headings, with our own citations inside them |

The common error: rating a paper by the **weight of the claim we currently
hang on it**, rather than by what it might let us claim. One survey remains
unread on exactly that reasoning (`effvlasurvey2`), and its expected value
should be treated as unknown, not low.

### Held in reserve — the survey's §7.5, for Limitations

Not cited in Related Work, but directly usable where we discuss the GPU-model
gap. From *Future Prospects*, p19:

> *"Current VLA studies report heterogeneous metrics, rely on diverse
> datasets, and employ **varied hardware**, making cross-comparison difficult
> and obscuring true efficiency gains."*

This is a **third** independent 2025–26 source stating our motivation
(vla-eval and StarVLA are the other two), and it is the only one that names
hardware, which is the limitation we have to declare anyway.

⚠️ **OpenVLA-OFT is not training-free** — it is a fine-tuning recipe. We cite
it only as an example within axis 1 (*when the policy runs*), which the
section's structure already separates from the training-free interventions in
paragraph 2. Keep it that way; same rule as MoLe-VLA and Gromov.

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

---

## Foveated tokenization, and why a training-free study cannot use it

Added 2026-08-22 from two PDFs the author supplied. This is the provenance for
the clause *"Methods that foveate inside the encoder give distant patches a
coarser resolution and so shed tokens, but they give up the pretrained
weights, which were fitted to a uniform patch grid."*

### The two papers foveate at the tokenizer, not in pixels

**Look, Focus, Act** (GIAVA), abstract:

> *"we integrate gaze information into ViTs using a **foveated patch
> tokenization scheme**. Compared to uniform patch tokenization, this
> significantly **reduces the number of tokens**, and thus computation."*

**Segment This Thing** (Meta, arXiv 2506.11131), abstract:

> *"a novel **variable-resolution patch tokenization** in which patches are
> downsampled at a rate that increases with increased distance from the
> prompt. This approach yields **far fewer image tokens** than uniform patch
> tokenization."*

Its Fig. 2 caption states the pattern: *"A patch from the center maintains its
original resolution. A patch from the outer ring gets **downsampled by a factor
of 8**."*

### The sentence the clause rests on

Look, Focus, Act says outright what it costs, and what they had to do instead:

> *"One downside of using a foveated tokenization scheme is that **open-source
> pretrained ViT weights**, commonly used in robot learning for their
> significant performance benefits, **cannot be directly applied** as
> pretraining is done on fixed, uniform tokenization patterns. To address
> this, **we pretrain our own ViT-B models from scratch** using the Masked
> Autoencoder (MAE) objective."*

Segment This Thing records the same wall hit by a third system:

> *"GazeGPT ... This model was **pre-trained on uniform resolution images,
> which constrains** the GazeGPT foveation model (it uses three overlapping
> images of the same size but varying receptive field)."*

### Why this matters to us

Both papers train. LFA pretrains ViT-B from scratch on 60,000 images for 1,000
epochs, and STT trains each instance of its model. Our study is training-free
by construction, so the tokenizer is fixed, so the token count is fixed, so
foveation for us can only be a pixel-space edit.

**That is the reason our foveation saves no compute**, measured at within
$3.1\%$ of baseline in every condition we ran. It is not an implementation
oversight and it would not be fixed by skipping the resize back to full size,
because the backbones' own preprocessors resize to their fixed input anyway
(`inference_spatialvla_libero.py:121`).

### The claim boundary

The clause says foveating inside the encoder *sheds tokens* and *gives up the
pretrained weights*. Both halves are quoted above. It does **not** claim our
pixel-space variant is better, and it does not claim these papers should have
done otherwise. They train, so the trade is available to them and not to us.
