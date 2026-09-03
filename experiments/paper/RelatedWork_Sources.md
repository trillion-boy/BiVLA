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
| 5 | VLA-Pruner reproduces the degradation on another setting | read this session: its Table 2, FastV retains 73.1% of baseline on SIMPLER at 75% |
| 6 | FLOWER prunes 30–50% of its VLM layers by design | read this session: *"we prune between 30% and 50% of the pretrained VLM's layers"* |
| 7 | SmolVLA retains only the first sixteen | read this session: *"We use only the first 16 layers of the LLM within the VLM"* |
| 8 | TurboVLA removes the language model from the action path | read this session: *"avoiding … processing multimodal inputs through a billion-parameter language model"*, chunks emitted *"without autoregressive action-token generation"* |
| 9 | vla-eval unifies fourteen benchmarks and documents previously undocumented pitfalls | read this session: 14 simulation benchmarks, six model servers, *"documenting previously undocumented pitfalls"* |
| 10 | papers using several backbones change the benchmark at the same time | VLA-Cache (A.2: LIBERO→OpenVLA, SIMPLER→CogACT); SpecPrune-VLA (Table 1: LIBERO→OpenVLA-OFT, SimplerEnv→DB-OFT); VLA-IAP (Table 1 caption: DreamVLA, π0 LIBERO, π0.5 VLABench) |
| 11 | those with both axes leave the crossing cell empty | Gaze-Reg (A.5); VLA-Pruner (LIBERO has OpenVLA + OpenVLA-OFT, SIMPLER has OpenVLA only) |
| 12 | the tables we cite report mean success rates, not matched-episode outcomes | grep over every PDF read: *paired* 0, *McNemar* 0, and in vla-eval also *significance* 0, *statistic* 0, *confidence* 0, *error bar* 0. The draft says **"the tables we cite,"** not "no prior work" — see the re-check note below |
| 13 | FastV, SparseVLM and ToMe were developed for VLMs | all three PDFs read this session |
| 14 | OpenVLA-OFT uses action chunking (the draft cited it for parallel decoding until 2026-08-24, see the note below) | **upgraded from second-hand to primary, 2026-08-22.** Was verified only from VLA-Pruner's description of its own baseline; now from the paper itself (Kim, Finn & Liang, *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*), abstract p1: *"an Optimized Fine-Tuning (OFT) recipe that integrates **parallel decoding, action chunking**, a continuous action representation, and a simple L1 regression-based learning objective"* |
| 20 | our three axes are the field's own categories, not our invention | **PDF read 2026-08-22.** The CAS systematic survey (Guan et al., Institute of Automation, arXiv 2510.17111, confirmed from the listing page) divides efficiency work into §3.2 *Dynamic Computation Pathways*, §4 *Efficient Perception Feature*, §5 *Efficient Action Generation* — our axes 3, 2 and 1 — and populates them with the same papers we cite (§3.2 names SmolVLA and FLOWER; §4.1 names FastV and EfficientVLA; §4.2 names VLA-Cache) |
| 15 | foveation has been applied to VLAs by gaze-conditioned policies | `RelatedWork.md` A.5 (Gaze-Reg, arXiv 2603.23202) and A.6 (Look Focus Act, arXiv 2507.15833), both marked *원문 확인* — the two PDFs were opened and their tables cross-checked |
| 17 | log-polar was adopted in robot vision to cut data while preserving central resolution | **PDF read 2026-08-22**, Traver & Bernardino p3: *"Elegant trade-off solution between these three mutually opposing criteria: wide field of view, high visual resolution and little data to process … the reduced size of log-polar images (as much as 30 times smaller than uniformly-sampled Cartesian images have been reported) hugely facilitates real-time visual data processing. On the other hand, the radially logarithmic sampling entails that a higher resolution is devoted to the center of the scene (fovea area)"* — both halves of our clause, verbatim, with a number |
| 18 | layer redundancy in language models is well established (the `gromov` half) | **PDF read 2026-08-22**: *"we can remove a substantial fraction of the deepest layers from models with minimal degradation … for Llama-2-70B we can eliminate up to roughly half of the layers before the performance collapses."* **ICLR 2025**, not an arXiv preprint — `.bib` fixed 2026-08-24, `gromov` is now `@inproceedings` with `booktitle` ICLR and the arXiv id in a note |
| 19 | re-measuring methods one did not invent is an established form | **PDF read 2026-08-22**, Bag of Tricks (CNN) abstract: *"most refinements are either briefly mentioned as implementation details or only visible in source code. In this paper, we will examine a collection of such refinements and empirically evaluate their impact on the final model accuracy through ablation study."* Stronger than a genre claim — see the note below |
| 16 | the encoders we run each split the image into a uniform grid, so foveation is the one axis whose meaning does not change with the backbone | property of the patch tokenisers we run, not a claim about a cited paper; the corresponding statement in `RelatedWork.md` §2.3 (d) is what it compresses |

## Cited but **not** read — lineage only, no specific claim attached

These carry no number, quote or characterisation in the draft. They are cited
for provenance of an idea, which is the lowest-risk use, but they are unread
and that should not be forgotten before the bibliography is finalised.

| citation | what the draft asserts about it | risk |
|---|---|---|
| `effvlasurvey2` — arXiv 2510.24795, *A Survey on Efficient Vision-Language-Action Models* | only that such surveys exist | low — an existence claim. **The only remaining unread citation in the section.** Full bibliographic data verified, see below |
| ~~DQN (frame skip)~~ | **read 2026-08-24, see below** | moved off this table |
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

## Two negative claims that had no entry here, fixed 2026-08-23

Claim 12 above was written with care: the draft says *"the tables we cite,"*
not *"no prior work,"* because a search that finds nothing is evidence about
the search, not about the world. Two other sentences broke that same rule and
never got a row in this table, which is how they survived several passes.

| was | now | why |
|---|---|---|
| *"These rules have not been compared on a robot policy"* | *"We find no comparison of these rules on a robot policy"* | a claim about the entire literature, resting on our reading set. One reviewer who knows of a comparison ends it. The new form is a claim about our search, which is true and which we can defend |
| *"Unique to this paper, we measure the same interventions on both axes…"* | *"We measure the same interventions on both axes…"* | a priority claim, and the weakest kind, since it asserts a negative about everything unread. It was also redundant: the preceding three sentences already name three specific gaps, each with citations |

Neither change costs anything. The first still says the comparison is missing
and still says our depth results turn on it. The second still says what we do.
What both drop is the part a reviewer could refute with a single citation.

⚠️ `RelatedWork_Plan.md` still lists the *"one item that needs the mentor,"*
namely whether anyone knows of a counterexample to the first claim. That
question is now **optional rather than blocking**, since the sentence no
longer depends on the answer. Ask it anyway if there is a chance to, because a
known comparison would be worth citing.

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

---

## Pass on the prose, 2026-08-24: five pronouns and one scope

Run after the Introduction's passes had built a defect list, applying the same
classes here. Nothing in this pass changed a claim, a citation or a number.
Related Work still carries zero numbers.

### The scope defect, which contradicted the Introduction

*"No matter how the encoders differ, each splits the image into a uniform
grid."* That is a claim about **every** patch tokeniser ever built. Claim 16
in this file says what the evidence actually is: *"property of the patch
tokenisers we run, not a claim about a cited paper."* And the Introduction
already scopes it correctly, saying *"Our three backbones tokenize a fixed
patch grid."* So the two sections disagreed about how much we know.

Now *"The encoders we run differ in many ways, but each splits the image into
a uniform grid."* Same argument, same length, and it now claims exactly what
claim 16 supports.

### Five pronouns, the class that has bitten this paper twice

| was | problem |
|---|---|
| *"We divide **it** by the resource each method spends"* | `it` was meant to be the literature; the nearer noun is *surveys*. Now *"that literature"* |
| *"but **its** recipes disagree on which layers may go"* | `its` attaches to *layer redundancy*, and redundancy does not prescribe anything. Now *"the recipes built on it"*, which gives both nouns an owner |
| *"attributes this to **their** short action sequences"* | resolves to *VLAs* by proximity, but *methods* is also in range and is the wrong reading. Now *"the short action sequences VLAs produce"* |
| *"We include foveation because **ours** stays before the encoder"* | the same shape as the `ours` bug in the Introduction, where the antecedent had been edited away. Nearest nouns here are *grid*, *weights*, *Methods*. Recast as *"Our foveation stays before the encoder … and that is why we include it"* |
| *"Editing pixels leaves the token count **the backbone already has**"* | a contact clause with no *that*, reading as *"leaves the token count the backbone"* on the way in. Now *"leaves the backbone's token count unchanged"* |

### Two soft quantifiers, same family as the "two recipes" fix

*"VLA-Pruner reproduces the degradation on a **third** setting"* counts
settings the section never enumerates, so a reviewer can ask which three.
Claim 5 supports the reproduction, not the count. Now *"another setting."*

*"foveation comes from **the other direction**"* asked the reader to work out
which two directions were on offer. Now *"a different lineage,"* and the next
sentence delivers the lineage.

### One vocabulary fix

*"not only about a **switch**"* was a fourth synonym for the thing the paper
calls a *configuration* or a *setting*. Now *"an inference-time setting,"*
which is the Introduction's vocabulary.

### Checked and deliberately left alone

**The three passive axis definitions.** *"Visual tokens are reduced or
reweighted"*, *"Decoder layers are skipped"*, *"A predicted chunk is
executed"*. Active versions would have to name an agent (*"methods reduce…"*)
three times in four lines, and the passive correctly foregrounds the resource,
which is what an axis is. Passive is 5 of 39 sentences and three of those five
are these.

**"an established form"** is elliptical but idiomatic, and expanding it costs
words for nothing.

**"our depth results turn on which one holds"** presupposes a countable set,
but the two rules are named in the two sentences directly above it, so the set
is on the page.

---

## "Why these five?" — the one citation question with no written answer

Asked 2026-08-24. Contribution 4 says *"Of the five training-free methods we
survey~\cite{efficientvla, vlacache, fastv, shortgpt, vlapruner}, every one
sweeps its own configuration and none reports per-episode outcomes or tests a
difference for significance."* `Survey_2026-08.md` §4 lists **ten more** papers
that claim training-free in their abstracts. A reviewer who has that list will
ask why the count is five, and nothing in this repository answered it.

**The answer is that the claim defines the sample, not the other way round.**
It is a claim about **absence**. An abstract cannot show that a paper never
runs a paired test, because absence is only visible in the full text. The five
are exactly the training-free methods whose complete PDFs we read and grepped,
twice, under two term sets (`TableI_Cells.md` §3). The other ten are
abstract-only, and this file's standing rule, learned from MoLe-VLA, is that an
abstract is not evidence for what a paper did.

So the sample is not a convenience sample. It is the largest set for which the
claim is checkable, and each member is named in the sentence.

**What the sentence deliberately does not say** is *"no prior work reports
per-episode outcomes."* That would be the unbounded version, and it is the form
this file has refused twice already, at claim 12 and in the 2026-08-23 pass.
Five named papers, read in full, is a claim a reviewer can verify or refute by
opening five PDFs.

**If a reviewer still wants breadth**, the answer is to read more PDFs and
raise the number, not to cite the ten on abstract evidence. `Survey_2026-08.md`
§4 already argues against adding them for a different reason: the token-pruning
axis has ten-plus entries in eighteen months, so joining it as a competitor is
not our contribution. We use FastV as an **object** of measurement precisely
because two independent groups already report it differently on VLM versus
VLA.

---

## The axis paragraph, 2026-08-24: two reviewer catches and one they led to

### 1. The category list was in reverse order, and this file said so

The draft said our division *"recovers a recent survey's categories of dynamic
computation, perception and action generation"* and then listed the axes as
1) when the policy runs, 2) what it is shown, 3) how much network each call
uses. Those are the survey's categories **backwards**. Claim 20 in this file
records the mapping in exactly those words: *"our axes 3, 2 and 1."* A reader
had to invert the list to check the claim.

Our own order is not free to change, because `introduction.tex` fixes it:
action repetition, then foveation, then depth pruning. So the survey's list is
the one that moves. It now reads *"action generation, perception and dynamic
computation."* No claim changed and the mapping is now positional.

### 2. Axis 3 was stated more narrowly than axes 1 and 2

*"Decoder layers are skipped"* names one mechanism for an axis that also
covers MoE routing, early exit and architectural pruning. Axes 1 and 2 are
both stated generally (*"a lineage running from frame skip to parallel
decoding"*, *"visual tokens are reduced or reweighted"*), so the asymmetry was
the actual defect rather than the narrowness on its own.

The suggested repair was a hedge, *"For instance, decoder layers are
skipped."* A hedge makes the axis sound tentative. Generalising it to match
the other two is better and the same length: **"Only part of the decoder
runs."** That covers every mechanism in the axis, and paragraph 2 immediately
specifies layer skipping as the one we test. It also drops a passive.

### 3. What checking number 2 turned up, and it was worse than either

The closing paragraph read *"We do not cover quantisation, KV-cache
compression or learned early exit, which spend resources our axes do not."*

**Learned early exit spends exactly axis 3's resource.** It decides how much
of the network a call uses. So the section defined an axis and then excluded a
method from it on the ground that the method belongs to no axis, three
paragraphs apart.

The real reason to exclude it is the one already applied to MoLe-VLA: it is
**learned**, so it is not training-free. Now *"We do not cover quantisation or
KV-cache compression, which spend resources our axes do not, nor learned early
exit, which is not training-free."*

This is worth recording as a pattern. Both reviewer catches were in the axis
paragraph; the contradiction was in the closing paragraph and was only visible
because the axis definitions had just been re-read. **A defect can sit outside
the passage that reveals it.**

Passive in the section is now 4 of 39 sentences, down from 9 when these passes
began. Prose is 678 words.

### The VLA-Cache sentence, 2026-08-24: two of three taken

**Taken: the bare `this`.** *"VLA-Cache attributes **this** to the short action
sequences VLAs produce."* The referent is *poor transfer* one sentence back.
This is the pronoun class that has been fixed four times in this paper already,
so it goes the same way: *"attributes this poor transfer to."* The echo of
*"transfer poorly"* is deliberate. Repeating the noun is what binds the
reference, and claim 4 in this file uses the same words.

**Taken: the split.** The sentence chained two `and`s across three facts. It is
now two sentences, and the VLA-Pruner clause reads as its own corroboration
rather than as a tail.

**Declined: replacing `measures` with `finds that`.** The verb is carrying
evidence. Claim 4 records what is behind it: VLA-Cache's own numbers, 1.864 T
FLOPs unchanged and 51.91 to 53.28 ms. *"Finds that"* would state the same fact
while dropping the fact that they **measured** it, and our use of the sentence
depends on it being a measurement, since we cite it as evidence that FastV does
not save what it is credited with saving.

The construction itself is standard English, a measurement verb taking an
object and a participle, as in *"measured the beam bending."* It is not a
garden path: *"measures FastV"* parses, and *"leaving FLOPs unchanged"* then
says what was measured about it. The 0.1-second cost, if it is real, buys a
word that a reviewer cannot ask us to justify.

### "independent runs" was a claim about their setup that we never checked

Claim 12 used to read *"mean success rates **over independent runs**, without
matched-episode outcomes."* The evidence behind it is the grep in the right
column: `paired` 0, `McNemar` 0, and no dispersion of any kind. That
establishes **the absence of paired analysis**. It says nothing about whether
their runs were independent.

And SimplerEnv is deterministic given a seed and an initial state, which we
know because our own determinism checks depend on it. So the likely truth is
the opposite of what we wrote: those papers probably ran a fixed protocol and
then pooled the outcomes without pairing them. Calling that "independent runs"
mischaracterises the benchmark as much as the papers.

The clause is gone. *"The tables we cite report mean success rates, not
matched-episode outcomes"* is exactly what the grep supports, and it is the
sharper criticism, because the failure is in the analysis rather than in the
experiment.

### "axes" was carrying two meanings, and one collision was inside one paragraph

Paragraph 1 defines the term: *"We treat these as **axes** rather than
competing methods, because a claim about efficiency is a claim about one of
these **resources**."* After that, `axes` means resources. Two later sentences
used it for the grid's dimensions instead:

| where | was | now |
|---|---|---|
| paragraph 3 | *"Those with both **axes** present leave the crossing cell empty"* | *"both **factors**"* |
| closing | *"on both **axes** of a backbone $\times$ benchmark grid"* | *"over a backbone $\times$ benchmark grid that **moves both factors**"* |

The second is the worse of the two, and it was not the one raised. It sits
**two sentences** after *"which spend resources our axes do not"*, so one
paragraph used both senses of the same defined term. `factors` is the standard
word for the dimensions of a crossed design, and the new closing phrase now
echoes the Introduction's *"a grid that moves the backbone and the benchmark
independently."*

All three surviving uses of `axes`/`axis` are the resource sense.
`audit_sections.py` now fails on any use of `axes` next to `backbone`,
`benchmark` or `grid`, because the J-naming check could never have caught this
one: it looks for two names for one idea, and this was one name for two ideas.

### The closing paragraph, 2026-08-24: three of four, and one push-back

**Taken: the verb-object mismatch.** *"Measuring the settings papers leave to
their source code"* measures the wrong noun. We do not measure a setting, we
measure what one does to a result. Now *"Measuring the effect of choices that
papers leave to their source code."*

*"Leave to"* stays, and *"relegate to"* was declined. Claim 19 records the Bag
of Tricks wording we are echoing: *"only visible in source code."* Neutral
description is what that sentence is doing, and *"relegate"* would import a
judgement the citation does not carry. *"Choices"* rather than *"settings"*
also ties to the Introduction's *"A choice the specifications leave open."*

**Taken: `moves both factors`.** *"Varies"* is the term for a crossed design,
and *"independently"* adds the property that makes it factorial. Changed in
**both** sections: `introduction.tex` had *"a grid that moves the backbone and
the benchmark independently"* and now says *"varies."* Changing only Related
Work would have produced exactly the two-names-for-one-idea defect the
J-naming check exists for.

**Taken, differently: the `nor` chain.** Two relative clauses hanging off a
`nor` was choppy. The suggestion repeated the verb; two sentences are cleaner
still. *"We do not cover quantisation or KV-cache compression, which change
none of our three resources. We also exclude learned early exit, which is not
training-free."*

**Declined: `which spend resources our axes do not` to `which target different
computational bottlenecks`.** The objection is that KV-cache compression cuts
memory bandwidth and attention cost, so calling that a different *resource* is
loose.

Two reasons the replacement is worse.

1. It is a claim about **what those methods are for**, and we have read no
   quantisation or KV-cache paper. Neither is in our bibliography. We would be
   characterising an uncited literature to avoid a claim about our own
   definitions.
2. Our axes are **defined** two paragraphs earlier as when the policy runs,
   what it is shown, and how much network each call uses. Quantisation changes
   none of the three: same schedule, same input, same layers, different
   arithmetic. KV-cache compression changes none of the three either: same
   schedule, same input, same layers, different memory for the history.

So the original claim was true under our own definition, and the objection
comes from reading *resource* in the everyday sense rather than the defined
one. The fix is to make the definition visible rather than to retreat to a
vaguer claim: **"which change none of our three resources."** That points at
the definition, asserts nothing about anyone's intent, and is checkable
against paragraph 1.

### The same sentence, attacked twice, 2026-08-24

The exclusion clause has now drawn fire in two consecutive rounds, from two
different angles, and it was defensible both times. That is the point worth
recording: **a true sentence that keeps attracting the same objection is
badly worded, not misunderstood.**

| round | wording | objection |
|---|---|---|
| 1 | *"which spend resources our axes do not"* | KV-cache compression cuts memory bandwidth and attention cost, so it plainly spends a resource |
| 2 | *"which change none of our three resources"* | axis 3 maps to the survey's *dynamic computation*, and KV-cache compression is a dynamic-computation optimisation |

Round 1's fix was declined on good grounds and round 2's replacement is the
better sentence anyway. The problem in both versions is the word
**resource**. Paragraph 1 defines the axes *by* resource, so any sentence
using that word invites the everyday reading, in which of course quantisation
and KV-cache compression spend resources.

Now: *"which operate outside the three axes above."* It makes the same
definitional claim, points at where the definition lives, and never uses the
contested noun. It also does not assert what those methods are for, which is
the reason the round-1 alternative was refused, since neither is in our
bibliography.

⚠️ On the round-2 objection's premise: our record does **not** establish that
the CAS survey files KV-cache compression under §3.2. Claim 20 records §3.2 as
naming SmolVLA and FLOWER, both architectural layer reduction, and it records
VLA-Cache under **§4.2, perception**, not under dynamic computation. So the
mapping the objection assumes is unverified in either direction. The rewrite
does not depend on it.

### Two more, both taken

**"an established form" was doing genre criticism.** `Form` belongs to
discussions of poetry and painting. What we mean is that people do this and it
is respectable, which is **practice**. Now *"an established practice."*

**`Measuring` then `We measure` in consecutive sentences.** Sentence 4 keeps
`measure`, because it is the paper's own verb: contribution 1 is *"A uniform
re-measurement"* and contribution 4 is *"A measurement procedure."* Sentence 3
takes **`Isolating`**, which is both a change and an improvement: claim 19's
quote says Bag of Tricks evaluates its refinements *"through ablation
study,"* and isolating one choice at a time is exactly what an ablation does.
The pair now reads as a distinction rather than a repetition. The genre
isolates effects; we measure them on a grid.

### "papers leave to their source code", 2026-08-24: agent removed, not swapped

Third objection to the same sentence in three rounds, so by the rule recorded
above it gets changed regardless of whether the objection is the strongest one
raised. It is not: *"the paper leaves this to future work"* is unremarkable
English, and *"papers leave X to source code"* is the same construction.

The genuinely loose part was **`their`**. Papers do not own source code.

Three repairs were offered and none was taken, because each swaps one problem
for another.

| offered | why not |
|---|---|
| *"choices that **authors** leave to their source code"* | fixes the possessive by naming people. This paper is careful not to blame anyone, and says so about itself: *"our implementation added it anyway."* Pointing at authors changes that posture for one word of grammar |
| *"choices **omitted from papers and left to** source code"* | correct, and passive twice in one clause. Passive is 4 of 41 sentences and this would make it 5 |
| *"choices that papers **relegate to** source code"* | already declined once. `Relegate` means demoted to a lesser place, which is a judgement claim 19's source does not make |

Taken instead: **remove the agent.** *"Isolating the effect of choices that
appear only in source code is an established practice."*

No agent to personify, no possessive, no judgement, still active, one word
shorter. And it is now almost the source's own wording. Claim 19 quotes Bag of
Tricks: *"most refinements are either briefly mentioned as implementation
details or **only visible in source code**."*

The general lesson, since this is the second time in two rounds: when a phrase
keeps drawing fire, check whether the contested element can be **deleted**
before choosing among replacements. The previous round removed the word
*resource*; this one removes the subject.

### Axis 1 named the wrong endpoint, 2026-08-24

*"A predicted chunk is executed over several control steps, a lineage running
from frame skip in RL to **parallel decoding**."*

Axis 1 is *when the policy runs*. Frame skip and action chunking belong there:
both hold one decision across several control steps, so both change how often
the policy is called. **Parallel decoding does not.** It emits the action
tokens of a single call in one pass instead of autoregressively. That is a
decoding-cost change inside one call, so it belongs to axis 3 if anywhere. The
lineage sentence named it as the endpoint of a lineage it is not in.

Claim 14 already held the right word. It records OFT's abstract as integrating
*"parallel decoding, **action chunking**, a continuous action representation,
and a simple L1 regression-based learning objective."* OFT has both, and only
one of them is on this axis.

Now: *"One decision is executed over several control steps, a lineage running
from frame skip in RL to action chunking."* The first clause also changed, from
*"A predicted chunk"* to *"One decision"*, which removes a `chunk`/`chunking`
echo and matches this file's own description of the DQN citation, *"the origin
of executing one decision over several steps."* All four citations stay: DQN,
ACT, Diffusion Policy and OFT are all action chunking or its ancestor.

Claim 14 is rescoped accordingly. The draft no longer asserts anything about
parallel decoding, so the verification behind it is now unused rather than
wrong. Do not re-add parallel decoding to axis 1.

### The compact-VLA inference was stated but not shown

*"Whether a reduction transfers is therefore a question about architecture and
not only about an inference-time setting."* The `therefore` did work the
sentence never showed. *Transfers* from what to what was unstated, and a reader
had to reconstruct the argument.

The argument is short and worth writing out: FLOWER prunes 30--50\% of its VLM
layers, SmolVLA keeps only the first sixteen, and TurboVLA drops the language
model from the action path (claims 6--8). So for those models, removing layers
at inference starts from a stack that has already been cut. Now: *"Whether
removing layers helps therefore depends on what the architecture already leaves
out, not only on an inference-time setting."*

`Leaves out` rather than `already removed` because TurboVLA removed a component
rather than layers, and the phrase has to cover all three.

Note what this sentence still does **not** claim. It does not say the effect
reverses across backbones. That is result 3, it belongs to the Introduction,
and Related Work should not assert our own findings.

### Two declined

**`The tables we cite` to `These prior studies`.** The scoping is the point,
and claim 12 records it: the draft says *"the tables we cite,"* not *"no prior
work."* `Tables` is also the more precise subject for what the sentence
asserts, which is what a printed table shows. And it matches the house
construction already used in the Introduction, *"the five training-free methods
we survey."* Plain is not informal.

**The comma in `produce, and measures`.** Strictly optional in a compound
predicate, but it is licensed when the predicates are long, and here it
prevents a stumble. Without it the text reads *"the short action sequences VLAs
produce and measures FastV,"* and a reader scanning `produce and measures` has
to check the agreement to rule out a compound verb. Removing a comma to gain
nothing and reintroduce a hesitation is the wrong trade in a section that has
had three garden paths removed.

### Round three on the exclusion clause, 2026-08-24: stop asserting, start applying

Third objection, third wording. Rounds 1 and 2 are recorded above. Round 3
called *"which operate outside the three axes above"* a cop-out and proposed
*"which target memory bottlenecks rather than the three computational axes."*

**That proposal is declined, and it is factually worse than the two before
it.** Quantisation does not only target memory. Int8 and int4 kernels are
faster arithmetic, so the claim is wrong for one of the two methods it
describes, and we have read neither paper. Round 1's version was refused for
being a claim about uncited work; this one is the same defect with a narrower,
falsifiable body.

**But the objection is right that the clause asserted a verdict instead of
showing its working.** Three rounds of that is enough. The clause now applies
the test rather than announcing its result:

> *"We do not cover quantisation or KV-cache compression. Neither changes when
> the policy runs, what it is shown, or how much network each call uses."*

That restates the three axes verbatim from paragraph 1 and lets the reader
check each method against them. Quantisation: same schedule, same input, every
layer still runs, different arithmetic. KV-cache compression: same schedule,
same input, every layer still runs, smaller history. Neither makes **only part
of the decoder run**, which is what axis 3 says.

To refute this a reviewer now has to argue that KV-cache compression makes part
of the decoder not run, which is false. The previous wordings could be refuted
by disagreeing with a word. Costs eight words and ends the argument.

### The comma, round two: the collision was the problem, not the comma

*"the short action sequences VLAs produce, and measures FastV"* drew the comma
objection twice. It was declined the first time on the grounds that the comma
prevents a reader from parsing `produce and measures` as a compound verb.

That defence was right about the hazard and wrong about the fix. **The hazard
is the verb `produce`**, and once it goes, the comma has nothing to do:

> *"attributes this poor transfer to **VLAs' short action sequences** and
> measures FastV leaving FLOPs unchanged"*

No verb collision, so no comma, and the possessive is explicit, which is what
`produce` had been added to achieve when the bare `their` was removed. One word
shorter than either previous version.

Third time in three rounds the answer was deletion rather than substitution:
the word *resource*, then the subject of *leave*, now the verb *produce*.

---

## DQN read 2026-08-24, and one of our own notes was wrong

The PDF arrived, so this citation is no longer lineage-only. Two things came
out of it, and the second is a correction to this file.

**The claim is confirmed, verbatim, and it is a closer match than expected.**
Methods, on frame skipping:

> *"the agent sees and selects actions on **every kth frame** instead of every
> frame, and **its last action is repeated on skipped frames**."*

That is axis 1 as `relatedwork.tex` states it, *"One decision is executed over
several control steps,"* in the source's own words. The paper even gives the
efficiency rationale our axis rests on:

> *"Because running the emulator forward for one step requires much less
> computation than **having the agent select an action**, this technique allows
> the agent to play roughly k times more games without significantly increasing
> the runtime."*

The expensive thing is the policy call, so spending fewer of them is the
saving. That is exactly why *when the policy runs* is a resource axis and not
a scheduling detail. There is a second instance too: the random baseline
*"chose a random action at 10 Hz which is every sixth frame, repeating its last
action on intervening frames."*

⚠️ **This file described DQN as "the origin of executing one decision over
several steps." That is wrong.** The sentence introducing the technique reads
*"**Following previous approaches** to playing Atari 2600 games, we also use a
simple frame-skipping technique"*, with a citation of its own. DQN adopts frame
skip, it does not invent it.

**The draft is unaffected and needs no edit.** `relatedwork.tex` says *"a
lineage running **from frame skip in RL** to action chunking"* and cites four
papers for the lineage. It names the technique as the starting point, not DQN
as its inventor. The error was in this file's summary, which is the more
dangerous place for it: a wrong provenance note is what a later editor would
have promoted into the prose.

**Bibliographic data, all from the PDF's own first page.** Nineteen authors,
where this entry previously stopped at ten and *"and others"*. Nature 518, and
the running head gives the rest: *"26 FEBRUARY 2015 | VOL 518 | NATURE | 529"*,
`doi:10.1038/nature14236`, received 10 July 2014, accepted 16 January 2015.

---

## Six-family revision, 2026-09-03: claims carried in from the mentor's audit

`relatedwork.tex` v3 adds the three families the mentor's project introduced
(VLA-Cache as baseline, guarded reuse and temporal fusion as candidates). The
new claims below are taken from `docs/literature_review.md` on the `tricks`
branch (audit dated 27 August 2026), which the mentor wrote against the
primary PDFs. Each row names the sentence in the draft and the line in that
document it rests on. Where the draft carries a number, the number is the
mentor's.

| # | claim in the draft | source in `docs/literature_review.md` |
|---|---|---|
| 21 | FlashVLA reports a 0.7-point average success decrease with training-free action reuse | §2: *"FlashVLA uses training-free action reuse as part of its acceleration pipeline, although the full method reports a 0.7-point average success decrease."* |
| 22 | SpecPrune-VLA separates coarse movement from precision-sensitive phases | §2: *"SpecPrune-VLA similarly distinguishes coarse movement from precision-sensitive manipulation phases."* The draft says "separates", not "avoids the decrease"; the source supports only the distinction. |
| 23 | VLA-Cache: static patches reused, dynamic and task-relevant recomputed, task relevance from language-model attention, history retained unlike token dropping | §5, first paragraph, and `docs/methodology.md` §8. |
| 24 | VLA-Cache on base OpenVLA across LIBERO, 75.0% dense vs 74.7%, 51.91 ms vs 31.83 ms | §5: *"the paper reports average success of 75.0% for dense inference and 74.7% for VLA-Cache, with latency changing from 51.91 ms to 31.83 ms."* Draft rounds to 31.8 / 51.9. |
| 25 | "nearly maintained rather than improved" | §5: *"a strong efficiency result with nearly maintained, not improved, average accuracy."* |
| 26 | guarded reuse skips a call only when image and recent trajectory are both stable | §4 and `docs/methodology.md` §7 (global and local signature gates, two-action cosine, translation floor, gripper sign, cap of one). |
| 27 | FlashVLA ties reuse to action similarity and visual-token stability | **PDF read 2026-09-03**, §1: *"a token-aware reuse mechanism that compares both action similarity and visual token stability to decide whether to skip computation and reuse the previous action."* Ablation: without token stability the model *"tends to reuse too aggressively"*. The earlier wording "action stability" was too narrow and is fixed. Also confirmed from the PDF: *"only a 0.7% drop in task success rate"* (row 21). |
| 28 | TTF-VLA raises base OpenVLA on LIBERO by four points at under 2% overhead, a denoising result | §6: *"TTF-VLA (AAAI 2026) reports that training-free temporal token fusion improves base OpenVLA LIBERO success from 68.4% to 72.4% with less than 2% runtime overhead. Its evaluated fusion is a temporal-denoising intervention rather than a speed mechanism."* |
| 29 | VLA-InfoEntropy selects which tokens to reuse by image and attention entropy | **PDF read 2026-09-03**, abstract: *"we use image entropy to quantify the grayscale distribution characteristics of each visual token and introduce attention entropy to capture the distribution of attention scores over task-related text"*; the reuse path is VLA-Cache's KV cache with a different token selector (§1, Fig. 1). Table: OpenVLA 75.0 → 76.4, 51.91 → 31.25 ms, matching the mentor's audit. Earlier wording "budgets cache reuse by image entropy" dropped the attention half. |
| 28a | TTF-VLA numbers | **PDF read 2026-09-03**, abstract: *"4.0 percentage points average on LIBERO (72.4% vs 68.4% baseline)"*; §Results: *"TTF introduces less than 2% additional runtime overhead"*. Their OpenVLA baseline is 68.4, not the 75.0 that VLA-Cache and VLA-InfoEntropy report, so the three papers do not share an evaluation setup. |
| 30 | VLA-IAP prunes tokens by interaction alignment | §6: *"IAprune ... is a token-pruning method"*, and the paper title in `main.bib` (*Training-Free Visual Token Pruning via Interaction Alignment*). v2 of the draft wrongly grouped it with cache reuse. |
| 31 | selecting patches for reuse is not itself new, and concurrent work is converging on coordinated cross-modal reuse | §6: *"A recent anonymous submission also studies coordinated cross-modal token reuse. Therefore, the paper should not claim that a shared mask alone is novel."* **The anonymous submission is no longer cited, 2026-09-03.** It is under review on OpenReview (R6d86jMO74), non-archival, with no public arXiv version (searched 2026-09-03), so a reviewer cannot verify it and IEEE has no reference format for it. The non-novelty admission it was carrying is now supported by rows 28--30, which are citable: TTF-VLA already fuses tokens temporally and VLA-InfoEntropy already selects reusable tokens by a computed signal. **What is and is not recovered.** The weak half of the mentor's point, that selecting patches by a computed signal is not new, is fully carried by rows 23, 28 and 29. The strong half, that someone already coordinates one mask across the denoising and cache paths, rests on the anonymous submission alone and no citable paper replaces it. The draft therefore states only the weak half. An earlier fix kept a "concurrent work is converging on coordinated reuse" clause with no citation; that was an uncited gesture at a paper we cannot cite and it was removed on the same day. The paper does not claim the shared mask is novel in either direction, because the contribution is the evaluation, not the mechanism. If the submission becomes public before camera-ready, restore the citation and the stronger sentence. |
| 32 | what we test: one mask driving denoising and cache reuse, contact-aware fallbacks, optimized dense baseline, paired episodes | §6, the four "defensible differentiators". |
| 33 | a speedup against eager attention overstates the gain | §"Exact inference optimization": *"Comparing only with an artificially slow eager implementation would inflate acceleration claims."* Stated in the draft as a fact about measurement, not as a count of papers, because no paper was checked for it. |
| 34 | positive only when speed and success both clear a preregistered gate | `docs/experiment_protocol.md`, "Claim gates" 1 and 2. |

### Sentences from v1 that were changed, and why

- *"VLA-Cache and VLA-Pruner attribute [poor transfer] to short action
  sequences"* (v2) reverted to v1's split: VLA-Cache attributes, VLA-Pruner
  reproduces. Row 5 above supports only reproduction for VLA-Pruner.
- *"Many also compare against an artificially slow eager baseline"* (v2)
  replaced, see row 33.
- The exclusion sentence lost "KV-cache compression". VLA-Cache reuses KV
  computation, so keeping that exclusion would contradict the section.
