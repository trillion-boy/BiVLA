# How to write Related Work, measured against *Bag of Tricks*

*Every structural claim below was measured from the PDF, not remembered. The
self-check is §5.*

---

## 0. Correction, 2026-08-22 — the 429 was wrong, and it was the benchmark

This file said *Bag of Tricks*' Related Work is **429 words**, with paragraphs
of 156 / 99 / 114, and said it was measured from the PDF. It is not
reproducible. The section runs from the "2 Related Work" heading to
"3 Preliminares", it opens on its roadmap sentence and closes on "Unique to
this paper, we are the first to…", and measured between those two points it is:

| basis | words |
|---|---|
| prose only, citations removed | **350** |
| as rendered, citation markers counted | 359 |

The earlier count almost certainly came from a truncated span. A regex looking
for the next section matched a fragment of a table caption partway through
paragraph 1, and PDF extraction had also interleaved Figure 1's internal text
into that paragraph, so the section boundaries were wrong in both directions.

**This matters because 429 was the number every length decision in this project
was measured against.** On the honest basis the comparison is:

| section | *Bag of Tricks* (prose) | ours (prose) | ratio |
|---|---:|---:|---:|
| Introduction | 688 | 857 | 1.25x |
| Related Work | 350 | 616 | **1.76x** |

So Related Work is not 43% longer than the model paper, as this file implied.
It is **76% longer**. That is still not a defect on its own, since the page
budget rests on an unverified words-per-column estimate and only a real compile
settles it, but the comparison should be stated correctly wherever it appears.

---

## 1. What their Related Work actually is

Measured from the paper:

| | |
|---|---|
| total length | **350 words of prose**, 359 as rendered with its citation markers. See the correction box below |
| structure | one roadmap sentence + **three** paragraphs + one closing sentence |
| paragraph headings | bold run-in, no numbering: *Reasoning with LLMs.* / *Inference-Time Computation of LLM Reasoning.* / *Benchmarks of LLM Reasoning.* |
| paragraph lengths | **125 / 80 / 85 words** of prose |
| closing | 39 words, starting *"Unique to this paper, we are the first to…"* |
| **tables inside Related Work** | **none** — Table 1 sits in the Introduction, before this section |
| taxonomy | given as *"1) Prompt engineering – … 2) Post-training techniques – … 3) Search-based methods – …"* **inside a paragraph**, not as a list |

Two habits worth copying exactly:

**Each paragraph ends by positioning us, not by summarising them.** Their
first paragraph closes *"This work focuses on test-time computation…"*; the
third closes with the gap, *"most research has focused on task performance
rather than inference-time computation, leaving key optimization techniques
underexplored."* Every paragraph earns its place by ending on why it matters
to this paper.

**The roadmap sentence comes first.** One sentence naming the three
subsections before writing any of them. It costs a line and makes the section
skimmable.

---

## 2. The map onto our material

The correspondence is close enough that this is a restructuring job, not a
writing job.

| theirs | ours | source we already have |
|---|---|---|
| Reasoning with LLMs — the field, with a 1)2)3) taxonomy | **VLA policies and their inference cost** — why they are slow, and the 1)2)3) of what people reduce | `RelatedWork.md` §2.1, `Overview.md` opening |
| Inference-Time Computation — the specific methods measured | **Training-free efficiency interventions** — our three axes, with their published lineage | §2.2 (time), §2.3 (vision), §2.4 (compute) |
| Benchmarks of LLM Reasoning — the evaluation literature and its gap | **VLA evaluation practice** — SimplerEnv/LIBERO, the harness papers, and what the practice cannot answer | §2.6, plus vla-eval, StarVLA, the two surveys |
| *"Unique to this paper…"* | our positioning sentence | §2.6's conclusion + the five-paper crossing-cell pattern |

### Where §2.5 goes — it does not get a paragraph

`RelatedWork.md` §2.5 (the per-task split hiding inside four-task averages) is
**a finding, not background.** It already appears in the Introduction as
"fourteen of fifteen configurations," and the evidence for it belongs in
**Table I in the Introduction**, exactly where *Bag of Tricks* puts its Table 1.

In Related Work it becomes **one clause** at the end of the third paragraph
pointing forward. Giving it a fourth paragraph would both bloat the section and
put a result in the wrong place.

---

## 3. The budget, corrected

`PaperPlan.md` currently allots **0.75 pages** to Related Work. That is too
generous, and the correction is worth taking because Results is the section
the same document flags as likely to overflow.

- *Bag of Tricks* uses 429 words. We need slightly more — three axes to name
  rather than one method family, plus the evaluation-practice paragraph
  carrying more weight for us. Call it **450–550 words**.
- **Estimate, not a measurement:** IEEEtran two-column at 10pt runs roughly
  500–550 words per column, so 500 words is about **one column ≈ 0.5 page**.
  This should be checked against the real template rather than trusted.

**Proposal: Related Work 0.5 page, and give the freed 0.25 to Results.**

---

## 4. Was the mentor right that Overview + RelatedWork are enough?

**Yes — and more so now than when they said it.** Nothing found in the recent
reading invalidated the structure. What it did:

| | effect on the plan |
|---|---|
| VLA-Pruner, SpecPrune-VLA, VLA-IAP, vla-eval, SparseVLM, DeeR-VLA read | **strengthened §2.6** — "we are not aware of anyone" became a five-instance structural pattern |
| the five expansion models read | **added a paragraph** — three of them build our axes into the architecture |
| the configuration count | **fixed one number** — twelve → fourteen of fifteen |
| MoLe-VLA read | **removed one wrong citation** from a training-free list |
| the survey sweep | **added citations**, changed no claim |

So the honest status is: the mentor's judgement holds, the material got
*better*, and the remaining work is compression from **9,558 words to about
500** — a factor of **22**. That is the whole job. It is a large cut, but it is
a cut, not a search.

### What the cut actually removes

The appendices (A.1–A.7, ~3,800 words of per-paper notes) do not go into the
paper at all — they are our reading record and stay as one. Of the main
sections, §2.3 alone is 1,835 words and has to become roughly one sentence
plus a citation. That is the discipline the section needs.

---

## 5. Self-check — what I verified, and what I did not

The proposal above rests on claims about two documents. Here is the status of
each.

**Measured directly from the *Bag of Tricks* PDF:**

| claim | how |
|---|---|
| Related Work is 429 words | word count over the extracted section |
| three paragraphs, 156 / 99 / 114 words | measured between the run-in headings |
| closing sentence is 39 words and begins "Unique to this paper" | string match + count |
| no tables inside Related Work | `"Table"` absent from the section text |
| Table 1 precedes Related Work | character offsets: Table 1 at 3,832, Related Work at 7,712 |
| the taxonomy is 1)/2)/3) inside a paragraph | string match on all three markers |

**One correction made during checking.** My first regex for the numbered
taxonomy used an 80-character window and returned *False*. That was a false
negative from the window being too small, not a fact about the paper — the
markers are all present. Re-checked with a wider window and confirmed. Noting
it because an unchecked *False* there would have produced a wrong claim in
this document.

**Measured from our own files:**

| claim | how |
|---|---|
| `RelatedWork.md` is 9,558 words | word count |
| §2.3 is 1,835 words; appendices A.1–A.7 total ~3,800 | per-section counts |
| the 22× ratio | 9,558 / 429 = 22.3 |
| `PaperPlan.md` allots 0.75 pages | grep of the budget table |

**Explicitly not verified — treat as estimate:**

- *"IEEEtran two-column runs 500–550 words per column."* This is a rule of
  thumb, not something I measured against the actual template. Every page
  figure downstream of it (the 0.5-page proposal, the 0.25 page freed) inherits
  that uncertainty. Compile one column of real text before committing to the
  budget.

**Relayed, not verified:** that the mentor said Overview and RelatedWork could
be polished into Introduction and Related Work. That is your report of a
conversation; I have no independent access to it and have not treated it as
evidence for anything except what you should expect them to accept.
