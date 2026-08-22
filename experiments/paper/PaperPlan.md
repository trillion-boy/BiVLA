# Paper plan — ICRA, 6 pages + 2 of references

Two things: how to read a `.tex` file, and how long each section gets.

---

## 1. Reading `.tex` — what to look at and what to skip

In Overleaf the coloured parts are **markup**, the black parts are **text**.
"Read only the black" is *almost* right. The one exception is that some
commands' braces hold text and some hold labels.

**Braces that hold text you read:**

| you see | it renders as | read it as |
|---|---|---|
| `\emph{action repeat}` | *action repeat* | action repeat (italic) |
| `\textbf{Contributions.}` | **Contributions.** | Contributions. (bold) |
| `\texttt{move near}` | `move near` | move near (monospace) |
| `\footnote{Model time only...}` | ¹ + a note at the page bottom | the note |
| `$+15.6$` | +15.6 | +15.6 — `$…$` is just math mode |
| `$p < 10^{-4}$` | *p* < 10⁻⁴ | p less than ten to the minus four |

**Braces that hold labels, not text:**

| you see | it renders as | read it as |
|---|---|---|
| `\cite{openvla}` | [12] | "[ref]" — the key is a `.bib` name |
| `\ref{sec:limitations}` | V-B | "Sec. (number)" — filled in at compile |
| `\label{fig:grid}` | *(nothing)* | the anchor a `\ref` points at |

**Punctuation and spacing:**

| you see | it renders as |
|---|---|
| `---` | — (em dash) |
| `~` | a space that never breaks across a line |
| `\%` | % |
| `%` at line start | nothing — it is a comment |
| a blank line | a new paragraph |

So this:

```latex
Vision-language-action (VLA) policies are slow. In our own runs a single
forward pass costs UniVLA~\cite{univla} $2.80$~s.
```

reads as: *Vision-language-action (VLA) policies are slow. In our own runs a
single forward pass costs UniVLA [12] 2.80 s.*

Nothing in the current file changes the **font size**. `\textbf` is weight,
`\texttt` is a monospace face, `\emph` is italics — all normal in a paper, and
all handled by the IEEE template.

---

## 2. What an ICRA Introduction looks like

The reference paper (*Bag of Tricks for Inference-time Computation of LLM
Reasoning*) is the same genre as ours — re-measure existing training-free
methods, find that overlooked settings dominate the result — so its
Introduction is a good model. Its shape:

1. **Field context.** Why the problem matters. (~1 paragraph)
2. **The gap.** Existing results are mixed, methods are sensitive to
   hyperparameters nobody reports — *with a table showing exactly which
   settings differ across prior papers*. (~1 paragraph)
3. **What we do.** "In this study we investigate…" (~1 paragraph)
4. **Contributions**, as a numbered run-on paragraph with bold lead-ins —
   `1) Evaluation of Key Tricks: … 2) Combination of Techniques: …` — not a
   bulleted list.

Their Introduction is **688 words of prose**, measured from the PDF with
citations excluded, plus a table. Ours is **857** on the same basis, about 25%
longer, plus two footnotes. (Two earlier notes here were wrong: "about 900"
was from memory, and 698/834 counted citation markers as words.)

### Two things worth copying

**A table right after the introduction is normal — if it is about prior work,
not about our results.** Their Table 1 lists six prior methods against the
configuration knobs each one uses, showing that the field does not standardise
them. We have the identical move available and it is directly on thesis:

> **Table I — What prior work reports.** Rows: EfficientVLA, VLA-Cache,
> FastV, ShortGPT, VLA-Pruner (the training-free ones). Columns: candidate
> scope · selection constraint · keep value · **knob ablated?** · per-task
> split · **per-episode records** · **paired test**.

⚠️ **Corrected 2026-08-22, against the five PDFs.** This entry used to read
*"mostly empty cells, and each empty cell is one of the values we show changes
the answer,"* and to justify the table as evidence that the knob is
**unreported**. That is false. All five papers report their knob and **all
five ablate it** — FastV even publishes a K×R grid showing the same
interaction we find. Building the table from our notes would have shipped a
wrong claim.

What is uniformly empty is the last two columns: **no episode-level records
and no paired test in any of the five** (`TableI_Cells.md` §3, verified by
grep over the complete text of each). So the table's thesis moves from
*"they hide the configuration"* to *"the field is careful about configuration
and silent about uncertainty."* That is a smaller claim and a true one, and it
is the claim our method contribution actually rests on.

Full cell-by-cell provenance, with page numbers and quotes, plus two
reframings the evidence does support: **`paper/TableI_Cells.md`.**

**Contributions as a numbered paragraph, not `itemize`.** A four-item
`itemize` costs roughly eight lines of vertical space in two columns; the same
text as one paragraph with bold `1)`…`4)` lead-ins costs about five. At six
pages that difference is real. The current draft already uses the paragraph
form.

### One thing not to copy

That paper is single-column NeurIPS with 9 pages and unlimited references. We
have two columns and 6 pages. Everything has to be about a third shorter than
it feels like it should be.

---

## 3. Page budget

Six pages of content, two of references, IEEEtran two-column.

| section | pages | notes |
|---|---:|---|
| Title, abstract, Introduction | 0.75 | intro ≈ ¾ column; Fig. 1 takes the rest of page 1 |
| **Fig. 1** — the grid as a signed heat map | (in the above) | 7 conditions × 5 cells; the reader should see signs disagreeing within a row before reading a word |
| **Table I** — what prior work reports | 0.25 | see above |
| Related Work | **0.5** | three run-in paragraphs, **616 words of prose** against the model paper's **350** on the same basis (`RelatedWork_Plan.md` §0; the 429 this project used for months was a miscount). The 0.5 page rests on an unverified words-per-column estimate, so compile before cutting |
| Setup and protocol | 1.0 | backbones, benchmarks, the eight conditions, pairing, determinism, the correction family |
| Results | **2.25** | the three results; four tables. Takes the 0.25 freed from Related Work |
| What breaks (failure typing) | 0.5 | the one table with five buckets |
| Limitations and conclusion | 0.75 | GPU recording, third benchmark, single-cell mechanism |
| **total** | **6.0** | |
| References | 2.0 | ~50 entries fits comfortably |

**Results is the section that will overflow.** Four tables plus prose in two
pages is tight, so decide early which contrasts are figures and which are
tables — a heat map carries the grid far more cheaply than a 5×8 table of
numbers.

---

## 4. Where the Markdown tables end up

`Introduction.md` is a reading document: it compresses the whole paper into one
file, which is why it has seven tables. In the paper they scatter:

| table in `Introduction.md` | destination |
|---|---|
| the three axes | Intro body text (already prose in the `.tex`) |
| grid coverage 3 × 2 | Setup, **Table II** |
| eligible-window contrast (4 rows) | Results, **Table III** |
| foveation keep sweep | Results, **Table IV** |
| cross-cell comparisons | Results, **Table V** |
| failure typing (5 buckets) | Results, **Table VI** |
| "where each number comes from" | not in the paper — our own index |

Nothing is lost by keeping the Introduction table-free; the same content
appears once, in the section that owns it.

---

## 5. What changes when the mentor's five models land

Asked because the expansion tables for TurboVLA, CoTinyVLA, FLOWER, MiniVLA and
SmolVLA are still empty, and it was not obvious whether Introduction and
Related Work could be written before the runs come back. Checked against the
files rather than guessed.

**Related Work needs no change at all.** Its closing sentence says *"a complete
backbone $\times$ benchmark grid"* with **no numbers in it**, deliberately, so
it claims a method rather than a coverage record and survives any change of
scope. Three of the five models are cited exactly once, in the sentence about
compact VLAs building reductions in by design, and that claim rests on their
own papers' architecture descriptions, not on any result:

| model | what the draft asserts | source |
|---|---|---|
| FLOWER | prunes 30–50% of its VLM layers by design | *"we prune between 30% and 50% of the pretrained VLM's layers"* |
| SmolVLA | keeps only the first sixteen layers | *"We use only the first 16 layers of the LLM within the VLM"* |
| TurboVLA | drops the language model from the action path | *"without autoregressive action-token generation"* |

CoTinyVLA and MiniVLA do not appear in either section.

**The Introduction has exactly four places to update**, all reporting the size
of our own grid and none carrying an argument. ⚠️ *Corrected 2026-08-22: this
said two. The tex gained two more grid-size figures during the review passes.*

| where | what it says now |
|---|---|
| the setup paragraph | "three open backbones", "two SimplerEnv suites", "five of the six cells", "eight conditions", "$7{,}198$ episodes" |
| result 1, last sentence | "Across all five cells the contrast spans $2.1$ to $50.4$ points" |
| result 3, last sentence | "ten of the fourteen intervention conditions our two Fractal cells ran" |
| contribution 1 | "a $3 \times 2$ backbone-benchmark grid, five filled cells with eight conditions each" |

**Paragraph 2 needs no change and that is deliberate.** It says separating
method from configuration *"takes a grid that moves the backbone and the
benchmark independently."* That justifies **crossing the axes**, not the
number $3 \times 2$, so it stays true at $8 \times 3$ or at $2 \times 2$. The
smallest grid that crosses both axes is $2 \times 2$; ours is the largest we
could run. Do not write that $3 \times 2$ was necessary, because it was not.

**The three results are unaffected.** They come from our five cells. New cells
would add evidence rather than revise it, and the 45.9-point window contrast is
a within-cell result that more backbones cannot disturb.

**So the real question is scope, not data.** If the five models go into *this*
paper, those two blocks get new numbers and the grid becomes 8 × 3. If they are
a follow-up, nothing changes. Either way the two sections can be written,
reviewed and sent now, which is what the mentor asked for.

One thing to carry forward: `FiveModels_Read.md` already records that depth
pruning is only applicable to two of the five, since FLOWER and SmolVLA prune
layers as part of their architecture and TurboVLA has no language model in the
action path. So an 8 × 3 grid would not fill uniformly, and the Setup section
has to say so rather than implying a complete expansion.

---

## 6. The five-model expansion, and whether the Introduction can be written now

Asked 2026-08-22, when the mentor listed TurboVLA, CoTinyVLA, SmolVLA, FLOWER
and MiniVLA to be run on Bridge and Fractal. Answered from `FiveModels_Read.md`,
which read all five sources.

### The short answer

**Yes. Write it now.** All three results are within-cell or two-cell
comparisons, so no number of new backbones can revise them.

| result | scope | what new cells do to it |
|---|---|---|
| 1, the $45.9$-point window | one cell, OpenVLA/Fractal | nothing. New backbones cannot reach inside it |
| 2, the keep sweep | one cell, OpenVLA/Bridge | nothing |
| 3, the sign reversal | two cells, both Fractal | nothing |

New cells **add** evidence for the thesis rather than revising it. The claim is
that an effect *can* be a property of the configuration, and every backbone
added is another configuration.

### One correction to the model list

The list circulated as SmolVLA 4B. **SmolVLA is 450M**, from its own paper:
*"Our main model contains 450 million parameters."* A factor of nine. This
changes the framing, since the new models are 0.2--1B and our current grid is
4--8.5B, so the expansion is two clusters with a hole between 1B and 4B rather
than a scale sweep.

### Why the expansion will not give a clean $8 \times 3$ grid

`FiveModels_Read.md` §2 and §3, from the five sources:

| | depth pruning applies | SimplerEnv Bridge | SimplerEnv Fractal |
|---|:--:|:--:|:--:|
| TurboVLA | ✗ no LLM in the action path | ✗ | ✗ |
| CoTinyVLA | ✓ | ✗ | ✗ |
| MiniVLA | ✓ | ✓ our exact four tasks | ✗ |
| FLOWER | ✗ prunes 30--50\% by design | ✓ | ✓ |
| SmolVLA | ✗ first 16 layers only | ✗ | ✗ |

Two of five take depth pruning. Two have published Bridge numbers and one has
Fractal, so for the other three there is **no published figure to validate the
setup against on SimplerEnv**, which is the check that catches `unnorm_key` and
gripper errors before the intervention rows are spent. Action repeat is also
not comparable down a column, since native chunk lengths run 1, 1, 5 for ours
against 8, 20--50 and 50 for theirs.

**So the honest expanded grid is ragged**, and that raggedness collides with
contribution 4, which requires every cell to run the same conditions.

### The resolution, and it is not a fudge

The uniformity rule is a rule about **what may be compared**, not a claim that
our grid is complete. A cell that cannot run a condition gets reported as
inapplicable by architecture, with the reason. That is itself the paper's
point in stronger form: FLOWER and SmolVLA have built our depth intervention
into the architecture, and TurboVLA removed the language model outright, so
whether a reduction transfers has already become a question about architecture
rather than about a switch. `relatedwork.tex` already says this and cites all
three.

### Recommendation for this submission

**Keep this paper at the current grid and treat the five as a follow-up.**
Three reasons, in order of weight.

1. Three of the five have no SimplerEnv baseline to validate against, so those
   rows would carry a setup risk the current five cells do not.
2. Depth pruning, which carries results 1 and 3, applies to two of the five. An
   expansion that cannot run the headline intervention on most of its new rows
   buys coverage rather than evidence.
3. Six pages. `Setup` and `Results` are already the tight sections.

If the mentor wants them in, the expansion is **LIBERO-primary**, since all
five have LIBERO and only FLOWER has both SimplerEnv suites, and FLOWER is the
first to run because it is the only cell where a wrong setup gets caught.
Either way the four spots in §5 take new numbers and nothing else moves.
