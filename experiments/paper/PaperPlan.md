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

Their Introduction is **698 rendered words**, measured from the PDF, plus a
table. (An earlier note here said "about 900" from memory; 698 is the count.)
Ours is **834 body words**, about 20% longer, plus two footnotes.

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
| Related Work | **0.5** | three run-in paragraphs, **614 words** as written, against *Bag of Tricks*' 429 (`RelatedWork_Plan.md`). The 0.5 page rests on an unverified words-per-column estimate, so compile before cutting |
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

**The Introduction has exactly two places to update**, both reporting the size
of our own grid and neither carrying an argument:

| where | what it says now |
|---|---|
| the setup paragraph | "three open backbones", "two SimplerEnv suites", "Five of the six cells", "$7{,}198$ episodes" |
| contribution 1 | "a $3 \times 2$ backbone-benchmark grid, with five filled cells, eight conditions each, $7{,}198$ episodes" |

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
