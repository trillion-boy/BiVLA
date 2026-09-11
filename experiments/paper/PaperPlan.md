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
In our own runs a single call to the policy averaged $2.81$~s of model time
for UniVLA~\cite{univla}.
```

reads as: *In our own runs a single call to the policy averaged 2.81 s of
model time for UniVLA [12].*

(`\footnote{...}` would render as a superscript number plus a note at the page
bottom. Neither section uses one. See the header of `introduction.tex` for
why.)

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
citations excluded, plus a table. Ours is **972** on the same basis, about 41%
longer, with no table and no footnotes. (Earlier notes here said "about 900",
from memory, then 698/834, which counted citation markers as words, then 857,
before the review passes added the qualifiers that took it to 972.)

### Two things worth copying

**A table right after the introduction is normal — if it is about prior work,
not about our results.** Their Table 1 lists six prior methods against the
configuration knobs each one uses, showing that the field does not standardise
them. We built the identical move and then **dropped it on 2026-08-22**; the
draft, the three reasons and the revival instructions are in `tableI.tex`. The
record below is why it looked attractive:

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
| ~~**Table I** — what prior work reports~~ | ~~0.25~~ | **dropped 2026-08-22** (`tableI.tex`). Its 0.25 page returns to Results |
| Related Work | **0.5** | three run-in paragraphs, **616 words of prose** against the model paper's **350** on the same basis (`RelatedWork_Plan.md` §0; the 429 this project used for months was a miscount). The 0.5 page rests on an unverified words-per-column estimate, so compile before cutting |
| Setup and protocol | 1.0 | backbones, benchmarks, the eight conditions, pairing, determinism, the correction family |
| Results | **2.5** | the three results; four tables. Takes the 0.25 freed from Related Work and the 0.25 freed by dropping Table I, which the five-model rows will need |
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
| grid coverage | Setup, **Table II** (3 × 2 today, larger once the five land) |
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

**Related Work needed three changes, not none.** ⚠️ *Corrected 2026-08-22.*
This paragraph used to argue it needed no change, because its closing sentence
carried no numbers. That reasoning was wrong: the sentence said *"a **complete**
backbone $\times$ benchmark grid"*, and a word can be a coverage claim without
being a number. It contradicted the Introduction's *"five of the six cells"*
even before any expansion. See the "Related Work under the expansion" note at
the end of §6 for all three fixes. Three of the five models are cited exactly
once, in the sentence about
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
| the setup paragraph | "three open backbones", "two SimplerEnv suites", "five of the six cells", "eight conditions", "$6{,}910$ episodes" |
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

**Decided 2026-08-22: the five go into *this* paper.** So the four blocks above
get new numbers when the runs land, and both sections can still be written,
reviewed and sent now, which is what the mentor asked for. §6 works through
what that decision costs, and the note below §6 records the one methodological
sentence that had to be written in its final form rather than renumbered.

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

**So the honest expanded grid is ragged**, which is why contribution 4 was
rewritten. It used to require every cell to run the same conditions. It now
holds the condition set fixed *across the cells we compare*, which the ragged
grid satisfies.

### The resolution, and it is not a fudge

The uniformity rule is a rule about **what may be compared**, not a claim that
our grid is complete. A cell that cannot run a condition gets reported as
inapplicable by architecture, with the reason. That is itself the paper's
point in stronger form: FLOWER and SmolVLA have built our depth intervention
into the architecture, and TurboVLA removed the language model outright, so
whether a reduction transfers has already become a question about architecture
rather than about a switch. `relatedwork.tex` already says this and cites all
three.

### How to run the expansion

**Decided: the five go into this paper.** The recommendation here used to be to
keep them as a follow-up, and the reasons behind it are now risks to manage
rather than reasons to decline.

1. Three of the five have no SimplerEnv baseline to validate against, so those
   rows carry a setup risk the current five cells do not. **Run FLOWER first.**
   It is the only model with published numbers on both suites, so it is the
   only cell where a wrong setup gets caught before intervention rows are
   spent. MiniVLA is second, on our exact four Bridge tasks.
2. Depth pruning, which carries results 1 and 3, applies to two of the five.
   Those new rows add breadth on the other two axes and the inapplicability
   itself is a finding, but they do not add evidence to the headline result.
3. Six pages. `Setup` and `Results` are already the tight sections, so the new
   rows need to arrive as figure area rather than as prose.

**LIBERO is the natural primary for the expansion**, since all five have it and
only FLOWER has both SimplerEnv suites. Whatever the coverage ends up being,
the four spots in §5 take new numbers and nothing else moves.

### Related Work under the expansion

Checked 2026-08-22. **Three phrases needed changing, and one of them was
already contradicting the Introduction.**

**1) "a complete backbone $\times$ benchmark grid" was false today.** The
Introduction says we fill *five of the six cells*, so the two sections
disagreed about our own coverage, and the expanded grid will be far more
ragged since depth pruning applies to two of the five new models. It now reads
*"on both axes of a backbone $\times$ benchmark grid"*, which claims the
crossing rather than the coverage. That is the thing we actually do, and it
stays true at any grid size.

⚠️ This corrects §5 of this file, which argued Related Work needed no change
because its closing sentence carried no numbers. A word can be a coverage
claim without being a number.

**2) "its two recipes" and "The two rules"** presupposed the record holds
exactly two prescriptions for layer pruning. The Introduction dropped the same
presupposition earlier. Now *"its recipes"* and *"These rules"*.

**3) "the newest compact VLAs"** dates the sentence, and reads oddly once
FLOWER, SmolVLA and TurboVLA are our own experimental subjects rather than a
trend we observe. Now *"several recent compact VLAs"*.

**What needs no change, and why that is the point.** Related Work carries
**zero numbers**, confirmed by `audit_sections.py`, so nothing in it has to be
recomputed when the runs land. The three models cited for building reductions
in by design stay correct when we also evaluate them, since Related Work
describes them as prior art and Results measures them. And the axis framing in
paragraph 1 is defined by the resource each method spends, not by how many
backbones we hold, so new models slot into the existing three axes.

**One item still needs the mentor.** *"These rules have not been compared on a
robot policy"* is a negative claim about the whole literature. We searched and
found no counterexample, but absence of a counterexample in our reading is not
proof. Ask whether anyone knows of one.

## 7. What the per-episode files (2026-09-11) add, and what the paper now claims

Written after the full audit (`DataAudit_2026-09-11.md`). Before the
per-episode files the paper had aggregate rates only; now every table cell
has a paired test, a per-task breakdown, gate-firing counts and a
per-step latency that means the same thing on every harness.

### Claims that got stronger

1. Controls behave as controls. Action repeat k = 2 collapses on WidowX
   and LIBERO in every task at once (UniVLA 2 wins / 152 losses, p below
   1e-4) and is flat on Fractal. Foveation is slower everywhere (+2 to
   +17 % per step) and its drops sit in one or two tasks. Both were
   claims before; now they are paired, significant and localised.
2. The "same trick, different backbone, opposite sign" claim is now
   testable. Depth pruning: +9.5 on CogACT WidowX (p = 0.009) against
   significant drops on MiniVLA, SpatialVLA, SmolVLA and OpenVLA Spatial.
   Fusion: +8.0 CogACT WidowX and +6.8 OpenVLA Fractal against nothing
   elsewhere. Only 26 of 110 table cells move significantly.
3. Why guarded reuse does nothing is now known: the gates open on at most
   10 % of steps and on UniVLA, SpatialVLA WidowX and CronusVLA WidowX
   almost never. This turns "no effect" into a mechanism statement.
4. Avg. Steps is explained: every non-crashed failure runs to the step cap
   (21,495 of 21,495), so the column follows success. A reviewer can no
   longer read it as an independent efficiency signal.
5. Latency claims are on one footing. Per-step wall-clock reproduces
   from the episodes for every harness; the CSV cycle/policy columns do
   not (three definitions). Depth pruning is the only non-repeat family
   that lowers per-call latency monotonically; task-aware costs +7 to
   +20 % per call where attention is collected.

### Claims that got weaker or need a footnote

- UniVLA WidowX has no valid fusion result: task-aware crashed (81 OOM
  episodes), conservative-adaptive never engaged.
- CronusVLA WidowX guarded reuse is strict three times; CronusVLA fusion
  is one unknown setting three times.
- CogACT ran two fusion settings, not three (no attention path).
- SmolVLA latency is not comparable across its two implementations and
  five GPUs; its depth rows are not the Methods selector.
- Conservative-adaptive is "ineffective" on SpatialVLA and UniVLA (the
  mentor's word); report 0 and flag it.

### What goes where

- IV-B Results: Tables I and II with significance stars; one paragraph
  per family unpacking the settings (keep 20 vs 50, k, budget, presets,
  three fusion settings); the reversals across backbones and environments.
- IV-C Analysis: gate-firing rates, step-cap effect, action-repeat time
  not 1/k, task-aware cost, concentrated vs spread drops per task.
- V Conclusion: two significant gains only, controls collapse, reuse
  inert; limitations one seed, simulation only, SmolVLA mixed stack,
  CogACT no attention, three settings per family.
- Footnotes: the five items above, ready to insert or delete as the
  mentor's answers arrive.

### Still extractable, not yet used

- Per-task tables for a supplementary figure (2,286 rows in
  `experiments/data_audit_2026-09-11/per_task_success.csv`).
- Bootstrap CIs on step change (already in `paired_results.csv`).
- Reuse count per episode as a histogram for the reuse paragraph.
- Per-GPU SmolVLA latency, if SmolVLA latency is kept with a footnote.

## 8. Checklist for 2026-09-12

Drafted and in the branch (2026-09-11): IV-A `setup.tex` (mentor's section,
brackets for GPU, versions, calibration source), IV-B and IV-C
`results.tex`, Korean explanations `Setup_ko.md`, `Results_ko.md`.

Waiting on the mentor: the three reruns (UniVLA WidowX task-aware,
CronusVLA WidowX reuse presets, CronusVLA fusion settings), SmolVLA stack
decision, SmolVLA depth indices, GPU and versions for IV-A, CogACT
task-aware confirmation, OpenVLA Long conservative-adaptive.

To do next, in order:
1. Introduction: send the mentor the two bundles (five result-driven edits,
   three style edits) with before and after sentences; the results
   paragraph is rewritten around the three claims (sign not of the rule,
   sign not of the backbone, three distinct quantities plus gates).
2. Results: replace "We propose no setting on the strength of these cells"
   with the supported thesis sentence, and add the gate-record
   recommendation at the end of IV-C (both drafted in chat, not applied).
3. Section V Conclusion, Limitations, Future work: three conclusions, the
   three recommendations the Intro promises, limitations (one seed, one
   run, simulation only, SmolVLA mixed stack, CogACT no attention path,
   thresholds not tuned per backbone, peak memory not measured), future
   work (eligibility window from the August study, per-backbone threshold
   sweep, real robot).
4. Abstract: rewrite from the Overleaf three-method version.
5. Related Work: drop or restate the "preregistered margin that Section
   IV-A fixes" sentence, since IV-A uses the paired test instead.
6. Optional: significance stars in Tables I and II (one flag in the two
   generators).
7. When reruns arrive: copy folders into artifacts/results/mentor_2026-09-11/,
   rerun make_simpler_table.py, paired_analysis.py, update the three cells
   and shrink the "Cells that are not what their row says" paragraph.
