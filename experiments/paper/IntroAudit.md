# Introduction, every sentence against every defect class

Written 2026-08-22 after four review passes each found a *new kind* of
problem, which meant the passes were sampling rather than checking. The fix is
to fix the method, not to run a fifth sample. This file enumerates the defect
classes first, then runs all of them over all forty sentences.

**If a future pass finds a defect whose class is not in section 1, add the
class here and re-run section 2. That is the only way this converges.**

---

## 1. The twelve defect classes

| | class | what it looks like | how it was found before |
|---|---|---|---|
| A | numeric | a figure that does not match `results/` | UniVLA $2.80$ vs $2.81$~s |
| B | attribution | a claim about another paper that the paper does not make | OFT's Hz attributed to "its control tasks" |
| C | provenance | a claim with no source anywhere in the repository | the OFT figures, before the PDF arrived |
| D | quantifier | every, all, none, often, almost, without a count behind it | "often properties of the configuration" |
| E | untested inference | a comparative claim asserted without the test we demand of others | "the backbone matters more than the benchmark", Fisher $p = 0.40$ |
| F | self-contradiction | a claim our own data refutes elsewhere in the paper | "what saves compute is not what changes success", refuted by action repetition |
| G | strawman | a position attributed to nobody in particular | "a gain attributed to compression" |
| H | presupposition | a phrase smuggling in an unstated claim | "the **single** configuration", "the **two** prescriptions on record" |
| I | soft vagueness | a word hiding a weak spot | "reproducible", "a spread no wider than" |
| J | scope overreach | one case stated as a general rule | "needs a grid" |
| K | logic | dangling antecedent, ambiguous pronoun, non-sequitur | "the same hiding", "we chose **them**" |
| L | promise | something committed to that Setup or Results must deliver | per-episode records, determinism check |

---

## 2. All forty sentences

Numbers verified against `results/` through the same `paired()` the report
uses, not against `Report.md`, since the two could be wrong together.

| # | claim | classes checked | verdict |
|---|---|---|---|
| 1 | VLAs becoming capable of general-purpose control, running one is expensive | D, J | fine, and it is the mentor's abstract wording |
| 2 | $2.81$ / $0.90$~s over 96 baseline WidowX-Bridge episodes | A, C | recomputed $2811.5$ / $902.1$~ms. Benchmark named after the footnote was cut |
| 3 | OFT at $3$ to $5$~Hz, below $25$ to $50$ | A, B, C | **PDF p1**, *"too slow (3-5 Hz) for high-frequency control (25-50+ Hz)"*, and p13 attributes the requirement to real-time deployment |
| 4 | a literature has grown up around this | D | fine |
| 5 | three mechanisms, four citations | B | each mechanism is covered. Layers by ShortGPT, tokens by FastV and VLA-Cache, action holding by EfficientVLA's cache interval |
| 6 | layer skipping also pursued with training, out of scope | B | MoLe-VLA, correct |
| 7 | such papers share one form of claim | D | was "converge on", which asserted all of them do |
| 8 | the claim template, and what separating takes | H, J | was "the **single** configuration", which VLA-Pruner refutes with 2 backbones x 4 suites. "needs" softened to "takes" |
| 9 | the premise is what we test | — | fine |
| 10 | three backbones, two suites | A | correct |
| 11 | chose these three because each spends a different resource | K | was "chose **them**", which attached to the suites |
| 12 | the three axes | — | fine |
| 13 | five of six cells, eight conditions each, $7{,}198$ episodes | A | **counted per cell: 8, 8, 8, 8, 8.** Episodes $7198$ |
| 14 | pairing and McNemar | A, K | was *"we count only the episodes whose outcome flipped"*, which misdescribes the delta. The delta is over every paired episode |
| 15 | three results, same shape | — | fine |
| 16 | the factor a result would be credited to is not the one that moved it | G | was *"what a reported number was credited to"*, which reads as an accusation when all three results are our own measurements |
| 17 | heading, $45.9$ points, and what was held fixed | A, H | $15.6 - (-30.4) = 45.9$. Layer *count* held fixed, not identities, which is right since BI recalibrates per run |
| 18 | $+15.6$ to $-30.4$, compute $-10.6$ to $-11.9\%$ | A, I | all four recomputed. Dropped *"a spread no wider than repeating the same measurement produces"*, since the only re-runs are on another backbone |
| 19 | the ranking criterion never changed | A | BI in both, correct |
| 20 | the prescriptions on record disagree | H | was "the **two** prescriptions", which claims the record holds exactly two |
| 21 | ShortGPT and EfficientVLA constrain nothing, Gromov keeps the final layer | B | quoted from both PDFs in `TableI_Cells.md` §2 and §4(d) |
| 22 | a restriction an implementation can add without recording it, and ours did | J | was *"a guard an implementer would add"*, speculation about other people. `--depth-min-layer` is ours |
| 23 | the losing setting deletes the final layer | A | `window875` is L28--31 of 32, so L31 goes |
| 24 | span $2.1$ to $50.4$ across five cells | A | $2.1 / 5.2 / 6.2 / 45.9 / 50.4$, checked by `verify_all` |
| 25 | heading, and the keep sweep | A, F | **was *"what saves compute is not what changes success"*, refuted by our own action repetition at $-81.2$ while saving compute.** Sweep figures checked |
| 26 | compute moved $-3.1\%$ to $+2.7\%$ | A | recomputed per cell. $-3.1$ is SpatialVLA/Fractal blur, $+2.7$ is SpatialVLA/Bridge log-polar |
| 27 | the gain is largest where only the round trip is left | I, J | was *"the gain is reproducible and tracks the log-polar round trip"*. Report 3.4 records the foveation path moving per task across sessions, and "tracks" implies a link the sweep does not isolate |
| 28 | heading, and $+15.6$ against $-17.8$ | A, E | **was the axis-ranking heading, which fails at Fisher $p = 0.40$. See `AxisClaim.md`.** The reversal is tested end to end |
| 29 | averaged over the two cells it looks like nothing | A, K | $(+15.6 - 17.8)/2 = -1.1$. Was *"the same hiding"* with no antecedent |
| 30 | pick coke can degrades less, almost every intervention | D | **7 of 8 recomputed**, in `PerTaskRows.md` |
| 31 | contribution 1 | A, L | figures match. Records release is a promise |
| 32 | contribution 2, a reported effect **can be** a property of the configuration | D | was "**often** properties of", a frequency we cannot count |
| 33 | contribution 3 | A, F, I | **was *"the intervention with the largest effect"*, which is action repetition and does save compute.** Also dropped *"compute saving we hold fixed"*, since we observed it rather than held it |
| 34 | contribution 4, the procedure | L | *"grid uniformity rule"* was jargon used once. Now says what it is |
| 35 | five methods sweep their configuration, none tests significance | D | grep over all five full texts, `TableI_Cells.md` §3. `McNemar` 0, `paired` 0, `std` 0, `seeds` 0 |
| 36 | we propose no method, no general failure claim | — | fine |
| 37 | every change measured against its own cell's baseline | **F** | **was *"every contrast above is paired inside its own cell"*, which is false.** The cross-cell comparison shares no episodes, which is why it uses Fisher and not a paired test |
| 38 | four cells above, one $4.2$ below | A, I | `Report_EN.md` §3.8(c)① gives $+14.6$, $+3.7/+15.5$, $-4.2$, $+4.3/+13.7$, $+11.4$ |
| 39 | our claim is narrower | — | fine |
| 40 | as results are reported now, the evidence does not separate the two | D, J | hedged by the opening clause, and it is the thesis |

---

## 3. What is still open

**Two promises Setup and Results must keep**, both from contribution 4. The
determinism check, and the requirement that every cell run the same
conditions. Section 2 row 13 confirms the second is true of the data, so Setup
only has to state it.

**One claim that lives outside the paper.** Contribution 1 promises
per-episode records. Nothing in the repository publishes them yet.

**Nothing in the Introduction is unsourced.** Every figure is recomputed from
`results/`, and every claim about another paper is quoted in
`TableI_Cells.md`, `PerTaskRows.md` or `AxisClaim.md`.
