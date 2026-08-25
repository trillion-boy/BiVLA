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
| M | sampled verification | a check that sampled instead of enumerating, so the verdict itself was wrong | "7 of 8" for the per-task split, which had tested 4 of 7 conditions per cell |
| O | cross-section contradiction | a claim one section makes that another section of our own paper denies | "foveation is not an efficiency technique" against Related Work's "methods that foveate inside the encoder shed tokens" |
| N | heading unsupported by its own paragraph | a result heading whose claim the paragraph beneath it cannot establish | "compute saved does not predict success" over a paragraph where compute never varies |

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
| 13 | five of six cells, eight conditions each, campaign $6{,}910$ episodes | A | **counted per cell: 8, 8, 8, 8, 8.** Grid $= 3\times96\times8 + 2\times135\times8 = 4464$, plus 2,446 control and diagnostic episodes. **Two earlier answers were wrong. $7{,}198$ counts 384 LatentSaccade episodes from June, a different study. $4{,}464$ is the grid alone and leaves out result 1's own losing arm, which is a control run. See `EpisodeCounts.md` §5 and §6.** |
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

---

## 4. Pass five, 2026-08-22, and one new defect class

Run after the previous "exhaustive" pass, on the rule in the header: a new
kind of defect means a new class, added here, and the table re-read against
it.

**Class M, sampled verification.** A check that sampled instead of
enumerating, so the verdict column itself was wrong. Row 30 claimed "7 of 8"
for the per-task split, but the script behind it tested 4 of the 7
intervention conditions per cell. The full enumeration is 14 comparisons,
with **ten ahead, one tie, three behind**, and the three breaks include
`depth pruning 1` on both backbones. "Almost every" was an exaggeration on
the true count. The sentence now carries the count itself, which needs no
disclosure because nothing is excluded from it.

The other findings of this pass, by class:

| sentence | class | defect | fix |
|---|---|---|---|
| 3 | B | "OFT **measures** ... at 3 to 5 Hz". Its own tables measure 4.2 and 1.8 Hz. The 3 to 5 figure is one it states citing prior work | "reports" |
| 18 | A/K | "the compute **saved** stayed between $-10.6$ and $-11.9\%$" says we saved negative amounts | "stayed 10.6 to 11.9\% below baseline" |
| 22 | A | "a restriction an implementation can add **without recording it**, and ours did" is false about ours, which is a flag with help text | "can add on its own, and ours did" |
| 26 | A | the $-3.1$ to $+2.7\%$ range came from the grid foveation cells but sat next to the sweep, reading as the sweep's. The sweep's own runs sit within $0.6\%$ | scoped to "across every foveation condition we ran, within $3.1\%$" |
| 28/29 | style | "the same ... the same ... the same" three times in two sentences | the intervention is named instead |
| 30 | M | see class M above | the count |
| 6, 13, 31, 34, 37 | style | passives and noun-phrase contributions | active voice, "we run ... and we release" |

Sweep compute, measured for row 26: `keep10` $+0.3\%$, `keep40` $-0.6\%$,
`keep100` $-0.4\%$ against the 96-episode baseline mean of $517.7$~ms.
`keep20` has no millisecond records of its own, since that cell is the
imported foveation campaign.

---

## 5. Pass six, 2026-08-22, fresh read

**A verdict in section 2 was itself wrong.** Row 5 claimed the action-holding
mechanism was covered by "EfficientVLA's cache interval". Re-reading the PDF,
that cache reuses intermediate attention and MLP features across
\emph{denoising} steps inside one action generation, and VLA-Cache reuses
token computation across frames. Neither holds a predicted action across
control steps, and nothing in the four-paper cite list does. The sentence now
lists the three mechanisms those papers actually use, with a citation on each,
and the chunk-execution lineage stays where Related Work already cites it
correctly.

| sentence | class | defect | fix |
|---|---|---|---|
| 2 | B | "a single forward pass averaged $2.81$~s". One autoregressive call runs one forward pass per generated token, and the harness measures ms per call | "a single call to the policy" |
| 5 | B | action-holding attributed to four papers, none of which does it | per-mechanism citations, temporal caching named instead |
| 25 | J | the keep sweep ran in one cell and the sentence did not say so | "In one cell we swept" |
| 38 | K | "the fifth is $4.2$ points below, which is the wrong direction for a broken setup" anchors the clause to the low cell, and a low cell is exactly what a broken setup produces | "A broken setup would push all five down." |

**Closed rather than changed.** Sentence 35's grep never covered plain
statistics vocabulary, so "none tests a difference for significance" was
verified against `McNemar`/`paired`/`std` but not against `significant`,
`p-value` or `t-test`. Re-grepped all five PDFs with the wider set. Every hit
is colloquial or citation noise, so the sentence stands, and `TableI_Cells.md`
§3 records the term list.

**Checked and kept.** "Re-measured" stays. All three intervention families
have prior proposals, Related Work frames each lineage with its own
citations, and the framing is the paper's premise. The mentor's abstract uses
"study three representative approaches" for the same content.

---

## 6. The published baseline figures, author-confirmed 2026-08-22

The one verification this repository could not repeat is closed. The author
confirms **OpenVLA/Bridge 1.0\%, SpatialVLA/Bridge 34.4\%, UniVLA/Bridge
69.8\%** against the papers, and adds one fact the hand-check had not
recorded: **34.4\% is SpatialVLA's zero-shot figure, and its fine-tuned
figure is 42.7\%.**

Which comparator is honest depends on which checkpoint we ran. The campaign
scripts pin it: `run_spatialvla_fractal_grid.sh` and
`run_spatialvla_foveation.sh` both default to
`IPEC-COMMUNITY/spatialvla-4b-224-pt`, the zero-shot release, and no script
or config in the repository loads the mix or fine-tuned variant. So our
30.2\% is a zero-shot evaluation, 34.4\% is the like-for-like published
figure, and the $-4.2$ comparison stands.

The Introduction now says "the figures published **for the checkpoints we
run**" instead of "the published figures", so a reader who knows the 42.7
cannot read the sentence as comparing against the wrong number.

**For Setup or Results**: when the baseline-comparison table appears, print
the checkpoint id next to each published figure and label SpatialVLA's
comparator zero-shot, with the fine-tuned 42.7 in a note. That is the full
answer to the reviewer who asks.


---

## 7. Pass seven, 2026-08-22, raised by the author

**Class N, a heading its own paragraph cannot support.** Result 2 was headed
*"Compute saved does not predict what happens to success."* Foveation saves
nothing, which the paragraph itself says at within $3.1\%$ of baseline, so
compute never varies inside that paragraph and there is no variation for a
"does not predict" claim to rest on. The heading also invited the reader to
expect that foveation had saved something.

The finding the paragraph actually establishes is that the gain runs opposite
to how much is discarded, peaking at $100\%$ keep. That is a
credit-misattribution result, which is exactly what the transition sentence
two paragraphs earlier promises, so the heading is now **"What foveation
discards is not what makes it help."**

The paragraph also now states the no-saving result outright, *"so the
discarding buys nothing,"* instead of leaving it as an aside explaining a
number. It is a finding about the intervention, not a caveat.

**Contribution 3 was checked and kept.** *"Evidence that compute saved does not
predict success change"* rests on two results, not one. Foveation saves nothing
and gains up to $30.2$ points, and depth pruning holds its saving near $11\%$
while success swings from $+15.6$ to $-30.4$. Across both, the compute figure
carries no information about the success figure, which is what the
contribution claims.

**Earlier headings for this paragraph, and why each went.** *"A gain
attributed to compression does not come from compression"* attributed a
position to nobody in particular (class G). *"Our visual intervention helps
most when it discards nothing"* was accurate but sat below the altitude of the
other two headings. *"What saves compute is not what changes success"* is
refuted by our own action repetition, which saves compute and costs UniVLA
$81.2$ points (class F). The current heading is accurate, at altitude, and
about our own intervention rather than someone's claim.


---

## 8. Pass eight, 2026-08-22, raised by the author reading paragraph 6

**Class O, a contradiction between our own two sections.** Result 2 ended with
*"which makes **foveation** an input transformation rather than an efficiency
technique."* That is a claim about foveation in general. Related Work, updated
the same day from the Look-Focus-Act and Segment This Thing PDFs, now says
*"Methods that foveate inside the encoder give distant patches a coarser
resolution and **so shed tokens**."*

**Both sentences cannot be true.** A reviewer reading them together finds our
paper asserting that foveation both is and is not a way to spend less.

The author found it by reading their own draft and glossing sentence 3 as
*"other papers' efficiency framing is a fiction."* That gloss is exactly the
misreading the sentence invited, and it would be unfair as well as wrong,
since both papers really do reduce token counts.

### What changed

| before | after |
|---|---|
| "Across every foveation condition we ran, measured compute stayed within $3.1\%$ of baseline, since neither the image size nor the token count changes, so the discarding buys nothing." | "Training-free foveation edits pixels rather than the tokenizer, so the encoder sees its usual patch grid and the discarding buys no compute. Measured compute stayed within $3.1\%$ of baseline in every condition we ran." |
| "which makes **foveation** an input transformation rather than an efficiency technique" | "so **in this form** foveation is an input transformation and not a way to spend less" |

The first now gives the mechanism, which pre-empts *"why not just skip the
resize and shed tokens"*, and scopes the claim in the same clause. The second
scopes the definitional claim with three words.

The heading stays. *"What foveation discards is not what makes it help"* is
about the discarding, and the body establishes which kind of foveation before
any general claim is made.

### The rule this leaves behind

**Every claim about an intervention in general has to survive Related Work.**
The two sections were written weeks apart and checked separately, which is how
this survived seven passes. When Related Work gains a sentence about what a
method family can do, re-read the results for a sentence that denies it.


---

## 9. Pass nine, 2026-08-22, three questions from the author

**1. "3.1\% of what?"** The word *compute* never said what was measured.
It is `model_ms_per_infer`, milliseconds of model time per policy call, read
from the episode records by `discover_cost`. That matters, because wall clock
carries run-to-run noise where a FLOP count would not, and a reader who
assumed FLOPs would read a $3.1\%$ band as a real saving rather than as
nothing. Both results now say **model time per call**, so they agree with each
other and with what the files hold.

**2. "Is it true that editing pixels cannot save compute?"** Not in general,
and the sentence overstated. A backbone with a dynamic-resolution encoder
turns fewer pixels into fewer tokens, so a pixel-space edit *would* save there.
What is true is about **our** backbones: OpenVLA, SpatialVLA and UniVLA each
tokenize a fixed patch grid, so whatever image they are handed becomes the same
number of tokens. The sentence now says that, and it also answers in the paper
the question that prompted it, *why not skip the resize and shed tokens*.

**3. "Is the heading too aggressive?"** The concern was that *"what foveation
discards is not what makes it help"* reads as calling the whole premise false
when in fact retraining makes it work.

The heading is kept, and it is accurate. It is about the **discarding** being
the cause of the **gain**, not about efficiency, and our sweep shows the gain
peaking where nothing is discarded. It also does not conflict with the two
papers we cite, whose benefit comes from where the gaze looks rather than from
what is thrown away.

What was fixed instead is the last sentence, which said *"in this form"* and
left the reader to guess which other form exists. It now says **"foveation
without retraining"**, which names the condition and concedes in three words
that retraining changes the answer. That is where the perceived aggression
lived, since a naked *"is not a way to spend less"* sounds like a verdict on
foveation rather than on the training-free case.


---

## 10. Pass ten, 2026-08-22, two reviewer objections raised on the draft

### Objection 1, and it was a real gap

*"You said the token count cannot change. If the work is identical, why does
the time move at all? Is the log-polar warp inside the timed call?"*

**It is not**, and the paper did not say so. `eval.py` lines 558--571:

```python
policy_image = apply_foveation(image, args, fov_gaze)   # warp happens here
_t_model = time.time()                                  # timer starts after
raw_actions, env_actions = model.step(policy_image, ...)
model_time += time.time() - _t_model
```

So `model_ms_per_infer` measures `model.step` alone. The residual band is
wall-clock variance, and re-running a condition unchanged moves a mean by
$2.4\%$ on SpatialVLA/Fractal and $0.4\%$ on UniVLA/Bridge
(`EpisodeCounts.md` §4), which brackets the $3.1\%$.

The sentence now says **"which excludes the warp itself"**, four words that
answer the objection before it is raised. They also make the figure the
**charitable** reading for foveation, since counting the warp would make it
look worse rather than better, and a reviewer who notices that sees a
conservative measurement rather than a flattering one.

**Not adopted:** tightening `and` into `so`. The suggestion was to make the
$3.1\%$ follow causally from the fixed token count. It does not. A fixed token
count implies the *work* is identical, and the $3.1\%$ is what the clock does
on identical work, so a causal connective would claim the wrong thing. The two
facts now sit in two sentences instead.

### Objection 2, and the reading is correct

*"'Foveation without retraining' concedes that retraining it into a
token-dropping scheme would make it a real efficiency technique. Is that
intended?"*

**Yes, and it is the position the paper should hold.** Look-Focus-Act and
Segment This Thing demonstrate exactly that, and Related Work now says so with
their citations. The alternative stance, that foveation is bad regardless of
training, is not something our grid can support and is contradicted by two
papers we cite.

So the concession is not a retreat. It is the boundary that makes the claim
defensible, and it costs nothing, because the paper's subject is the
training-free case throughout.


---

## 11. Pass eleven, 2026-08-22, four reviewer objections

### 1. The GPU vanished with the footnotes

Removing both footnotes took the card assignment with them, so $2.81$~s had no
scale. Naming the card in an Introduction is normal, `Hardware.md` §8 measured
several comparable papers doing it to make a latency figure legible, and here
it **strengthens** the sentence, since UniVLA is three times slower than
SpatialVLA while running on the **faster** of the two cards. Now reads *"for
UniVLA on an L4 and $0.90$~s for SpatialVLA on a T4."*

### 2. "This is your bug, not the field's configuration problem"

The sharpest objection raised so far. ShortGPT and EfficientVLA specify
*constrain nothing*; we added a deep-end restriction; the losing arm deletes
the final layer. So the $45.9$ points look like the distance between two of
our own wrong implementations rather than a fact about the literature.

**Our own control runs answer it, and the paper was not using them.**

| setting | eligible | layers removed | delta |
|---|---|---|---:|
| grid `depth_prune4` | back half, L16--31 | **[17, 23, 25, 27]** | **$+15.6$** |
| `window25` | back three quarters, L8--31 | **[17, 23, 25, 27]** | **$+15.6$** |
| `window875` | last eighth, L28--31 | [28, 29, 30, 31] | $-30.4$ |

**Widening the window changes nothing.** Block Influence already ranks those
four highest, so the restriction is not binding at the winning setting and the
arm behaves as an unconstrained implementation would. It binds only at
$0.875$, where four candidates remain for four removals, the ranking is
disabled, and the final layer goes because nothing else is left.

Three sentences now say this. The reframing matters: *"and ours did"* stops
reading as a confession of deviation and starts reading as the dial being
described, since the paper now shows the dial doing nothing over most of its
range.

**Not adopted:** packaging the restriction as *"open-source implementations
commonly do this."* `TableI_Cells.md` §4(b) records that we have no evidence
about anyone else's implementation, and inventing that evidence to soften our
own admission is the failure this audit exists to prevent.

### 3 and 4. Two readability fixes

*"With the control and diagnostic runs the results below rest on"* wedged a
relative clause into the middle of a sentence, and *diagnostic runs* was a term
the paper never defines. Now *"Counting the control runs and sweeps the results
below rest on."*

*"stayed $10.6$ to $11.9\%$ below **it**"* made the reader resolve a pronoun
back to *the same baseline*. Now *"below baseline."*

---

## 12. Pass twelve, 2026-08-23: every negative claim, under the Related Work rule

Run because the same sweep on `relatedwork.tex` found two claims that had
survived every earlier pass, and both failed the rule `RelatedWork_Sources.md`
claim 12 had already written down: *a search that finds nothing is evidence
about the search, not about the world.* If two got through there, the
Introduction had to be swept the same way.

**The rule, stated once.** A claim of the form *"nobody does X"* is safe only
when the set it quantifies over is named and small enough to enumerate. *"Of
the five methods we survey, none reports X"* is enumerable and defensible.
*"No prior work reports X"* is neither, and one citation ends it.

### Every negative claim in the section, and its verdict

| # | claim | scope | verdict |
|---|---|---|---|
| 1 | *"an implementation can add without any specification calling for it"* | all specifications | ❌ **refuted by our own citation two sentences earlier** |
| 2 | *"published SimplerEnv comparisons print the same split without discussing it"* | all published comparisons | ❌ unscoped plural, evidence covers two named papers |
| 3 | *"none publishes per-episode outcomes or tests a difference for significance"* | the five we survey | ⚠️ scope fine, verb overreaches |
| 4 | *"The methods we re-measure rank every layer and constrain nothing"* | two named papers | ⚠️ true of the candidate set, false as written |
| 5 | *"Of the five training-free methods we survey ... every one sweeps its own configuration"* | the five we survey | ✅ enumerable, and enumerated |
| 6 | *"a pixel-space edit cannot change how many tokens they process"* | our three backbones | ✅ scoped to models we run |
| 7 | *"We propose no new efficiency method"* | us | ✅ |
| 8 | *"no result depends on our absolute rates matching anyone else's"* | us | ✅ |
| 9 | *"the evidence does not separate a property of the method from a property of the configuration"* | current reporting practice | ✅ **kept** — see below |

### Number 1 was the real defect, and it is class F, not class D

The draft said, of confining layer removal to the deep end, that it is *"a
restriction an implementation can add without any specification calling for
it."* Two sentences earlier the same paragraph cites Gromov et al., **a
specification that calls for exactly that** — a contiguous block at the deep
end. The paragraph refuted itself, in the space of three sentences, using a
citation we put there ourselves.

`TableI_Cells.md` §4(b) has the accurate scope and always did: *"Both layer
papers specify all layers, rank by BI, cut the lowest n, no spacing rule.
Neither says anything about restricting the candidate set"* — and *both* there
means ShortGPT and EfficientVLA, the two we implement. Gromov is the third and
is not among them.

Now: *"a restriction neither method we implement asks for, and ours added it
anyway."* This is not merely safer, it is the more interesting sentence. Our
implementation drifted toward a **competing** published prescription without
anyone choosing it, which is the paper's thesis happening to the authors.

### Numbers 2, 3 and 4 were scope and verb, not substance

**2.** `PerTaskRows.md` §3 ends on the defensible form: *"neither paper remarks
on the split."* Two papers, both read, both tabulated row by row in that file.
The Introduction had inflated this to a plural generic. Now *"the two published
SimplerEnv comparisons we cite ... without remarking on it"*, which is what we
checked and uses that file's own verb.

**3.** *"publishes"* invites a reviewer to look at a GitHub repository. Our
evidence is two greps over the five PDFs (`TableI_Cells.md` §3, both term
sets). *"reports"* is a claim about the paper, which is what we read.

**4.** *"constrain nothing"* reads as *no constraint whatsoever*, and both
methods constrain n, the number of layers removed. `relatedwork.tex` already
said *"constrains nothing further"*; the Introduction had dropped the word.
Restored, and the two sections now agree.

### Number 9 stays, and this is a decision rather than an oversight

*"As results are reported now, the evidence does not separate a property of the
method from a property of the configuration it was measured in."* This is a
universal, and it is the paper's thesis. Three things keep it:

1. **It is conditioned on reporting practice**, not on the existence of a
   paper. The refuting move is not "here is one counterexample" but "here is
   how the field reports," which is the argument we want.
2. **Contribution 4 supplies the enumerable version** immediately above it, so
   a reader reaches the universal already holding the count.
3. **Two independent sources say the same of the field** — vla-eval on
   undocumented evaluation pitfalls and StarVLA on fragmented protocols, both
   cited in Related Work.

Softening it to *"the evidence we surveyed"* would leave the paper with no
thesis. A claim can be universal when it is the conclusion of an argument. It
cannot be universal when it is a substitute for having read something.

### What this pass leaves behind

Every negative claim in either section now names the set it quantifies over,
except number 9, which names the practice instead and is argued for rather
than asserted. Three of the four defects had a correct, scoped version already
written down in a provenance file. The draft was not short of evidence. It was
losing the scope on the way from the provenance file into the prose, which is
a copy-edit failure mode and not a research one, and it is worth checking for
directly the next time a sentence is tightened.

---

## 13. Pass thirteen, 2026-08-23: prose only, no claim changed

The section is past the stage where checks find defects, so this pass read for
rhythm, reference and garden paths instead. Twelve edits, none of them touching
a number, a citation or a claim. Recorded because several of them fix things a
reader trips over silently, and a later editor should not undo them.

### Two garden paths, both from a missing "that"

| was | why it stalls |
|---|---|
| *"the control runs and sweeps the results below rest on"* | `sweeps` reads as a verb and `the results` as its object. The reader gets to *"sweeps the results"* before backing up |
| *"how much of the observation foveation keeps"* | `observation foveation` reads as a compound noun |
| *"the fourteen intervention conditions our two Fractal cells ran"* | same shape as the first, milder |

The first and third took a `that`. The second was rewritten to *"how much of
the observation to keep"*, since inserting a word there does not separate the
two nouns.

### One ambiguous pronoun, which is a class we have already been caught on

*"below the $25$ to $50$~Hz **it** says real-time deployment needs."* The
nearest noun is *autoregressive OpenVLA*, not the OpenVLA-OFT paper. Now
*"that paper says."* This is the same defect as the *"below it"* pronoun raised
in pass eleven, in the same paragraph, and it survived that pass. Pronouns in
this section should be checked against the nearest preceding noun, not against
what the author meant.

Two more of the same kind inside result 1. *"It stays inert while it is loose"*
and *"Widening it"* both leaned on a *"restriction"* / *"window"* chain that
had three referents in four sentences. Both pronouns are now the noun.

### Verb placement, voice and one imperative that read as a command

- *"we held the backbone, the benchmark, the method and the number of layers
  removed (four) fixed"* put eighteen words between `held` and `fixed`. Now
  *"we fixed the backbone, …"*.
- *"Five cells are filled"* was the last easy passive in the section. Passive
  is now 3 of 60 sentences.
- **"Average the two cells and the effect nearly vanishes."** The author read
  this as an imperative and asked whether `Average` was a command. That is the
  reading, and one reader hitting it is enough. Now *"Averaged over the two
  cells, …"*.
- *"The premise is what we test"* → *"That premise"*, which points at the
  sentence before it rather than making the reader decide which premise.
- *"Our claim is narrower"* sits four sentences after *"we do not claim that
  these interventions fail in general,"* with the baseline defence in between,
  so the link had gone cold. Now *"What we do claim is narrower,"* which
  echoes the earlier *do not claim* and re-opens the thread.
- A sentence opened on a lowercase `\texttt{pick coke can}`. Now *"The task
  \texttt{pick coke can} …"*, which also tells a first-time reader that these
  are task names.

### One vagueness kept on purpose

*"one of them by a wide margin"* stays vague, and it should not be replaced
with the number. The cell is OpenVLA on Bridge, ours $15.6\%$ against a
published $1.0\%$, and `Report.md` §7.1 records why that figure cannot go in
an Introduction: the $1.0\%$ has **one** independent source, and OpenVLA's own
training mix is $13.3\%$ Bridge, so a fifteen-fold gap raises a question about
the published evaluation setup rather than about our policy. Printing $15.6$
against $1.0$ here would demand the whole disclosure on the spot. The margin
belongs in Setup with its explanation. What the Introduction needs from the
sentence is only that the deviation is upward, which is what it says.

Keeping the phrase also keeps us honest. `Report.md` §7.1 explicitly forbids
writing *"nowhere lower,"* and the sentence after this one gives the $4.2$
point exception by name.

### What did not change, and was checked

Mixed tense is deliberate. Past for what we did (*we chose*, *we swept*, *we
fixed*), present for what the paper does (*we fill*, *we run McNemar's*, *what
we test*). That split is consistent throughout and is the normal convention.

Longest sentence rose from 35 to 41 words, all from inserted relative pronouns
and clarifying nouns. The ceiling is 45 and the mean is 17.7.

---

## 14. Pass fourteen, 2026-08-23: a regression pass thirteen introduced

Three items raised on the pass-thirteen text. One was a defect **created** by
pass twelve, which is the part worth recording.

### The regression, and how it happened

Pass twelve rewrote this sentence to kill a false universal:

| pass | sentence |
|---|---|
| before | *"…is a restriction an implementation can add **without any specification calling for it**, and **ours** did."* |
| pass 12 | *"…is a restriction **neither method we implement asks for**, and **ours** added it anyway."* |
| now | *"…is a restriction neither method we implement asks for, and **our implementation** added it anyway."* |

The fix was right. *"Any specification"* was refuted by Gromov, cited two
sentences earlier. But **the phrase that carried `ours` was `an
implementation`**, and the rewrite deleted it. After pass twelve the only noun
`ours` could attach to was *method*, in *"neither method we implement."*

So the sentence read *our method added the restriction*, in a paper whose last
paragraph opens *"We propose no new efficiency method."* Worse than the
contradiction, it was simply wrong about what happened. The thing that added
the restriction was our code, and `TableI_Cells.md` §4(b) points at the two
source lines.

**The lesson is narrow and worth stating.** When a sentence is rewritten to fix
one defect, the pronouns downstream of the edit have to be rechecked against
the *new* nouns, not the old ones. Pass thirteen ran a pronoun sweep and
cleared three pronouns in this same paragraph. It missed this one because it
checked pronouns the earlier drafts had, and `ours` had been fine in every
earlier draft. **A pronoun that was correct before an edit is not evidence it
is correct after one.**

### Two reorderings, both cheaper than the fix that was suggested

**`That premise` had a sentence between it and its referent.**

> A. *That last step holds only if the effect belongs to the method rather than to the configuration it was measured in.*
> B. *Telling the two apart takes a grid…*
> C. *That premise is what we test, not any individual method.*

C points at A across B. The suggested repairs were *"That underlying premise"*
or *"This dependency"*, but neither closes the gap, they only label it. **Swapping B and C** closes it for free, and it pays a second time: the
paragraph now ends on *"takes a grid that moves the backbone and the benchmark
independently"*, and the next paragraph opens on *"three open backbones and
two SimplerEnv suites"*. The hinge between the two paragraphs is now a setup
and its payoff.

The swap created one new reference, since *"telling the two apart"* now sits
further from the pair it counts. That pronoun is gone too: *"Telling the
method from the configuration."* Same length.

**The floating `each` came from pass thirteen and was a lateral move.** That
pass changed *"over 96 baseline WidowX-Bridge episodes each"* to *"each over
96 baseline WidowX-Bridge episodes"*, trading one ambiguous attachment for
another. Both were readable and neither was clean. Splitting the sentence
removes the quantifier entirely: *"Both averages cover 96 baseline
WidowX-Bridge episodes."* It also shortens the section's second sentence,
which was carrying two models, two GPUs, two timings and an episode count.

### Paragraph rewrapping is now scripted

The .tex is wrapped at 78 columns and hand-editing across a wrap boundary
leaves ragged lines, which is how a *"comparisons we"* fragment went missing
for one commit in pass twelve. Paragraphs are rewrapped mechanically after
edits now. This is also why the earlier naming self-test failed silently, and
the two problems have the same root: the file's line breaks are not where its
sentences are.

---

## 15. If the Introduction has to shrink: what to cut, in order

Written 2026-08-25 because Related Work has carried a cut list with costs
since the first draft and the Introduction never has. A cut list written under
page pressure gets made in the wrong order, and the cheapest sentence to
delete is rarely the cheapest one to lose.

**Measured shape, 1066 words over nine paragraphs.**

| | paragraph | words | share |
|---|---|---:|---:|
| 1 | opening, the cost of inference | 127 | 12% |
| 2 | the claim form we test | 78 | 7% |
| 3 | setup and the grid | 127 | 12% |
| 4 | the two-sentence hinge | 24 | 2% |
| 5 | result 1, the 45.9-point window | **224** | **21%** |
| 6 | result 2, foveation | 107 | 10% |
| 7 | result 3, sign reversal | 97 | 9% |
| 8 | contributions | 176 | 17% |
| 9 | scope and the baseline defence | 106 | 10% |

### Cut in this order

| # | cut | saves | what it costs |
|---|---|---:|---|
| 1 | the OFT frequency sentence in ¶1 | ~30 | third-party corroboration that the cost is not ours alone. Our own two timings survive, but they become the only evidence |
| 2 | the per-task split in ¶7, from *"The cell mean hides"* | ~45 | the weakest-supported claim in the section, ten of fourteen, and the one whose provenance needs a whole file. **Cut this before anything in ¶5** |
| 3 | the two-sentence hinge, ¶4 | 24 | the reader meets three bold headings with no warning that they share a shape. Cheap in words, and the shape is the argument |
| 4 | *"Four of our five baseline cells…"* through *"push all five down"*, ¶9 | ~45 | the answer to *"your setup is broken."* It is the objection a reviewer reaches for first. **Do not cut this to save space; cut it only if Setup carries it instead** |
| 5 | the widening sentence in ¶5 | ~45 | the proof that the restriction is inert until tight, which is the answer to *"this is your bug, not the field's problem."* `TableI_Cells.md` §4(b) records that the whole result rests on it |

### Do not cut

**¶2 at 78 words.** It states the claim form the paper tests. Every result is
an instance of it and the paper has no thesis without it.

**The 45.9 contrast itself, ¶5.** It is the strongest result and the only one
that is fully within-cell, so no reviewer can reach it by disputing a
comparison across cells.

**The four contributions, ¶8 at 176 words.** Long, and it is the block a
reviewer skims before deciding. Contribution 4's enumeration, *"of the five
methods we survey, none reports…"*, is the paper's method claim in its
checkable form.

### The real lever is not prose

Table I spans both columns and holds 42 cells. Dropping the empty
UniVLA/Fractal column would save nothing worth having, but folding the two
foveation rows or the three depth rows would. Before cutting an argument,
check the table, the figure area and the reference list, in that order. The
bibliography alone gives back **0.8 pages** by applying IEEE's own rule that
more than six authors becomes *et al.*, which is 173 of 248 printed names, and
costs no argument at all.
