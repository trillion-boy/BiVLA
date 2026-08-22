# Introduction

*Reading draft. Same content as `introduction.tex`, written to be read rather
than compiled — citations are spelled out, and every number is one that
`experiments/verify_all.py` recomputes from the per-episode records.*

---

## The claim we are testing

Vision-language-action policies are slow. In our own runs, one forward pass
costs **UniVLA 2.81 s** and **SpatialVLA 0.90 s**, averaged over the 96
baseline episodes of the WidowX-Bridge suite. (Model time only, excluding
simulator stepping, on Colab T4/L4-class hardware. We give these as a scale
rather than a comparable benchmark: our result files do not record *which* card
each run used — see the limitation note below.)

So a steady stream of **training-free** interventions has appeared: methods
that leave the checkpoint untouched and make inference cheaper by skipping
decoder layers, caching or pruning visual tokens, or reusing one predicted
action across several control steps. EfficientVLA, VLA-Cache, FastV and
ShortGPT are the training-free ones we read closely. MoLe-VLA pursues layer
skipping too, but **with training** — a learned router plus self-distillation
— so it sits outside what we test, and we cite it as such.

Papers of this kind report their result in a recurring form:

> Applying method **M** to backbone **A** on benchmark **B** reduced FLOPs by
> **X%** while success dropped only **Y** points. Therefore **M** is efficient.

For that last sentence to follow, the measured effect has to be a property of
**the method** — not of the single configuration it happened to be measured
in. Change the backbone, change the benchmark, or change an implementation
detail the paper does not report, and the direction of the result should hold.

**That premise, rather than any individual method, is what this paper tests.**

---

## What we measured

Three existing interventions, re-measured under one protocol. They were chosen
because each spends a different resource:

| axis | what it changes | our intervention |
|---|---|---|
| time | how **often** the policy is called | action repeat *k* |
| vision | what the policy is **shown** | foveation (log-polar / blur) |
| compute | how much of the **network** each call uses | depth pruning *k* |

Three open backbones × two SimplerEnv suites:

| | OpenVLA | SpatialVLA | UniVLA |
|---|:--:|:--:|:--:|
| **WidowX-Bridge** (96 episodes) | ✓ | ✓ | ✓ |
| **Google Robot / Fractal** (135 episodes) | ✓ | ✓ | — |

Five filled cells — UniVLA's public checkpoint is Bridge-only — with eight
conditions each: baseline, action repeat 2 and 4, foveation log-polar and
blur, and depth pruning of 1, 2 and 4 layers. **7,198 episodes in total.**

### The measurement is paired

We do not subtract two aggregate success rates. Every condition replays the
**same initial states** as its cell's baseline; we keep only the episodes whose
outcome changed and test those with McNemar's exact test. Episodes where both
conditions agree carry no information about which is better, and pooling them
into two averages throws away the pairing that makes the comparison sensitive.

The simulator is deterministic — we checked that re-running a condition
reproduces it episode for episode — so there is no run-to-run variance and the
remaining uncertainty is entirely in the *p*-values.

Correcting for multiple comparisons: **38 paired tests** inside the grid, so
α ≈ 0.0013, of which **8 cells** are significant; **43 cross-cell
comparisons**, so α ≈ 0.0012, of which **7** survive.

---

## Three results, and they have the same shape

### 1. A choice the specifications leave open moves the result by 45.9 points

Inside one cell (OpenVLA × Fractal, 135 episodes, baseline 38.5%), holding
**everything** fixed — backbone, benchmark, method, number of layers removed
(four) — and changing only which layers were *eligible* to be removed. Compute
saved is not a single fixed value but a range, −10.6% to −11.9%, whose spread
is the noise on re-running the same measurement (`Report.md` §7.1):

| eligible window | success change |
|---|---:|
| back half of the stack (L16–31) | **+15.6** |
| widened forward (L2–31) | +5.9 |
| back half + spacing rule | +1.5 |
| last four layers only (L28–31) | **−30.4** |

The ranking criterion is the same in all four rows. What differs is the
candidate window and the spacing constraint. **These are reported — and the
literature disagrees about them.** ShortGPT ranks every layer and constrains
nothing; EfficientVLA §3.2.2 does the same and calls it "non-contiguous";
Gromov et al. (ICLR 2025) instead remove a contiguous block and find that
"keeping the very last layer is essential." Nobody has compared the two
prescriptions on a robot policy. (Do **not** write "the papers do not report
this" — it was checked against the PDFs and it is false; see
`paper/TableI_Cells.md`.) The last row is the sharpest: with only four
candidates and four layers to remove, the ranking does nothing at all, so
−30.4 is the score of a condition with **no selection in it**.

Run in all five cells, the same two-row contrast spans **2.1 to 50.4 points**.
So the sensitivity to layer choice is itself a property of the configuration,
not of the method.

### 2. A gain attributed to compression does not come from compression

Foveation keeps a fraction of the observation and discards the rest; the
efficiency argument is that the discarding is what buys you something. Sweeping
that fraction on OpenVLA/Bridge:

| keep | success change | *p* |
|---:|---:|---:|
| 10% (discard most) | +4.2 | 0.57 |
| 20% | +18.8 | 0.0051 |
| 40% | +19.8 | 0.0013 |
| **100% (discard nothing)** | **+30.2** | **4.2 × 10⁻⁷** |

**The best setting is the one that discards nothing**, and the benefit shrinks
monotonically as more is discarded. Measured compute moves by −3.1% to +2.7%
across the five cells, because the image size and the token count never change.

The effect is real and we can trace it: the log-polar round trip is lossy
asymmetrically. The centre is stretched across many columns and folded back
without loss; the periphery is averaged down and cannot be recovered. So the
transform already produces "sharp centre, soft surround" **before anything is
subsampled**. What it is not is an efficiency technique.

We are careful about the scope here. This cell has a 15.6% baseline, and the
same intervention scores −19.3 on Fractal. The claim is not "foveation helps"
but **"this gain cannot be explained by compression."**

### 3. Changing the backbone moves results more than changing the benchmark

Results 1 and 2 look inside one cell. This one compares cells directly, asking
whether an intervention keeps its direction across conditions.

**One** benchmark-axis comparison passes correction. **Six** backbone-axis
comparisons do, five of them at *p* < 10⁻⁴.

| held fixed | changed | one side | other side | *p* |
|---|---|---|---|---:|
| OpenVLA, foveation log-polar | benchmark | Bridge **+18.8** | Fractal **−19.3** | 5.5 × 10⁻⁶ |
| Fractal, depth prune 4 | backbone | OpenVLA **+15.6** | SpatialVLA **−17.8** | 4.3 × 10⁻⁷ |
| Bridge, action repeat 2 | backbone | SpatialVLA **+12.5** | UniVLA **−69.8** | 1.7 × 10⁻¹³ |

The second row is the cleanest sign reversal in the campaign: same 135
episodes, same intervention, and **both sides pass correction individually
while pointing opposite ways.** The third is the widest — 82 points — and there
the compute saved is identical by construction (−50% at *k*=2), so what the
backbone changes is not how much is saved but what is lost for it.

*(We do not know why UniVLA in particular collapses under action repeat. It
emits actions as discrete tokens, and holding one prediction for two steps
costs this backbone far more than the others; our data cannot separate a
tokeniser cause from a training-distribution one.)*

---

## The same split is already in the published tables

This is not only visible in our grid. Splitting the Google Robot results that
prior work already prints, `pick coke can` holds up better than `move near` in
**fourteen of fifteen configurations** reported across four author groups
(EfficientVLA, VLA-Cache, FastV, VLA-Pruner). Where a capacity ladder exists,
the direction is monotone in eight of them.

**The one exception is informative rather than damaging.** It is VLA-Pruner —
the single method in the set designed specifically to stop pruning
action-critical tokens (97.0% of baseline preserved on `move near` against
94.9% on `pick coke can`). The pattern holding for the salience-based methods
and breaking for the method built to fix it is what its mechanism predicts.

None of those papers discusses the split. The per-task numbers **are** printed
— EfficientVLA's SIMPLER table has `PickCan | MoveNear | Drawer | DrawerApple |
Average` columns, and VLA-Pruner's Table 2 has `Move Near | Pick Coke Can |
Open/Close Drawer | Overall`. What is average-only is the *discussion*: every
summary sentence in those papers quotes the aggregate, which is the operation
that hides the split. Do not write "they report only the average" — it is false
and a reviewer with the PDF will catch it.

*(Stated precisely: fifteen configurations printed in **two** tables —
twelve in EfficientVLA's, three in VLA-Pruner's — spanning four method
families, of which only **three** are independent sources. FastV never reports
SimplerEnv itself; its rows in both tables are comparisons run by others. Do
not write "four author groups." Two random-dropping rows are excluded from the
twelve as a control rather than a proposed method, and that exclusion has to be
disclosed wherever the count appears.)*

---

## What breaks, not just how much

Everything above measures **how much** performance moves. In the cell with the
largest drop we also asked **what** breaks, by classifying failures inside a
single task rather than comparing across tasks.

`move_near` asks the policy to move object A next to object B, and the
environment records enough per step to separate two failure modes: moving the
**wrong object** (misunderstanding the instruction) versus **never moving
anything** (knowing the goal but not reaching it).

Every episode lands in exactly one bucket (60 per row):

| | success | dropped | **wrong object** | misplaced | **never moved anything** |
|---|---:|---:|---:|---:|---:|
| original policy | 50 | 4 | **1** | 5 | **0** |
| 4 layers removed | 31 | 7 | **4** | 6 | **12** |
| change | | | **+3** (*p* = 0.375) | | **0 → 12** (*p* = 0.0005) |

**Only one of the two diagnostic buckets moved.** The original policy shifted
*something* by more than 3 cm in all 60 episodes; after removing four layers,
12 episodes end without the policy having touched anything at all, while
picking the wrong object is statistically unchanged.

Removing layers damages **execution** before it damages **grounding** — the
policy still knows what to pick up; the arm no longer gets there.

We designed this measurement to test the opposite hypothesis, and it did not
survive. We report that, because an earlier version of this work would have
argued the reverse from a between-task comparison — and arguing a cause from a
single between-task comparison is precisely the error we are raising about the
field.

---

## Contributions

1. **A uniform re-measurement** of three training-free VLA interventions over a
   3 × 2 backbone-benchmark grid — five filled cells, eight conditions each,
   7,198 episodes — with the per-episode records released.

2. **Evidence that reported effects are often properties of the configuration,
   not of the method.** An eligibility choice the specifications leave open
   moves one result by 45.9
   points, and the same intervention reverses sign across backbones with both
   directions individually significant.

3. **Evidence that compute saved does not predict success change.** The
   intervention with the largest effect in our grid saves approximately no
   compute, and its benefit is largest when it discards nothing.

4. **A measurement procedure for this class of claim** — episode-level pairing,
   an explicit determinism check, and a uniformity rule for the grid — together
   with an account of what current reporting practice omits that makes these
   claims unreproducible.

---

## A note on the hardware limitation

We ran on Colab, on T4 and L4 cards. The limitation is not that we don't know
the hardware class — it is that **"T4 or L4" is two different cards, and no
per-run record says which one each run used.**

That would be a bookkeeping nit if the card made no difference. It does. We
have a direct measurement, because UniVLA/Bridge has two baselines with
identical settings run on different cards:

| | mean | episodes differing |
|---|---:|---:|
| `baseline` (08-05) | 78.1% | — |
| `baseline_l4` (08-10) | 81.2% | **11 of 96** |

And inference time cannot recover the card: 2802 ms versus 2812 ms, essentially
identical while 11 episodes disagree. Commit dates show that at least one
cell's conditions were run across several sessions.

There is a sharper version of the same measurement. Re-running UniVLA/Bridge
foveation on a different card left the mean **identical to the decimal**
(86.5% → 86.5%) while **all four task rates moved** (−4.2 / +4.2 / +8.3 / −8.3)
and cancelled out. The baseline over the same card change moved 0 of 96
episodes — so what shifted was the foveation path specifically, and we could
not separate a GPU cause from a `cv2` build difference.

**What this costs us, and what it doesn't.** Attaching every condition in that
cell to both baselines shifts every delta by 3.1 points and flips no
significance decision; the two that change sign have p ≥ 0.6 either way. Our
headline values (−69.8, −40.0, −28.1, −19.3, +18.8, +15.6) are an order of
magnitude larger. So the effect is bounded, and hardware confounding cannot
overturn the conclusions.

But the sentence we can defend is **"even if the card was not fixed, the effect
does not exceed this size"** — not "the card was fixed." The fix costs one line
in the next campaign: write the GPU, driver and library versions into the
result file. The notebooks we hand over now do this.

*(This is also why we present the two latency figures above as a scale rather
than a benchmark: we cannot rule out that UniVLA and SpatialVLA were timed on
different cards.)*

---

## What we are not claiming

We do not propose a new efficiency method, and we do not claim that any of the
three interventions fails in general.

Four of our five baseline cells sit **above** the published numbers and the
fifth is 4.2 points below (SpatialVLA/Bridge, 30.2% against a published 34.4%).
A systematic setup error would push *all* of them down, so that pattern does
not fit one. Do **not** write that the baselines "match" or "reproduce" the
published numbers — OpenVLA/Bridge is 15.6% against a reported 1.0%, and the
argument works because our numbers are mostly *higher*, not because they agree
(`Report.md` §3.8 c①).

The claim is narrower, and we think more useful: **the evidence currently
offered for these methods does not distinguish a property of the method from a
property of the one configuration it was measured in.**

---

### Where each number comes from

| claim | source |
|---|---|
| 2.81 s / 0.90 s per forward | `results/univla_bridge_0805/**baseline_l4**` (2811.5 ms, the run the grid pairs against, not `baseline` at 2801.5 ms), `results/spatialvla_bridge_0805/baseline` (902.1 ms), 96 episodes each |
| 45.9-point window contrast | `Report.md` §4.4 (c) |
| keep sweep | `Report.md` §4.3 (b); reproduce with `experiments/measure_foveation_roundtrip.py` |
| cross-cell comparisons | `Report.md` §5.1 (benchmark axis), §5.2 (backbone axis) |
| twelve-configuration split | `RelatedWork.md` §2.5 — all twelve rows are EfficientVLA Table 2 |
| failure typing | `Report.md` §6.4 |
| 38 / 43 test families | `Report.md` §3.3, §5 — both derived from row counts, not counted by hand |

All of the above are recomputed from the raw records by
`experiments/verify_all.py` (216 checks, 0 failures).
