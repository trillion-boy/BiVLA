# What This Project Is, and Where It Stands

> **10-minute read.** This document is the map; the detailed evidence lives in
> the other two. Each section ends with a "for more" pointer.
>
> | Document | What | Length |
> |---|---|---|
> | **This one** | **The whole project in 10 minutes** | ~470 lines |
> | `Report_EN.md` | How we measured (§3), what came out (§4–6), what remains (§7) | ~2,300 lines |
> | `RelatedWork_EN.md` | Prior work, with notes on each paper | ~1,240 lines |

---

## One sentence

**VLA papers report numbers of the form "this method saves compute and keeps
performance." We show, systematically, that those numbers do not hold up when
the conditions change even a little.**

We do not build a new method. We take three existing interventions, re-measure
them **under one shared protocol**, and see how far each number actually
carries.

---

## Why do this

VLA policies are slow — for one model, a single observation takes 2.8 seconds
(UniVLA, on an L4 GPU). So people keep proposing interventions that make
inference cheaper **without any retraining**. The claim in such a paper usually
looks like this:

> We applied intervention M to backbone A. On benchmark B, FLOPs dropped by X%
> and success only dropped by Y points. Therefore M is efficient.

For that sentence to mean anything, the effect has to be a **property of the
method** — the direction should survive a change of backbone or benchmark.
**That assumption is exactly what we test.**

---

## What we measured

**Three interventions** — chosen so that each one touches a different resource.

| Axis | What it reduces | Our intervention |
|---|---|---|
| Time | **How often** the policy is called | action repeat k |
| Vision | **What the policy is shown** | foveation (log-polar / blur) |
| Compute | **How much of the network** runs per call | depth pruning k |

**The grid** — 3 backbones × 2 benchmarks = **5 cells** (UniVLA's public
checkpoint is Bridge-only, so one cell stays empty), with 8 conditions per
cell.

| | OpenVLA | SpatialVLA | UniVLA |
|---|:--:|:--:|:--:|
| SimplerEnv **Bridge** (96 ep) | ✓ | ✓ | ✓ |
| SimplerEnv **Fractal** (135 ep) | ✓ | ✓ | — |

**7,198 episodes in total.** On top of the grid there are sweep experiments
(same intervention at different strengths) and control runs.

**The core of the method is pairing.** We never subtract two success rates.
We **pair episodes that start from the same initial state, count only the ones
whose outcome flipped**, and run McNemar's exact test on those. The simulator
is deterministic (we confirmed that a re-run reproduces results bit for bit),
so there is no run-to-run noise — the p-value is the entire uncertainty.

> For more — intervention specs `Report.md` §3.0 · pairing §3.3 ·
> determinism §3.4

---

## The whole grid at a glance

This is the body of the campaign. **Rows are interventions, columns are cells,
and each value is the change in success rate against that cell's own baseline
(measured pairwise).** Bold marks the eight cells whose **own change** passes
our bar — we ran **38 tests** on the grid (the 35 cells in this table plus two
`prune 8` cells and one `prune 2 + repeat 2` cell), so to filter out what
chance alone could produce, a cell only counts if p < 0.0013.

| | OpenVLA<br>Bridge | OpenVLA<br>Fractal | SpatialVLA<br>Bridge | SpatialVLA<br>Fractal | UniVLA<br>Bridge |
|---|---:|---:|---:|---:|---:|
| *baseline success* | *15.6%* | *38.5%* | *30.2%* | *84.4%* | *81.2%* |
| action repeat 2 | −8.3 | +5.2 | +12.5 | ±0.0 | **−69.8** |
| action repeat 4 | **−11.5** | −1.5 | −12.5 | **−40.0** | **−81.2** |
| foveation log-polar | +18.8 | **−19.3** | −8.3 | +0.7 | +5.2 |
| foveation blur | +17.7 | −8.9 | ±0.0 | −1.5 | −8.3 |
| depth prune 1 | +2.1 | +0.7 | −10.4 | +8.1 | −3.1 |
| depth prune 2 | ±0.0 | ±0.0 | −9.4 | +3.0 | −4.2 |
| depth prune 4 | +1.0 | **+15.6** | **−28.1** | **−17.8** | −2.1 |

> **Only eight of the 38 tests pass.** The rest may show a direction, but no
> single one of them can carry a claim on its own. Even the +18.8 in finding ③
> is not bold here — that cell's own p is 0.0051. What does pass is the test
> that asks whether **Bridge and Fractal differ** (Fisher p = 5.5 × 10⁻⁶).
> Those are two different questions.

**Three things you can read straight off this table.**

1. **No row keeps its sign** — except action repeat 4, and that row's sign is
   negative (so "consistently bad in all five cells" is the only consistency
   we found).
2. **Within a single row, the spread reaches 82 points.** Action repeat 2 is
   +12.5 on SpatialVLA/Bridge and −69.8 on UniVLA/Bridge. Same benchmark, same
   intervention, same k.
3. **How much you save predicts nothing about what happens.** Action repeat
   saves 50–75% of compute and splits as above; foveation saves ≈0% and still
   moves results by ±19 points.

The next three sections each dig into the most tightly controlled spot in this
table, and the section after that measures **what** breaks, inside a single
task.

> For more — per-axis detail `Report.md` §4.2 (time) · §4.3 (vision) ·
> §4.4 (compute), cross-cell comparisons §5


---

## The three findings

### ① Same method, but which layers you delete swings the result by 45.9 points

This is the most tightly controlled result we have. **Inside one cell**
(OpenVLA × Fractal, 135 episodes, baseline 38.5%) we held everything fixed —
backbone, benchmark, method, number of deleted layers (4), compute saved
(−11%) — and moved **one flag**.

| Deleted layers | Change in success | Compute saved |
|---|---:|---:|
| [17,20,23,26] etc. | **+15.6** | −11.9% |
| [2,4,23,26] | +5.9 | −10.9% |
| [17,23,27,31] | +1.5 | −10.9% |
| [28,29,30,31] | **−30.4** | −10.6% |

Every change is **against the same baseline (38.5%)**, measured on paired
episodes — +15.6 means 38.5% → 54.1%, and −30.4 means 38.5% → 8.1%. In that
last condition, the three pick tasks all drop to 0 out of 25.

**What we moved is the "candidate window," plus one selection rule.** Deleting
layers happens in two steps: ① decide which range of layers is allowed to be
deleted (the window), then ② inside that window, pick the 4 layers that
matter least, by score. Three of the four rows above moved **① (the
window)**; one row ([17,23,27,31]) kept the default window and only added a
**spacing rule to ②** (chosen layers must be at least 3 apart).

| | Window | Candidates | |
|---|---|---:|---|
| **Default window** — rows 1 & 3 | L16–31 | 16 | picks from the back **half** (row 3 adds the gap-3 rule) |
| **Widened forward** — row 2 | L2–31 | 30 | early layers become candidates too |
| **Shifted back** — row 4 | L28–31 | **4** | only the **last 4 layers** are candidates |

**The shifted-back window has 4 candidates and deletes 4.** There is nothing
left to choose, so the ranking does nothing and the last four layers are
deleted by force — the −30.4 is not "the worst set the score picked" but the
value of **a condition where no choice exists at all**.

**So if a paper only says "we deleted 4 layers," the experiment cannot be
reproduced.** All four rows use **the same ranking score (BI) and the same
count (4 layers)**, yet the results run from +15.6 to −30.4. What decided the
outcome was not the method's quality but **where the implementation put the
candidate window and what constraint it placed on the selection** — and prior
papers usually do not report either.

**And that sensitivity itself depends on the cell.** We took just the two
extreme conditions (default window vs. shifted-back window) and ran the same
contrast in all five cells.

| Cell | Default window | Shifted-back window | **Gap between the two** |
|---|---:|---:|---:|
| SpatialVLA / Fractal | −17.8 | **−68.1** | **50.4** |
| OpenVLA / Fractal | **+15.6** | **−30.4** | **45.9** |
| UniVLA / Bridge | −2.1 | −8.3 | 6.3 |
| OpenVLA / Bridge | +1.0 | −4.2 | 5.2 |
| SpatialVLA / Bridge | −28.1 | −30.2 | 2.1 |

The rightmost column is not a success rate — it is **how far apart the two
conditions' results land**. The bigger it is, the more "which layers you pick"
decides the outcome. Since layer counts and flag meanings differ per backbone
(§3.5.1), the window shift was matched like this — OpenVLA and UniVLA (32
layers): L16–31 → L28–31; SpatialVLA (26 layers): L2–24 → L13–24.

**Between the two Fractal cells (45.9, 50.4) and the three Bridge cells (2.1,
5.2, 6.3) there is a gap of about 40 points, and no cell falls inside it.**
The gap follows the **benchmark**, not the backbone.

**It is not a floor effect.** UniVLA/Bridge starts at 81.2% — 81 points of
room to fall — and its gap is still only 6.3. SpatialVLA/Fractal starts at
almost the same height (84.4%) and its gap is eight times larger.

**How to read this.** A paper that concludes "this method is robust to layer
choice" on one benchmark could see that conclusion move by 45.9 points just by
switching benchmarks. In other words, **sensitivity to layer choice is a
property of the condition, not of the method.**

> For more — `Report.md` §4.4 (c)

### ② Foveation's gain does not come from what it throws away

Foveation is an intervention modeled on the human eye — **sharp in the center,
rough in the periphery.** The original logic is that the more you throw away,
the cheaper the computation. The knob that decides how much to throw away is
`keep`: keep 20% means only 20% of the samples survive.

We swept that knob on OpenVLA/Bridge (96 episodes each, baseline 15.6%).

| keep | Change in success | p |
|---:|---:|---:|
| 10% (throws away a lot) | +4.2 | 0.57 |
| 20% | +18.8 | 0.0051 |
| 40% | +19.8 | 0.0013 |
| **100% (throws away nothing)** | **+30.2** | **4.2 × 10⁻⁷** |

**The condition that throws away the least does the best.** The more you
discard, the worse it gets, and at 10% the gain nearly disappears. The original
logic predicts exactly the opposite.

#### What the code actually does

Four steps.

1. **Unroll the image into polar coordinates** (center magnified, edges
   compressed)
2. **Thin it down** by the `keep` fraction
3. **Resize** back up to the original size
4. **Warp back** to the original shape

**`keep` only controls step 2.** The round trip of steps 1 and 4 happens
**every time**, whether keep is 100% or 10%. So keep=100% is not "foveation
turned off" — it is **an image that skipped the thinning but still took one
round trip through polar coordinates**.

![the four log-polar stages](figures/logpolar_stages.png)

The top row is keep=100%. **Column 3 is exactly identical to column 2** — the
thinning did nothing. Yet the difference map on the far right is not empty. In
the bottom row (keep 20%), column 3 shrinks, and the differences spread past
the object outlines into the background.

#### Why the round trip alone smears the periphery

When the image is unrolled into polar coordinates, the horizontal axis is
**the logarithm of the radius**. So columns are assigned by the *ratio* of the
radius, not by pixel count. Measured on an actual 640×480 observation:

| Distance from center | Columns assigned | Source width (px) | Source pixels per column |
|---|---:|---:|---:|
| 1–2 px (dead center) | 49 | 1 | **0.02** ← one pixel spread across 49 columns |
| 64–128 px | 74 | 64 | 0.86 |
| 256–400 px (edge) | 47 | 144 | **3.06** ← three pixels squeezed into one column |

> **The "magnify/compress" here happens only inside the intermediate image of
> step 2.** Step 4 undoes it exactly, so **the output's geometry does not
> change** — put a marker at dead center, run the round trip, and it comes back
> to the **0.00 px** spot (edges come back within 1 px too). That is why the
> carrot and the plate do not look stretched.

**The shape comes back; the information does not. And the loss happens on one
side only.**

| | What the round trip does to it | Result |
|---|---|---|
| **Center** | one pixel **copied into 49 columns, then gathered back** | nothing lost |
| **Edge** | three pixels **averaged into one column, then spread back** | the differences among the three are gone for good |

Try it with five actual values:

```
original                     [10, 200,  30, 220,  40]
stretch 49x, then return     [10, 200,  30, 220,  40]   <- what the center goes through: unchanged
squeeze 3:1, then return     [153, 155, 164, 173, 175]  <- what the edge goes through: smeared
```

**Stretching loses nothing.** Spread one value across many columns and gather
it back, and you get the original. **Squeezing loses.** Average three values
into one, and the differences among them cannot be recovered.

![center vs. edge, magnified](figures/logpolar_zoom.png)

This is the actual output magnified 3×. The top patch (carrot and plate) is
nearly indistinguishable from the input, and the difference map only traces
thin lines along the plate's rim (mean 0.68). The bottom patch (the wall) has
**lost its texture entirely** (mean 6.66). **So the round trip alone already
produces "sharp center, blurry periphery" — foveation happens even when
nothing is thrown away.**

#### What lowering `keep` adds

The thinning in step 2 thins **the whole unrolled image evenly**. It does not
discard only the periphery — **the center gets cut too.**

So this transform has two components.

| Component | What produces it | Effect on the policy |
|---|---|---|
| **Center preserved, edges lost** | the polar round trip (independent of keep) | helps |
| **The whole image degrades evenly** | the thinning (worse as keep drops) | hurts |

**That is why the curve peaks at 100%.** Lowering keep leaves the first
component unchanged and only grows the second one.

#### How to read this

1. **The gain is not a payment for what was discarded.** The point that
   discards nothing is the best, and the more you discard, the smaller the
   gain gets.
2. **It does not save compute either.** Across the five cells the change is
   −3.1% to +2.7%. Image size and token count stay the same, so there is
   nothing that could shrink in the first place.
3. **So this is not an "efficiency" technique — it is an input-changing
   technique.** When a paper says it gained efficiency through foveation, what
   actually changed is **not the budget but the picture the policy sees.**

**How far we take the claim.** This cell's baseline is 15.6% — the policy can
barely do the task — and the same intervention is −19.3 on Fractal. So the
claim we make is not "foveation is good" but **"this gain cannot be explained
by compression."**

> For more — `Report.md` §4.3 (b) · to reproduce:
> `experiments/measure_foveation_roundtrip.py`

### ③ Change only the backbone, or only the benchmark, and results split — the backbone matters more

Findings ① and ② dig into one cell. This one compares **across cells** and
asks directly: does the same intervention keep its direction when the
conditions change?

**Changing only the benchmark** — code and checkpoint stay identical.

| Intervention | Backbone | Bridge | Fractal | p |
|---|---|---:|---:|---:|
| foveation log-polar | OpenVLA | **+18.8** | **−19.3** | **5.5 × 10⁻⁶** |

**Changing only the backbone** — benchmark held fixed. Bridge has all three
backbones, so three pairs; Fractal has two backbones, so one pair.

| Benchmark | Intervention | One side | Other side | p |
|---|---|---|---|---:|
| Bridge | **action repeat 2** | SpatialVLA **+12.5** | UniVLA **−69.8** | **1.7 × 10⁻¹³** |
| **Fractal** | **depth prune 4** | OpenVLA **+15.6** | SpatialVLA **−17.8** | **4.3 × 10⁻⁷** |
| Bridge | **action repeat 4** | SpatialVLA −12.5 | UniVLA **−81.2** | **1.4 × 10⁻⁶** |
| **Fractal** | action repeat 4 | OpenVLA −1.5 | SpatialVLA **−40.0** | **1.4 × 10⁻⁵** |
| Bridge | depth prune 4 | OpenVLA +1.0 | SpatialVLA **−28.1** | **1.6 × 10⁻⁵** |
| Bridge | depth prune 4 | SpatialVLA **−28.1** | UniVLA −2.1 | **3.6 × 10⁻⁴** |

**The backbone axis is stronger.** Changing the benchmark produces **one**
comparison that passes our bar; changing the backbone produces **six**, five
of them with p < 10⁻⁴. All three backbones take part, and the axis passes on
**both benchmarks**.

**The second row is the cleanest sign flip.** Same Fractal, same 135 episodes,
same `depth prune 4` — yet OpenVLA gets **+15.6** and SpatialVLA gets
**−17.8**. It is the only pair where **both sides individually pass the bar
and their signs are opposite.**

**The largest split is action repeat 2.** Same benchmark, same intervention,
same k — SpatialVLA **goes up** (+12.5) while UniVLA **drops sharply**
(−69.8). That is an 82-point spread. At k=4, UniVLA's success rate **becomes
0.0%** — all four tasks at 0/24.

Yet **the compute saved is identical across backbones: −50% (k=2) / −75%
(k=4).** On this axis, what depends on the backbone is not "how much you save"
but only **"what you lose in exchange."**

> **Why only UniVLA drops this much, we do not know.** We record the
> observation only — UniVLA emits actions as discrete tokens, and executing
> one prediction for two steps costs this backbone unusually much. Whether
> that is the tokenizer or the training distribution, our data cannot tell.

**How to read this.** The claim *"this method works on VLA"* **cannot rest on
a single backbone.** In our grid, changing the backbone shakes the outcome
harder than changing the benchmark.

And this pattern is not our discovery — **it already sits inside the tables
those papers published.** In all 12 settings of EfficientVLA's Table 2,
`pick coke can` goes up and `move near` goes down, and no paper discusses it
in its text. They all report **only the 4-task average.**

> ⚠️ **The two axes stand on different footings.** UniVLA is Bridge-only, so
> it cannot join the benchmark axis, and Fractal's backbone axis has just one
> pair. So **the benchmark axis rests on two backbones, and the backbone axis
> leans toward Bridge.** This is one reason a third benchmark is needed.

> For more — `Report.md` §5.1 (benchmark axis) · §5.2 (backbone axis) ·
> §4.2 (action repeat) · **`RelatedWork.md` §2.5** (the prior papers' tables)

---

## What breaks — measured inside a single task

The results above say **how much** things get worse, not **what** gets worse.
So in the cell with the largest drop (SpatialVLA/Fractal, depth prune 4), we
counted the failures **by kind**.

`move_near` is "move A next to B," and the environment records enough state at
every step to split failures like this — **moved the wrong object**
(misunderstood the instruction) vs. **touched nothing at all** (knows what to
do, but the arm does not get there).

| | Success | Wrong object | Touched nothing | p |
|---|---:|---:|---:|---:|
| original policy | 50 | 1 | **0** | |
| 4 layers deleted | 31 | 4 | **12** | |
| **change** | | +3 (p=0.375) | **0 → 12** | **0.0005** |

**Only one thing moved.** The original policy moved something by more than 3cm
in **all 60** episodes; delete 4 layers, and 12 episodes end **without
touching anything.** "Moved the wrong object" did not move statistically.

**So when you delete layers, what degrades first is not "understanding" but
"doing."** The policy still knows what to pick up — the arm just does not get
there.

> **Why we designed this measurement.** At first we believed the opposite. We
> compared across tasks (`move_near` −31.7 vs. `pick_coke_can` −6.7) and tried
> to explain it as "the ability to understand the instruction degrades first."
> But those two tasks differ in five ways at once, so that comparison cannot
> isolate anything. **Claiming a cause from a single between-task comparison
> is exactly the mistake we call out in this field.** So we built a tool that
> measures **inside** one task — and the signal came out opposite to what we
> expected.

> For more — `Report.md` §6

---

## So what is the conclusion

The three findings have **the same shape**: the reported number was a
**property of the condition, not of the method.**

| | What changed (method untouched) | How far the result moved |
|---|---|---:|
| ① | one candidate-window flag | **45.9 points** |
| ② | the `keep` value | **26 points** (within one campaign) |
| ③ | the benchmark | **+18.8 → −19.3** (sign flip) |
| ③ | the **backbone** | **+12.5 → −69.8** (82 points, p = 1.7 × 10⁻¹³) |
| ③ | the backbone (sign flip) | **+15.6 → −17.8** (same Fractal, `prune 4`, p = 4.3 × 10⁻⁷) |

Three things follow.

1. **A number measured in one condition cannot carry a claim about the
   method.** A sentence like "we applied M and success only dropped Y points
   on B" says nothing about whether it holds outside that one condition. In
   our grid, such sentences mostly failed to survive a change of condition.
   **A single backbone especially cannot carry it** — changing the backbone
   shook results harder than changing the benchmark.

2. **What papers currently report is not enough to reproduce them.** For ①
   you need the candidate window; for ② the variant and the `keep` value; for
   ③ the per-task breakdown. All three are missing from current practice, and
   the **4-task average** in particular hides exactly the split we saw in ③.

3. **The "efficiency" label gets attached without being checked.** The
   intervention in ② saves ≈0% compute yet gets cited as an efficiency
   technique. In our grid, the amount saved and the change in success did not
   predict each other.

**What we offer is not a new method but the measurement procedure that exposes
these three things** — episode pairing, determinism checks, and grid
uniformity rules. Prior work cannot be re-examined this way because it does
not release per-episode records, and that fact is itself one of our points.

---
---

## Current status

| | |
|---|---|
| **Simulation** | **Done. 5 cells × 8 conditions, no gaps.** 255 result files |
| Verification | Every number in the documents recomputed from the records — **all match. The mismatches found along the way number 98, and all are fixed** (`Report.md` §7.1) |
| Baselines | All five cells compared against the original papers — **four are higher, one is 4.2 points lower** (SpatialVLA/Bridge, 30.2% vs. 34.4%). A broken setup would be *consistently* low, so the argument stands |
| Prior work | **10 papers** checked against the original PDFs — 7 prior-work + 3 backbone papers |

**Next is the real robot.** One thing matters in that design: it should be
shaped as **testing on hardware the predictions the simulation grid already
made.** For example, the sim says "foveation helps on OpenVLA/Bridge." If it
helps on the real WidowX too, the sim predicts hardware; if it does not, we
get "what was measured in sim does not transfer to hardware" — which makes the
argument stronger, not weaker. **Either outcome is a result.**

Hardware has no determinism, but **pairing survives** — place the objects at
the same spots and alternate the two conditions. The original OpenVLA paper
evaluates hardware the same way.

**Three open items** (only ③ can be closed by running more simulation)

1. **A third benchmark** — our current two are **both SimplerEnv**. A
   benchmark on a different engine would make "we changed the benchmark"
   stronger
2. **UniVLA/Fractal** — cannot be filled; the public checkpoint is
   Bridge-only
3. **The drawer task family** — we used two of Fractal's three standard task
   groups. About 6 hours at diagnostic scope

> For more — `Report.md` §7

---

## When this becomes a paper

**We propose no new method, so what goes into Method is not an algorithm but
the measurement procedure.** Three parts.

1. **Per-episode pairing** — impossible to apply to prior work because they do
   not release episode records, which is itself one of our points
2. **The determinism check, and the statistical reading that follows from it**
3. **The grid uniformity rules** — what counts as a grid cell and what counts
   as a diagnostic

**Results has three tiers**, and the third one (findings ① and ②), which has
the most controlled variables, will likely carry the most weight.

> For more — `Report.md` §7.3

---

## How the three documents relate

| | `Overview.md` | `Report.md` | `RelatedWork.md` |
|---|---|---|---|
| Length | ~470 lines | ~2,300 lines | ~1,240 lines |
| Purpose | **grasp it in 10 minutes** | evidence, verification, records | what others measured |
| Contains | the three findings and status | every table, the 98-entry correction log, machine checks | §2 prior work + notes on 7 papers |
| When to read | first, or when explaining to others | when defending a number | when writing the paper's §2 |

`Report.md`'s §7.1 (correction log) and §7.2 (machine verification) are
**long, but do not delete them.** They are the reason this work can survive
review.
