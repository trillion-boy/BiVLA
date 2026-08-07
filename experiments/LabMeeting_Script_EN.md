# Lab meeting speaking script — English

Six slides, about seven minutes. The **[SAY]** blocks can be read as written.
Numbers come from `LabMeeting_Bridge_Fractal_0806.md`; regenerate the tables
with `python experiments/build_grid_report.py`.

The whole talk turns on **one number** — the **repair rate**. Hold onto that
and everything else follows.

---

## Slide 1 — What we set out to do (30s)

**[SHOW]**
> Goal: make VLA policies faster without losing accuracy
> Interventions: action repeat / foveation / layer skipping
> 4 backbones × 2 benchmarks, identical code at identical hook points

**[SAY]**
"We set out to make VLA policies cheaper to run without losing accuracy.
Three interventions: hold each action for several environment steps so the
model is called less often, remove information from the observation, or skip
decoder layers.

For that to be a method claim, the effect has to be a property of **the
method**. So we ran the *same code* through the *same hook points* across four
backbones and two benchmarks."

---

## Slide 2 — The whole result in one number (100s) ★core

**[SHOW]** — five horizontal bars, dashed vertical line at 50%

```
Of the episodes the intervention CHANGED, the fraction it FIXED

UniVLA     / Bridge    1/70  ▏                      1.4%
OpenVLA    / Bridge    3/14  ████                  21.4%
SpatialVLA / Fractal  11/22  ███████████           50.0%   ← coin flip
OpenVLA    / Fractal  22/37  █████████████         59.5%
SpatialVLA / Bridge   21/30  ███████████████       70.0%
                                          ┊
                                50% = no effect
```

**[SAY]**
"You can see this whole result in one number.

We run the same episode twice — once with the original policy, once with the
intervention. Most episodes come out the same either way: both succeed, or both
fail. Those tell us nothing, so we drop them.

We keep only the episodes where **the outcome changed**. Either the
intervention fixed a failure, or it broke a success. Then we ask: what fraction
did it fix?

If the intervention does nothing, that's a coin flip — **50 percent**. Above
50, it helped. Below 50, it hurt.

**[PAUSE — point at the screen]**

Same intervention. Same code. UniVLA is at 1 percent. SpatialVLA is at 70.

And look at the two middle rows. **That's one backbone, OpenVLA, sitting at 21
percent and 59 percent at the same time.** Same checkpoint, same weights, same
code. **Only the benchmark changed.**"

---

## Slide 3 — Could this be chance? (80s)

**[SHOW]**

| comparison | repair rates | p |
|---|---|---|
| SpatialVLA vs UniVLA (Bridge) | 70% vs 1% | **< 0.0001** |
| OpenVLA vs SpatialVLA (Bridge) | 21% vs 70% | **0.0037** |
| **OpenVLA: Bridge vs Fractal** | **21% vs 59%** | **0.027** |
| SpatialVLA: Bridge vs Fractal (repeat 4) | 31.2% vs 9.1% | **0.0085** |

**[SAY]**
"The obvious question is whether this is chance. We test two things.

First, **is one bar different from 50 percent?** If the intervention were
harmless, it would be a coin flip. On OpenVLA/Bridge, eleven episodes changed
and all eleven were broken. The chance of eleven coin flips all landing the
same way is 0.1 percent. That's the p-value.

Second — and this is our actual question — **are two bars different from each
other?** Bridge and Fractal share no episodes, so there is nothing to pair up.
We compare the rates instead. That's a Fisher exact test.

**[EMPHASISE]**

Here's the part that matters. OpenVLA's two cells are **not individually
significant** — 0.057 and 0.32. But **the difference between them is**, at
0.027.

If we had only reported the per-cell tests, we would have missed the exact
effect we set out to measure."

---

## Slide 4 — What we did learn (100s) ★highlight

**[SHOW]** — SpatialVLA / Fractal, one session, 135 episodes

| intervention | what it deletes | result |
|---|---|---|
| hold each action 4 steps | re-planning on 3 of every 4 steps | **−40.0** |
| foveation (log-polar) | 80% of the visual signal | +0.7 |
| foveation (blur) | 80% of the visual signal | −1.5 |

**[SAY]**
"It isn't all negative. One thing came out very cleanly.

One policy, one benchmark, one session — so none of the caveats from the
previous slides apply here.

**We deleted four fifths of what the policy can see.** Two different ways: one
warps the pixels, the other only removes detail. **Neither did anything.**

But holding each action for four steps **costs forty points**.

**[PAUSE]**

These policies are far more fragile to **slow hands than to bad eyes**.

The practical implication is that if you want to buy efficiency, take it out of
the visual pathway and leave the temporal one alone."

---

## Slide 5 — Our own explanation broke too (50s)

**[SHOW]**

> Hypothesis: "damage tracks distance from the trained execution length"
> SpatialVLA is deployed **below** its own chunk size (k≈4) → gains at repeat 2
>
> Bridge: **+12.5** ✓ as predicted
> Fractal: **+0.0** ✗ same checkpoint, same chunk size

**[SAY]**
"We did have an explanation. Damage should track how far you move from the
length the policy was trained to execute. SpatialVLA is the one backbone
deployed *below* its own action chunk, so holding an action actually moves it
*toward* its trained length.

**Bridge fits that exactly.**

Then the same checkpoint on Fractal, and the gain is gone. The chunk size never
changed.

**Even the most plausible mechanism does not cross benchmarks.** We report that
rather than hiding it — it's a stronger version of our own thesis than a tidy
explanation would have been."

---

## Slide 6 — Closing (30s)

**[SHOW]**

> **The effect of the intervention was not a property of the intervention.**
> It was a property of which backbone and which benchmark you measured on.
>
> → not a method paper, but a paper about **how this work is evaluated**

**[SAY]**
"To close.

We started out building a method, and found that the method's effect is not a
property of the method. It's a property of which backbone and which benchmark
you measure on.

So this isn't a method paper — it's a paper about how this class of work gets
evaluated. Efficiency papers routinely claim a free speedup from a single
backbone on a single benchmark, and we have a controlled counterexample."

---

# Expected questions

**Q. So the method is dead?**
> "As a method claim, yes. But we got a more general claim instead. And slide 4
> gives a design rule: there's slack in the visual pathway and none in the
> temporal one."

**Q. Are the individual numbers significant?**
> "OpenVLA's two cells are not — 0.057 and 0.32. But our question isn't about
> either cell, it's about **the difference between them**, and that clears at
> 0.027. All four backbone comparisons are significant and two clear Bonferroni."

**Q. Isn't the sample too small?**
> "96 to 135 episodes per condition — the full protocol in each case. Our
> detection floor is about ±7 points, so we can't resolve anything smaller.
> That's why we say 'no effect larger than 7 points' rather than 'no effect'."

**Q. Why does this happen?**
> (go back to slide 5) "Our own hypothesis explained Bridge and broke on
> Fractal. We don't have the mechanism yet. That's the next piece of work."

**Q. What are the italic cells?**
> "An earlier campaign that kept no per-episode records. They're unpaired, and
> their baseline is 32.3 percent rather than the 30.2 in this table. Two
> measurements of the same thing differ by 2.1 points, which happens to be the
> size of the effect those cells report — so we don't build on them. They're
> queued for re-measurement."

**Q. What's left to run?**
> "One cell: OpenVLA on Fractal with action repeat 4. That closes the two-by-two
> grid across both backbones and both benchmarks. About 35 minutes."

---

# Do not

- **Quote a partial cell.** Two moved today when their last task landed:
  +9.3 → +5.2 and +1.3 → −1.5.
- **Compare deltas across columns unguarded.** SpatialVLA sits at 84% on
  Fractal, OpenVLA at 38%. A +5 near the floor and a +5 near the ceiling are
  not the same thing. Always say the baseline alongside.
- **Say "no effect".** Say "no effect larger than about 7 points".

# Numbers to memorise (only these)

```
1.4%  /  21%  /  50%  /  59%  /  70%     repair rates (slide 2)
p = 0.027                                 benchmark difference (slide 3)
−40.0  vs  ~0                             temporal vs visual (slide 4)
```
