# Lab meeting speaking script — English

Six slides, about seven minutes. The **[SAY]** blocks can be read as written.
Numbers come from `LabMeeting_Bridge_Fractal_0806.md`; regenerate the tables
with `python experiments/build_grid_report.py`.

The whole talk turns on **one picture** — four curves of success rate against
how long each action is held. The statistics then hang off **one number**, the
repair rate.

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

## Slide 2 — Four curves (100s) ★core

**[SHOW]** — success rate when each action is held for 1, 2, then 4 steps

```
                  k=1     k=2     k=4
OpenVLA  /Bridge  15.6 →  7.3 →  4.2    \___  monotone collapse
OpenVLA  /Fractal 38.5 → 43.7 → 37.0    ────  flat
SpatVLA  /Bridge  30.2 → 42.7 → 17.7    /\__  peak at 2
SpatVLA  /Fractal 84.4 → 84.4 → 44.4    ‾‾\_  flat, then a cliff
```

**[SAY]**
"Let me take one intervention: how many environment steps we hold each action
for. One is the original policy; two and four mean calling the model that much
less often.

Two backbones, two benchmarks, four cells.

**[PAUSE — point at the screen]**

**All four are different shapes.** One collapses monotonically, one is flat, one
peaks at two, one is flat and then falls off a cliff.

The top two rows are **the same weights**. That's one policy, OpenVLA. On Bridge
it loses two thirds of its success by k=4. On Fractal it doesn't move at all.

**Only the benchmark changed.**"

---

## Slide 3 — Could this be chance? (80s)

**[SHOW]**

```
Of the episodes the intervention CHANGED, the fraction it FIXED
                                          (50% = coin flip = no effect)

OpenVLA / Bridge  repeat 4    0/11  ▏                  0%
OpenVLA / Fractal repeat 4   20/42  ███████████       48%
                                         ┊
```

| comparison | condition | p |
|---|---|---|
| **OpenVLA: Bridge vs Fractal** | **repeat 4** | **0.0038** ✓Bonferroni |
| OpenVLA: Bridge vs Fractal | repeat 2 | **0.027** |
| SpatialVLA: Bridge vs Fractal | repeat 4 | **0.0085** |
| SpatialVLA vs UniVLA (Bridge) | repeat 2 | **< 0.0001** ✓Bonferroni |

**[SAY]**
"We tested whether that's chance.

We look only at episodes whose outcome changed — the intervention either fixed a
failure or broke a success. Then: what fraction did it fix? If the intervention
does nothing, that's a coin flip, **50 percent**.

**[EMPHASISE]**

OpenVLA on Bridge with action repeat 4: eleven episodes changed, and **all
eleven were broken.** Zero percent. The chance of eleven coin flips landing the
same way is 0.1 percent.

The same weights on Fractal: forty-two episodes changed and it **fixed twenty**
of them. Forty-eight percent — essentially a coin.

Asking whether those are the same coin is a Fisher exact test, and it comes out
at **p = 0.0038**. That clears Bonferroni for the eight tests we ran."

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
> "The action-repeat axis is complete across both backbones and both benchmarks.
> What remains is re-measuring the legacy cells and running one baseline twice in
> a session to establish the per-episode noise floor."

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
four shapes:  collapse / flat / peak-at-2 / flat-then-cliff   (slide 2)
0%  vs  48%   OpenVLA repeat 4, Bridge vs Fractal             (slide 3)
p = 0.0038                                                    (slide 3)
−40.0  vs  ~0  temporal vs visual                             (slide 4)
```
