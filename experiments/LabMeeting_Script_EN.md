# Lab meeting speaking script — English

Ten slides, about twelve minutes. The **[SAY]** blocks can be read as
written. If you are tight on time, slide 3b is the one to drop — but then also
drop the foveation callback in slide 4b. **Never drop 4b or 4c while keeping
4** — that leaves the +8.1 standing unqualified, which is the one thing this
talk must not do. If you have to lose a whole beat, lose slide 5.
Numbers come from `LabMeeting_Bridge_Fractal_0806.md`; regenerate the tables
with `python experiments/build_grid_report.py`.

The talk has **two pictures**. The first (slide 2) is four curves of success
rate against how long each action is held — that carries the negative result,
and the statistics hang off **one number**, the repair rate. The second
(slide 4b) is one curve splitting into two that run in opposite directions —
that carries the mechanism, and it is the part worth rehearsing.

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
| **OpenVLA: Bridge vs Fractal** | **foveation** | **0.0000055** ✓Bonferroni |
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
at **p = 0.0038**. That clears Bonferroni for the nine tests we ran."

---

## Slide 3b — And it isn't only about time (40s)

**[SHOW]**

```
OpenVLA, foveation — throw away 80% of the observation

           baseline   foveated      Δ
Bridge       15.6%  →  34.4%     +18.8   (p = 0.005)
Fractal      38.5%  →  19.3%     −19.3   (p = 0.0004)
                                  ─────
                              38-point swing,  interaction p = 0.0000055
```

**[SAY]**
"One thing I should flag, because it landed this morning.

Everything so far was about holding actions — the *time* axis. So you could
still say: fine, only the temporal intervention is benchmark-sensitive.

This is a different intervention. We throw away eighty percent of what the
policy can see. On Bridge that **helps** by nineteen points. On Fractal, same
weights, it **hurts** by nineteen points.

**[PAUSE]**

And unlike the repeat case, **both of those are significant on their own.**
This isn't two null results whose difference happens to resolve. It's the
strongest test in the whole grid.

So the reversal is not a quirk of one axis."

---
## Slide 4 — Three axes, and one free lunch (110s) ★highlight

**[SHOW]** — SpatialVLA / Fractal, one policy, one session, full 135 episodes

| what we delete | how much | Δ | p | speed |
|---|---|---|---|---|
| **time** — re-planning | 3 of every 4 steps | **−40.0** | 0.0000 | 1/4 the calls |
| **compute** — decoder layers | **1 of 26 (4%)** | **+8.1** | **0.013** | **1.08×** |
| **compute** — decoder layers | 4 of 26 (15%) | **−17.8** | 0.0002 | 1.18× |
| **vision** — observation | **80%** | +0.7 / −1.5 | 1.00 / 0.83 | **1.00×** |

**[SAY]**
"It isn't all negative. This is the sharpest thing we have.

One policy, one benchmark, one session — none of the caveats from the previous
slides apply. We deleted three different things.

**Four fifths of what the policy can see.** Two ways: one warps the pixels, the
other only removes detail. **Neither did anything** — and neither made it any
faster, because foveation resamples back to the same resolution, so the token
count never drops.

**Holding each action four steps cost forty points.**

**And then the compute axis did something we did not expect.**

**[PAUSE — point at the two compute rows]**

**One of twenty-six layers: plus eight points. Four of twenty-six: minus
eighteen.** Same code, same redundancy criterion, same session, and the four-layer
set **contains** the one-layer set. Fourteen episodes fixed, three broken, and
it runs eight percent faster.

That is a genuine free lunch — the only intervention in this entire campaign
that makes the policy both **better and cheaper**. Hold that thought, because
the next slide is what it's actually for."

---

## Slide 4b — The average was hiding two opposite curves (90s) ★the payoff

**[SHOW]**

```
SpatialVLA / Fractal, decoder layers bypassed  —  aggregate

  0  ████████████████████  84.4%
  1  ██████████████████████  92.6%   +8.1
  2  ████████████████████▌  87.4%    +3.0
  4  ███████████████  66.7%          −17.8
                                        the nested sets: {10} ⊂ {8,9,10,19}

                        ↓  split by task

        pick_coke_can (75)        move_near (60)
  0        85.3%                     83.3%
  1        92.0%                     93.3%
  2        98.7%  ← 10 fixed         73.3%
  4        78.7%     0 broken        51.7%  ← 3 fixed, 22 broken
           ────────────              ────────────
           goes UP                   goes DOWN
```

**[SAY]**
"Now the same data split by task. And this is the thing I actually want you to
take away.

`move_near` is the only Fractal task where you have to work out **which object
is which** — pick the named one out of three and move it near another named one.
The three coke-can tasks have one instruction and one object.

**[PAUSE — point at the two columns]**

They run in **opposite directions.** At two layers gone, the pick tasks go to
**98.7 percent** — ten episodes fixed, **zero broken**, p equals 0.002, which
clears Bonferroni. The same two layers cost `move_near` ten points. And at four
layers `move_near` is down thirty-two.

**The plus-three aggregate was the average of plus-thirteen and minus-ten.**

**[THE POINT]**

Here's why that matters. Two slides ago, foveation on **OpenVLA** did the same
thing — `move_near` collapsed from 62 to 20 while the coke-can tasks didn't
move. Different backbone, different intervention, different resource deleted.

**Whatever these interventions are removing, it's the capacity for figuring out
which object you meant — not the capacity for moving the arm.** Motor control
isn't just surviving; at two layers it gets *better*.

That's the first mechanism this campaign has produced, and we wrote it down as
a prediction from the foveation data **before** this run existed.

**[the control — expect this question]**

"And before someone says `move_near` is just the fragile task: **action repeat
kills the pick tasks slightly harder** — minus 41 versus minus 38. So the pick
tasks aren't robust in general. They're robust to *this kind* of removal.

Take away re-planning and both families fall together. Take away vision or
depth and they come apart. **Time and capacity fail differently.**"

---

## Slide 4c — What I'd say if you pushed on it (30s)

**[SAY]**
"Three things I'd want said out loud.

**One.** The plus-eight-point-one on its own does not clear Bonferroni. It's
p = 0.013, one of eighteen tests, and unreplicated.

**Two.** The task split confirmed a prediction, but exactly once. And
`move_near` differs from the pick tasks in more than referential load —
horizon, object count, episode count. We can't separate those with the tasks we
have.

**Three.** And the meta-point. **If we had run one cell and reported one
number, we'd have a method paper and we'd have missed the mechanism.** The
aggregate at every dose is an average over two populations moving in opposite
directions. That's the strongest version of our thesis: not just that
single-cell numbers don't transfer — **single numbers hide the thing worth
knowing.**"

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
> It was a property of which backbone, which benchmark, and **which task** you
> measured on.
>
> → not a method paper, but a paper about **how this work is evaluated**
> → and one mechanism out of it: **what gets deleted is WHICH-OBJECT**

**[SAY]**
"To close.

We started out building a method, and found that the method's effect is not a
property of the method. It's a property of which backbone and which benchmark
you measure on.

So this isn't a method paper — it's a paper about how this class of work gets
evaluated. Efficiency papers routinely claim a free speedup from a single
backbone on a single benchmark, and we have a controlled counterexample.

And running the whole grid bought us one thing more. **What these interventions
delete is the capacity to work out which object you meant — not the capacity to
move the arm.** It's confirmed once so far. But it is the kind of thing a single
number would never have shown us."

---

# Expected questions

**Q. So the method is dead?**
> "As a general method claim, yes. But we got a more general *finding* instead,
> and slide 4 has one thing that actually works: on SpatialVLA/Fractal, bypass
> one decoder layer and you get eight percent faster and eight points *better*.
> I'd just never state it without the cell it came from."

**Q. Isn't the +8.1 too good to be true? A pruned model beating the full one?**
> "It's the right thing to be suspicious of, so here's everything against it.
> It's p = 0.013, which does *not* clear our Bonferroni threshold — it's one of
> eighteen tests. We have no measured noise floor; the closest thing we have is
> two measurements of the same baseline in different campaigns differing by 2.1
> points. And one of the four tasks is at 25 out of 25, so we're near a ceiling.
> What's in its favour is that the split is 14 fixed to 3 broken, and the gain
> is spread across three of four tasks rather than carried by one. I'd call it
> real and unreplicated, and re-running the baseline is now my top priority."

**Q. Why would deleting a layer *help*?**
> "Slide 4b is my best answer: it helps the tasks that are pure motor control
> and hurts the one that needs referential grounding. If those layers were
> contributing to object disambiguation and adding noise to the action head,
> you'd get exactly this. But I'd call that consistent-with, not shown. What I
> can say firmly is that the redundancy metric measures *something* real — it
> found a layer that's safe to drop — and does not predict *how many*, because
> its own next picks flip the sign."

**Q. Isn't the task split just post-hoc? You had four tasks and picked a line.**
> "Fair, and here's the one thing that makes it not that. We wrote the split
> down as a hypothesis when we saw the **foveation** result — different
> backbone, different intervention — and said in the report that one task on one
> intervention made it a hypothesis. The depth-prune-2 run happened after that.
> So it's a confirmed prediction rather than a discovered pattern. Confirmed
> once, on one benchmark. The falsification test is `move_near_v1`, which is a
> different scene with the same referential structure and is already in our
> protocol — if it doesn't collapse, the story is about `v0`'s scene, not about
> grounding."

**Q. OpenVLA on Bridge is at 15.6% — isn't it just broken, so anything helps?**
> "That's the right objection and we can't fully rule it out. But it predicts
> the effect should track the baseline — worst policy helped most, best policy
> hurt most. It doesn't. SpatialVLA has the *highest* baseline in the grid, 84%
> on Fractal, and foveation does nothing there at all. OpenVLA at a middling 38%
> loses nineteen points. A floor effect doesn't produce that ordering. So the
> reversal is established; the mechanism isn't, and I'd say that plainly."

**Q. Where exactly does foveation fail on Fractal?**
> "Almost entirely in one task. The three coke-can tasks are flat — 15 of 75
> versus 14 of 75. `move_near` goes 61.7 to 20.0. That's the one task where you
> have to pick which of three objects on the table to move near which other one,
> and log-polar keeps the fovea and throws away the periphery. Grasp rate halves
> too — 68% to 12% — so it's failing to reach, not failing to grip. It's one
> task, so I'd call it a hypothesis, not a result."

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
> "Everything is now chosen to break the grounding hypothesis rather than
> confirm it. First **`move_near_v1`** under depth prune 2 and log-polar — same
> referential structure, different scene, already in our protocol. If it doesn't
> collapse the way `v0` does, the story is about one scene and we need to know
> that before we write it down. Then **OpenVLA/Fractal depth prune** — the depth
> axis is inert on OpenVLA/Bridge, so the aggregate may well be zero, but the
> hypothesis says the *task split* should show up anyway. An aggregate null
> hiding the same split would be much stronger than the +8.1 itself. And a
> baseline re-run for the noise floor, which the +8.1 has made urgent."

---

# Do not

- **Quote a partial cell.** Three moved today when their last task landed:
  +9.3 → +5.2, +1.3 → −1.5, and depth pruning went from −6.7 (p=0.30) to
  **−17.8 (p=0.0002)** — the conclusion itself changed.
- **Compare deltas across columns unguarded.** SpatialVLA sits at 84% on
  Fractal, OpenVLA at 38%. A +5 near the floor and a +5 near the ceiling are
  not the same thing. Always say the baseline alongside.
- **Say "no effect".** Say "no effect larger than about 7 points".
- **Oversell the +8.1.** It does not clear Bonferroni and it has not been
  replicated. Say "plus eight, p = 0.013, one cell, unreplicated" every time.
  Slide 4c exists so that *you* are the one pointing at its limits.
- **State the task split as established.** It is a prediction, made from the
  foveation data, confirmed **once**. And `move_near` differs from the pick
  tasks in horizon and object count as well as referential load. Say
  "consistent with", not "shows that".

# Numbers to memorise (only these)

```
four shapes:  collapse / flat / peak-at-2 / flat-then-cliff   (slide 2)
0%  vs  48%   OpenVLA repeat 4, Bridge vs Fractal             (slide 3)
p = 0.0038                                                    (slide 3)
+18.8 vs −19.3  OpenVLA foveation, Bridge vs Fractal          (slide 3b)
+8.1 at ONE layer,  −17.8 at FOUR,  nested sets               (slide 4)
98.7%  pick tasks at two layers gone — 10 fixed, 0 broken     (slide 4b)
       ...while move_near goes the other way, 83 -> 73 -> 52  (slide 4b)
one sentence:  what gets deleted is WHICH-OBJECT, not HOW-TO-MOVE
```
