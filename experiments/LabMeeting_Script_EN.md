# Lab meeting speaking script — English

Eight slides, about nine and a half minutes. The **[SAY]** blocks can be read as
written. If you are tight on time, slide 3b is the one to drop — but then also
drop the "apply our own thesis" line in slide 4b, because it refers back to it.
Never drop 4b while keeping 4: that leaves the +8.1 standing unqualified, which
is the one thing this talk must not do.
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
| **compute** — decoder layers | 4 of 26 (15%) | **−17.8** | 0.0002 | 1.18× |
| **compute** — decoder layers | **1 of 26 (4%)** | **+8.1** | **0.013** | **1.08×** |
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

**Four of twenty-six layers: minus eighteen points. One of twenty-six: plus
eight.** Same code, same redundancy criterion, same session. Fourteen episodes
fixed, three broken, p equals 0.013. And it runs eight percent faster.

That is a genuine free lunch — the only intervention in this entire campaign
that makes the policy both **better and cheaper**."

---

## Slide 4b — ...and that is exactly the problem (45s) ★the turn

**[SHOW]**

```
SpatialVLA / Fractal, decoder layers bypassed

  0 layers  ████████████████████  84.4%
  1 layer   ██████████████████████  92.6%   +8.1   8% faster
  4 layers  ███████████████  66.7%          −17.8  18% faster
                     ↑
            the sign crosses somewhere in here
            and we have no measurement in between

  the pruned sets are NESTED:  {10}  ⊂  {8, 9, 10, 19}
```

**[SAY]**
"Now look at what that free lunch is sitting next to.

The four-layer set **contains** the one-layer set. We added three more layers
that the *same* redundancy metric ranked as *most redundant* — and it didn't
degrade gradually. It crossed zero and kept going. Comparing the two directly:
**minus twenty-six points, five fixed, forty broken.**

**[PAUSE]**

So here's my honest position on that plus-eight.

Paired, full protocol, significant, eight percent faster. **If we had run that
one cell and stopped, we would have a method paper.** That is what an efficiency
paper is built on.

We ran the neighbours. One step further along the same axis is minus eighteen.
And the whole depth axis is **completely inert on OpenVLA** — plus 2, plus 1,
zero, at one, four and eight layers.

**[if you kept slide 3b]** "Same for the other axis. Vision was the *free* one
on this slide, and two slides ago the identical foveation cost OpenVLA nineteen
points. So *time, compute, vision* isn't a fact about VLA policies — it's a fact
about this cell."

**The free lunch is real. It's also local.** And you can only find that out by
running the grid."

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
> "I don't know, and I'd rather say that than invent a story. The candidate
> explanations are that L10 is actively harmful on this distribution, or that
> it's some kind of regularisation. What I can tell you is that the redundancy
> metric that picked L10 clearly measures *something* real — it found a layer
> that's safe to drop — and equally clearly does not predict *how many* you can
> drop, because its own next three picks flip the sign."

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
> "The action-repeat axis is complete across both backbones and both benchmarks.
> Top of the list now is **depth prune 2 on SpatialVLA/Fractal** — we have +8.1
> at one layer and −17.8 at four and nothing in between, so we can't say where
> the sign crosses. That one run turns 'it depends on the dose' into a curve
> with a knee, which is the only number a practitioner would actually want.
> Then OpenVLA/Fractal **blur** — on Bridge, blur and log-polar agree to within
> a point, so if Fractal blur also lands near −19 the loss is about *how much*
> we removed, and if it stays flat it's about log-polar's geometry specifically.
> And then re-running one baseline twice in a session for the noise floor, which
> the +8.1 has made urgent."

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
  Slide 4b exists so that *you* are the one pointing at its limits.

# Numbers to memorise (only these)

```
four shapes:  collapse / flat / peak-at-2 / flat-then-cliff   (slide 2)
0%  vs  48%   OpenVLA repeat 4, Bridge vs Fractal             (slide 3)
p = 0.0038                                                    (slide 3)
+18.8 vs −19.3  OpenVLA foveation, Bridge vs Fractal          (slide 3b)
−40.0 / −17.8 / ~0  time / compute / vision                   (slide 4)
+8.1 at ONE layer,  −17.8 at FOUR,  nested sets               (slide 4b)
     ...and the whole depth axis is flat on OpenVLA           (slide 4b)
```
