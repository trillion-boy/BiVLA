# Lab meeting script — 10 minutes

Two screens only. **① the report's "One page, for the talk"** → **② the full
grid below it.** Say what is written here and stop.

---

## The one sentence

> **Take the same intervention, the same model, and change only the benchmark —
> and the *sign* of the effect flips. So a number measured on one benchmark is
> not evidence that the method works.**

---

# PART 1 — on "One page, for the talk" (0:00–5:00)

**Opening**

> "We set out to show that a cheap inference-time intervention — hold an action
> for a few steps, foveate the observation, skip redundant layers — buys
> wall-clock without costing success. Then we added a second benchmark and the
> **sign** of the effect changed."
>
> "Three things today: **the sign flips, it isn't noise, and we're halfway to
> knowing why.**"

**Setup**

> "Two SimplerEnv benchmarks — WidowX-Bridge, 96 episodes; Google Robot Fractal,
> 135 episodes. Four backbones: OpenVLA, SpatialVLA, UniVLA, RoboVLMs. Three
> intervention axes: time, vision, compute."

**Statistics — this sentence only. Do not name a test**

> "We don't put two success rates side by side. We **pair the same episodes and
> count only the ones whose outcome changed.**"
>
> (point at the `p` column)
> "This **p** is **the probability of seeing a difference this large if the
> intervention did nothing at all.** Smaller means harder to write off as
> chance; accounting for the 15 comparisons we ran, we fixed the bar at
> **0.0033** in advance. That's what the stars mark."

**Don't say "smaller is better."** Small p means *not chance* — it does **not**
mean *large effect*. Conflate them and you'll be pulled up on it. If asked:

> "Not better so much as **more certain**. The **size** of the effect is the
> delta next to it; p only says whether you can trust that delta. Foveation has
> both — deltas of +18.8 and −19.3, and a small p."

**① The sign flips** — point at the four rows

> "Foveation **gains 18.8 on Bridge and loses 19.3 on Fractal.** Same code, same
> model; only the benchmark differs. Depth pruning does exactly the same thing in
> the opposite direction — −10.4 on Bridge, +8.1 on Fractal."
>
> "Vision and compute survive multiple-comparison correction. Time doesn't yet —
> the sign flips but it isn't established; what *is* established there is a
> difference in **magnitude**."

**② It isn't noise** ← non-removable

> "We re-ran a baseline from scratch and compared **all 85 episodes — success,
> failure, even step counts, all identical.** Re-run variance is zero."
>
> "So we don't need the usual 'mean ± std over 3 seeds.' **There is no
> run-to-run component inside that p at all.** All that's left is which episodes
> the protocol drew — and that is exactly what p measures."

**③ Why it flips — half an answer**

> "Splitting the tasks inside Fractal, the divide isn't the benchmark — it's
> **the capability the task demands.** `pick_coke_can` has one object, nothing to
> select. `move_near` has three, and you must pick the one you were told."
>
> "**Remove capacity and the ability to work out which object was named dies
> first.** The ability to grasp is untouched, or improves. Five cells agree."
>
> "To 'isn't move_near just the weak task' — there's a control. Action repeat 4
> damages **pick harder, −41.3.**"

**Produce the counterexample yourself** ← non-removable

> "There's a result that breaks this. **On OpenVLA, pruning depth at 1, 2, or 4
> doesn't move `move_near` at all** — and the campaign's best number comes from
> there."
>
> "We think we found the cause. One option is read as a fraction in OpenVLA and
> a count in SpatialVLA, so **under one name we were deleting different layer
> ranges.** The range-matched run is going now; we'll know today."

---

# PART 2 — on the full grid (5:00–9:00)

**Pass 1 — across**

> "Each row is one intervention. Read left to right and you see the sign flip I
> just described."

**Pass 2 — down** ★ the strongest moment

> "Now read down. Each column is one backbone-benchmark pair. Take the best
> intervention in each —"

| Column | Winner | That same intervention elsewhere |
|---|---|---|
| OpenVLA / Bridge | foveation log-polar **+18.8** | −19.3 |
| OpenVLA / Fractal | depth prune 4 **+15.6** | −17.8 |
| SpatialVLA / Bridge | action repeat 2 **+12.5** | −8.3, −70.8 |
| SpatialVLA / Fractal | depth prune 1 **+8.1** | −10.4 |

> "**Four columns, four different winners. And all four are negative somewhere
> else in the table.**"
>
> "So to 'which intervention helps,' this table returns **four answers.**"

**Pass 3 — one extreme cell**

> "The most extreme cell: on UniVLA, action repeat 2 is **−70.8** — 78.1%
> collapses to 7.3%. On the **same Bridge benchmark**, SpatialVLA gets **+12.5**
> from the same intervention."
>
> "**Hold the benchmark fixed, change only the backbone, and the sign still
> flips.**"

---

# Closing — grid still up (9:00–10:00)

> "So we can't make a method claim like 'foveation helps' — one benchmark
> doesn't even fix the sign. **That is the result.**"
>
> "Which makes this an evaluation-methodology paper, not a method paper. Three
> claims. One, **a per-episode paired evaluation protocol** — the field
> currently eyeballs two success rates. Two, systematic evidence that
> **single-benchmark numbers are sign-unstable.** Three, **a partial
> mechanism.**"
>
> "Two things remain. The **range-matched run** I mentioned decides the third
> claim, and after that a **third benchmark** — with two, a reviewer says those
> two are peculiar; with three, it's a pattern."

---

## If asked (not before)

**"Which test?"**
> McNemar exact and Fisher exact, Bonferroni-corrected. The arithmetic is at the
> back of the report.

**"The benchmarks differ anyway — isn't this expected?"**
> Not a different magnitude, an **opposite sign**. The opposite of +18.8 isn't
> +5, it's −19.3 — and the stars come from testing that difference directly.

**"How many seeds?"**
> It's deterministic; re-runs are 85/85 identical. Seeds aren't the axis — the
> variation is between episodes.

**"What are the italic `†` cells?"**
> Old-campaign numbers with no per-episode records. They can't be paired, so we
> don't rest sign claims on them.
