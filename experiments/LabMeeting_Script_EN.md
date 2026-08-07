# Lab meeting script — 10 minutes, 5 blocks

Present from the report (`LabMeeting_Bridge_Fractal_0806.md`) on screen.
**Say nothing that is not in this script.** The statistics tutorial, the
hand-computed arithmetic, the legacy cells, the per-task enumeration are all in
the report — open them *when asked*. This script is the talk; the rest of the
report is the defense.

---

## The one sentence

> **Take the same intervention, the same model, and change only the benchmark —
> and the *sign* of the effect flips. So a number measured on one benchmark is
> not evidence that the method works.**

The whole talk proves this one sentence. Everything else is either evidence for
it or a counterexample to it.

---

## Block 1 — The problem (0:00–1:00)

> "We set out to show that a cheap inference-time intervention — hold each
> action, foveate the observation, skip redundant decoder layers — buys
> wall-clock without costing success. Then we added a second benchmark and the
> **sign** of the effect changed. I'll say three things today: the sign flips,
> it isn't noise, and we're halfway to knowing why."

Count them on your fingers. That's the outline.

---

## Block 2 — The setup (1:00–2:30)

Open the grid at the top of the report. Three things only.

> "Two SimplerEnv benchmarks: WidowX-Bridge, 96 episodes; Google Robot Fractal,
> 135 episodes. Four backbones — OpenVLA, SpatialVLA, UniVLA, RoboVLMs. Three
> intervention axes: **time** (hold each action k steps), **vision** (foveation,
> keeping 20% of pixels), **compute** (bypass redundant decoder layers)."
>
> "The part that matters is how we compare. We don't put two success rates side
> by side — we **pair episode by episode, by ID, and count only the episodes
> whose outcome changed.** Almost nobody in this area does that."

### End the statistics in one sentence

Do **not** say McNemar, Fisher, or Bonferroni out loud. Do say this. Without it,
"isn't that just luck?" takes down the whole grid in one question; with it,
nothing takes it down.

> **"Every cell pairs the same episodes and computes the probability that the
> difference between the two benchmarks is chance. The starred cells are the
> ones chance cannot explain even after accounting for how many comparisons we
> ran."**

That one sentence carries pairing, p-values, and multiple-comparison correction,
and names none of them. If asked which test, answer in one line — *"McNemar
exact and Fisher exact, Bonferroni-corrected; the arithmetic is in the
report"* — and stop.

---

## Block 3 — Three passes over the grid (2:30–6:30) ★ the body of the talk

Put the full grid up and **point at it**. Read 50 cells in no order and you lose
20 minutes. Three passes.

### Pass 1 — across. One row = one intervention (1 min)

> "Each row is one intervention. Look left to right and the sign splits."

| Axis | Intervention (backbone) | Bridge | Fractal | interaction p |
|---|---|---|---|---|
| vision | foveation log-polar (OpenVLA) | **+18.8** | **−19.3** | 0.0000055 *** |
| vision | foveation blur (OpenVLA) | **+17.7** | −8.9 | 0.0017 *** |
| compute | depth prune 1 (SpatialVLA) | **−10.4** | **+8.1** | 0.0018 *** |
| time | action repeat 2 (OpenVLA) | **−8.3** | **+5.2** | 0.0266 |

> "Foveation **gains 18.8 on Bridge and loses 19.3 on Fractal.** Same code, same
> model, same hook point; only the benchmark differs. Depth pruning does exactly
> the same thing in the opposite direction."
>
> "Vision and compute survive multiple-comparison correction. Time does not yet
> — the sign flips but the interaction isn't established. What *is* established
> on the time axis is a difference in **magnitude**."

That last sentence buys your credibility. Say it.

### Pass 2 — down. Name the winner of each column (2 min) ★ strongest moment

| Column | Winner | That same intervention elsewhere |
|---|---|---|
| OpenVLA / Bridge | foveation log-polar **+18.8** | −19.3 |
| OpenVLA / Fractal | depth prune 4 **+15.6** | −17.8 |
| SpatialVLA / Bridge | action repeat 2 **+12.5** | −8.3, −70.8 |
| SpatialVLA / Fractal | depth prune 1 **+8.1** | −10.4 |

> "Now read down. Each column is one backbone-benchmark pair. Take the best
> intervention in each — **four columns, four different winners. And all four
> winners are negative somewhere else in the table.**"
>
> "So to the question 'which intervention helps,' this table returns **four
> answers.** Without naming the cell you measured in, the question doesn't
> parse."

### Pass 3 — one extreme cell (1 min)

> "One last cell. On UniVLA, action repeat 2 is **−70.8** — 78.1% collapses to
> 7.3%. On the **same Bridge benchmark**, SpatialVLA gets **+12.5** from the
> same intervention."
>
> "**Hold the benchmark fixed, change only the backbone, and the sign still
> flips.** So this isn't only a claim about benchmarks."

This is stronger than the one-page summary. The summary says "change the
benchmark"; reading the grid downward gets you "change the backbone" for free.

---

## Block 4 — ② It isn't noise (6:30–7:30)

> "The obvious question is whether this is luck. Two answers."
>
> "First, **the pipeline is fully deterministic.** We re-ran a baseline from
> scratch and compared **all 85 episodes across two environment classes — every
> success flag, every step count, every grasp flag was identical.** Greedy
> decoding into a seeded environment, so re-run variance is **zero**."
>
> "Why that matters: papers usually report 'mean ± std over 3 seeds.' Our re-run
> variance is 0, so **the only remaining uncertainty is between-episode
> variation, which is exactly what a paired test handles.** The p-value is the
> *complete* account of uncertainty, not a partial one."
>
> "Second — as I said, we computed how likely those flips are by chance, and
> they survive a threshold that already accounts for running 15 comparisons.
> Foveation comes out at **55 in ten million**; depth prune 1 at **1.8 in a
> thousand**."

**Stop there.** Still no test names. If asked, open "How the statistics work" in
the report — the arithmetic is there.

---

## Block 5 — ③ Why it flips: half an answer, and the counterexample (7:30–9:00)

Open the task-family split.

> "We looked at *why* the benchmarks differ, and it turned out not to be the
> benchmark — it's **the capability the task demands.** Fractal splits in two."
>
> - `pick_coke_can` — one object on screen. **Nothing to select.**
> - `move_near` — three objects; you must resolve **which one was named**.

| Intervention | pick (single target) | move_near (referential) |
|---|---|---|
| OpenVLA log-polar | −1.3 | **−41.7** *** |
| OpenVLA blur | +9.3 | **−31.7** ** |
| SpatialVLA prune 2 | **+13.3** *** | −10.0 |
| SpatialVLA prune 4 | −6.7 | **−31.7** *** |
| SpatialVLA prune2+repeat2 | **+10.7** ** | **−16.7** ** |

> "Remove capacity and **the ability to work out which object was named dies
> first.** The ability to grasp is untouched, or improves. Five cells agree."
>
> "The obvious objection — 'move_near is just the fragile task' — has a control.
> **Action repeat 4 damages pick harder: −41.3.** So it isn't that stronger
> interventions break move_near first; it's that a *particular kind* of
> intervention does."

### Then produce the counterexample yourself (the most important 30 seconds)

> "There's a result that breaks this. In OpenVLA, pruning depth at 1, 2, and 4
> moves `move_near` by **+8.3, +0.0, +8.3** — it doesn't budge. And the
> **single best number in the whole campaign (Fractal +15.6, p<0.001)** comes
> from there."
>
> "We think we found the cause. `--depth-min-layer` is read as a **fraction** in
> OpenVLA and as a **count** in SpatialVLA. So under one name, 'depth prune 4,'
> OpenVLA was deleting the back half (L17–26) and SpatialVLA the middle (L8–19).
> **We were calling two different experiments by the same name.**"
>
> "We're running the range-matched version now. The first task already selected
> early layers (L2, L4) and success collapsed to **16%**, against 54.1% when
> only the back half was removed. We'll know today."

> **Why lead with the counterexample:** any reviewer or mentor will find it.
> Volunteering it reads as rigor; being caught on it reads as oversight.

---

## Block 6 — What this becomes (9:00–10:00)

> "So we can't make a method claim. 'Foveation helps' isn't sign-stable on one
> benchmark. **That is the result.**"
>
> "This is an evaluation-methodology paper, not a method paper. Three claims:"
>
> 1. **A per-episode paired protocol with verified determinism for VLA
>    evaluation.** The field currently eyeballs two success rates.
> 2. **Single-benchmark deltas are sign-unstable** — 3 axes × 2 benchmarks ×
>    4 backbones, surviving multiple-comparison correction.
> 3. **A partial mechanism** — removing capacity degrades referential grounding
>    first. Five confirmations, one disconfirmation.
>
> "Two things remain. The **range-matched depth run** decides claim 3 — results
> today. Then a **third benchmark**: with two, a reviewer says those two are
> peculiar; with three, it's a pattern."
>
> "We're targeting a robot-learning evaluation workshop at CoRL or NeurIPS
> first — venues that accept measurement results like this."

---

## Three questions you will get

**Q. The benchmarks are just different — isn't a different result expected?**
> Not a different magnitude — an **opposite sign**. Not "+18.8 on Bridge, maybe
> +5 on Fractal" but **−19.3**. And we tested that difference directly:
> p = 0.0000055.

**Q. How many seeds?**
> Seeds aren't the axis. The pipeline is deterministic — 85/85 episodes
> identical on re-run. The variation is **between episodes**, which is precisely
> what the paired test addresses.

**Q. What is McNemar / Fisher?**
> (Open "How the statistics work" in the report.) We count only the episodes
> whose outcome **changed** — the unchanged ones carry no information. 17
> changed; the chance of 3 or fewer landing on one side by coin flip is 0.0127.
> Fisher answers the next question: is the change rate on Bridge different from
> the change rate on Fractal?

---

## Do not

- Explain p-values from first principles → only if asked
- Read the hand-computed McNemar arithmetic → it's in the report
- Bring up legacy cells → if asked: "old-campaign numbers, unpaired, so we don't
  rest sign claims on them"
- Enumerate all four tasks → point at the table
- Apologize for the `--` cells → "in progress," and move on
