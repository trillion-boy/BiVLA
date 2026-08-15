# Training-Free Inference-Time Interventions Do Not Transfer to VLA

> **This work is split across three documents.**
>
> | Document | What | Length |
> |---|---|---|
> | `Overview_EN.md` | the whole project in 10 minutes — **start here if you are new** | ~470 lines |
> | **This one** | **how we measured (§3), what came out (§4–6), what remains (§7)** | ~2,300 lines |
> | `RelatedWork_EN.md` | prior work (§2) and per-paper notes (Appendix A) | ~1,240 lines |
>
> This document is not written to be read front to back. It is the evidence
> file you come to **when a specific number has to be defended.**

**The evidence document of this repository.** Every grid table, the close
reading of prior work, the correction log, and the machine verification live
here. Someone **taking over the project** should get the big picture from
`Overview.md`, then come down here when a specific section is needed.

**Every grid number** is generated from the per-episode records in `results/`
by `python experiments/build_grid_report.py`. There are, however, **three
values that come from outside the grid, and all three are marked in the
text** — the three ms values in §3.5.2 (the run was stopped, so no result
files exist), the right-hand column of §3.8 (the backbone papers themselves),
and the chunk-execution numbers in §2.2 (c) (an early exploratory record).
The table in §7.2 shows at a glance which values come from records and which
do not.

> **Writing rule.** This document states plainly only **what we measured**
> and **what we checked against the original papers.** Everything else
> carries a qualifier — hypotheses are called hypotheses, unverified columns
> are called inference, and negative claims ("none exists") are written as
> "we could not find." For a Δ that is not significant, we state the
> direction only and make no claim about size. New sentences must follow
> this rule too.

> **Status (2026-08-14).** **The grid and all diagnostic runs are complete**
> (§7.0). The three remaining open items need a new benchmark or a new
> checkpoint, so no amount of extra simulation opens them (§7).

---

## Table of contents

| Section | Content | Who reads it |
|---|---|---|
| §0 | one-page summary | everyone |
| §1 | background — what this field claims, what we test | everyone |
| §2 | **prior work** → **moved to `RelatedWork.md`** | **mentor / paper §2** |
| §3 | setup — backbones, benchmarks, pairing protocol, determinism | anyone reproducing this |
| §4 | results by intervention | everyone |
| §5 | results across cells — sign flips, variant choice, cost | everyone |
| §6 | mechanism — a hypothesis we built and rejected with our own data | everyone |
| §7 | what is settled and what is open | whoever takes over |
| **Appendix A** | **per-paper close-reading notes (7 questions)** → **moved to `RelatedWork.md`** | **mentor / whoever takes over** |

---

# §0. One-page summary

**One sentence.** Take the same intervention and the same model, change
**only the benchmark**, and the effect's **sign flips.** So a number measured
on one benchmark is not evidence that the method works.

**What we did.** We took three interventions that need no training — **action
repeat** (hold an action for k steps), **foveation** (concentrate the
observation at the center; two variants, log-polar and blur), and **depth
pruning** (skip redundant decoder layers) — put them beside the original
policy as four conditions, and re-measured everything across **three
backbones** (OpenVLA, SpatialVLA, UniVLA) and **two benchmarks** (SimplerEnv
WidowX-Bridge, 96 episodes; Google Robot Fractal, 135 episodes). We never set
two success rates side by side — **we pair the same episodes and count only
the ones whose outcome flipped.**

> ⚠️ **The 3×2 grid is not full.** OpenVLA and SpatialVLA run both
> benchmarks, but **UniVLA runs Bridge only.** Its authors evaluated
> SimplerEnv only on Bridge, and the public checkpoint is Bridge-only.
> Putting a Bridge-trained policy on Google Robot would measure domain
> transfer failure, not the intervention — so we leave that cell empty.
> **The benchmark axis therefore rests on two backbones, and the backbone
> axis on three.** §5 returns to the fact that the two axes do not use the
> same set of cells.

**What came out.**

1. **The sign flips.** We name the backbone and condition for each number —
   these come from different cells, and lumping them together would read like
   one model's story.

   | Intervention | Backbone | Bridge | Fractal |
   |---|---|---:|---:|
   | foveation (log-polar, keep 20%) | OpenVLA | **+18.8** (p = 0.0051) | **−19.3** (p = 0.0004) |
   | depth pruning (1 layer) | SpatialVLA | −10.4 (p = 0.0872) | +8.1 (p = 0.0127) |
   | action repeat 2 | SpatialVLA | +12.5 (p = 0.0428) | ±0.0 (p = 1.0000) |
   | action repeat 2 | UniVLA | **−69.8** (p < 0.0001) | *(no checkpoint)* |

   Same code, same model — only the benchmark differs. Changing the backbone
   splits things too: on the same Bridge, action repeat 2 drops UniVLA
   sharply (**−69.8**) while SpatialVLA does not drop (+12.5). What is
   certain here is **the large drop on UniVLA's side**; whether SpatialVLA's
   side is a real gain, this data cannot settle (p = 0.0428).

   **The backbone axis also has one sign flip where both sides are
   significant.** Same Fractal, same 135 episodes, same `depth prune 4` —
   **+15.6 on OpenVLA and −17.8 on SpatialVLA.** Both cells pass the
   correction individually, and their difference is p = 4.3 × 10⁻⁷ (§5.2).
   So the backbone axis shows not just "costly on one side, not on the
   other," but goes all the way to **"helpful on one side, harmful on the
   other."**

   > ⚠️ **The strength of evidence differs per sign flip.** The grid runs
   > **38 paired tests**, so Bonferroni gives α = 0.05/38 ≈ 0.0013. By that
   > bar, what passes is **foveation's Fractal side, −19.3 (p = 0.0004), and
   > UniVLA's −69.8** — while **the Bridge side's +18.8, at p = 0.0051, does
   > not pass.** An earlier draft wrote "both foveation values," which did
   > not match the table (§7.1 ②).
   >
   > **The sign flip itself still stands.** The Bridge-side plus is
   > established not by the grid cell but by **the capacity sweep** — keep
   > 100% gives +30.2 at p = 4.2 × 10⁻⁷ (§4.3 b). So the precise sentence
   > is: "gain on Bridge and loss on Fractal are both established, and the
   > Bridge-side evidence is the capacity curve, not the keep-20% cell."
   >
   > **Depth pruning's −10.4 / +8.1 both fail the bar** (p = 0.0872,
   > 0.0127). That axis's flip is **observed but not distinguishable from
   > chance** (§3.6 ④).
2. **It is not noise.** A re-run that changed nothing reproduced **85 of 85
   episodes** exactly — success, step count, even grasp — and a third
   backbone reproduced 24/24 in a separate check. The §6 re-run matched
   60/60. **With hardware held fixed**, run-to-run variance is zero in every
   case we checked, so the only remaining uncertainty is which episodes got
   sampled — which is exactly what our reported p-values measure.
   > What we checked directly is **two columns: SpatialVLA/Fractal and
   > UniVLA/Bridge.** For OpenVLA it is an **inference** that the same
   > decoding settings behave the same way (§3.4).
3. **We built a mechanism hypothesis and rejected it with our own data.** If
   the splits came not from the benchmark but from the ability each task
   demands, the grid seemed explainable — the reading was that cutting
   capacity **damages the ability to pick out the named object first.** The
   evidence was a single **between-task** comparison — `move_near` at −31.7
   vs. `pick_coke_can` at −6.7 — and those two tasks differ in five ways at
   once.
   > So we designed a measurement that counts failure kinds **inside** one
   > task (§6.2). Measured in the cell with the largest drop
   > (SpatialVLA/Fractal), **the predicted signal did not appear.** What
   > grew was not "moved the wrong object" (1 → 4, p = 0.375) but **"touched
   > nothing" (0 → 12, p = 0.0005).** What was damaged is not the pointing
   > but **the doing** (§6.4).

   **What remains is narrower.** Count the eleven conditions **as shares of
   failures** and `wrong_object` is a minority everywhere, 0–14%, and **does
   not grow with intervention strength** — in the campaign's lowest-success
   condition (4/60) it is 12.5%, while that backbone's baseline is already
   10.0%. Failures concentrate in **"touched nothing,"** and the share rises
   monotonically with strength only on **SpatialVLA, the one backbone with
   three points.** On OpenVLA it is not monotone (§6.5 ②).

   **The task split itself is not our discovery.** The pattern — `move_near`
   down, `pick_coke_can` up — **already sits in the published tables of two
   prior author groups**, and no paper discusses it in the text, because
   they all report only the 4-task average (§2.5). **That the split exists**
   is confirmed that way; **why it splits**, our §6 could not answer.
4. **Cost differs per axis, and the amount saved predicts nothing.**
   Foveation saves **≈0%** across the five cells (measured −3.1% to +2.7% —
   noise range; §4.3 a). Depth pruning saves in proportion to the layers
   (−11 to −16% at 4 layers). Only action repeat saves a lot (−50% / −75%)
   — and it produces the campaign's largest losses. **Across axes, we
   observe a tendency: the more compute an intervention saves, the more it
   loses.**
   > One exception. OpenVLA depth prune 4 keeps its sign on both benchmarks
   > (Bridge +1.0, Fractal +15.6) and saves −11%. So we cannot write "no
   > cell saves compute and keeps its sign." The precise statement: our grid
   > has **no cell that gives a significant gain on both benchmarks while
   > saving compute** — the Bridge side's +1.0 is p = 1.0000,
   > indistinguishable from chance.

5. **★ Within an axis, saving and outcome are not correlated at all.** The
   depth axis's +15.6 was the campaign's largest gain, so we shook that
   setting. **Holding backbone, benchmark, method, capacity (4 layers), and
   compute saved (−10.6 to −11.9%) all fixed**, we moved the candidate
   window and one selection rule.

   | Deleted layers | Candidate window | Δ | p |
   |---|---|---:|---:|
   | [17,23,25,27] / [17,20,23,26] *(varies by task)* | L16–31 | **+15.6** | 0.0011 |
   | [2,4,23,26] | L2–31 | +5.9 | 0.302 |
   | [17,23,27,31] | L16–31, gap 3 | +1.5 | 0.868 |
   | [28,29,30,31] | L28–31 | **−30.4** | 5.9 × 10⁻¹¹ |

   **45.9 points come out of one flag.** Delete the last four layers and the
   three pick tasks all go to **0/25**, with all 75 episodes running to the
   step cap. So a number reported as "depth pruning, 4 layers, +15.6" is not
   the method's performance but **a function of where that implementation
   put its candidate window.** By amount of control, this is stronger
   evidence than the benchmark-axis flip in point 1.

   **And the size of that swing itself depends on the condition — it follows
   the benchmark.** We ran the same contrast in all five cells.

   | Cell | Default window | Shifted-back window | Paired gap |
   |---|---:|---:|---:|
   | SpatialVLA / Fractal | −17.8 | −68.1 | **50.4** |
   | OpenVLA / Fractal | +15.6 | −30.4 | **45.9** |
   | UniVLA / Bridge | −2.1 | −8.3 | **6.3** |
   | OpenVLA / Bridge | +1.0 | −4.2 | **5.2** |
   | SpatialVLA / Bridge | −28.1 | −30.2 | **2.1** |

   **The two Fractal cells sit at 45.9 and 50.4, the three Bridge cells at
   2.1, 5.2, 6.3 — with a 40-point gap between the groups.** Baseline height
   does not explain it — the Bridge cells' baselines are scattered from
   15.6% to 81.2%, yet all three gaps are small. So the claim is not "it is
   sensitive to layer choice" but **"the sensitivity itself depends on the
   condition."** A paper that concluded "robust to layer choice" on one
   benchmark could swing 45.9 points on another (§4.4 c).

6. **On the vision axis, the premise itself fails.** To see whether
   foveation's gain comes from compression, we swept `keep` — and **the more
   we discarded, the worse it got**: 10% +4.2 / 20% +18.8 / 40% +19.8 /
   **100% +30.2**, all four points on 96 episodes. The setting that discards
   nothing is the campaign's largest and most significant gain
   (p = 4.2 × 10⁻⁷). So the gain comes from **the log-polar geometry
   itself**, not from compression — and the compute saved is ≈0%. **Our
   measurements give no grounds to classify this as an efficiency
   intervention** (§4.3 b).

**So this is not a methods paper — it is an evaluation-methodology paper.**
Three contributions: a **per-episode pairing protocol** with verified
determinism; systematic evidence that single-benchmark Δs are
**sign-unstable**; and **the record of building a mechanism hypothesis and
rejecting it ourselves** (§6) — that last one is a contribution as a case of
applying our own standard to ourselves.

---

# §1. Background — what this field claims, and what we test

VLA policies are slow. For some models one observation takes 2.8 seconds
(UniVLA on an L4: **2.81 s** — `avg_model_ms_per_infer` 2811.5 ms). So
interventions that **use less compute at inference while keeping success**
keep being proposed. What they share is that **no training is needed** — they
change the execution path without touching the weights. That makes them
cheap, and that is why there are so many.

The claim such a paper makes usually has this shape:

> We applied intervention M to backbone A. On benchmark B, FLOPs dropped X%
> and success only dropped Y points. Therefore M is an efficient
> intervention.

For that to hold, the effect must be **a property of the method.** The
direction has to survive a change of backbone or benchmark — otherwise "M
works" does not mean anything.

**That assumption is exactly what we test.** We propose no new method. We
take three existing interventions, cross backbone with benchmark, and watch
whether the sign holds.

The genre is not new. The Bag-of-Tricks line of work takes methods other
people built, re-measures them across conditions, and reports **what
actually reproduces.** In that genre, **the methods being pre-existing is
not a flaw — it is the precondition.** The more widely used the method, the
more "why measure this" answers itself.

---

# §2. Prior work

> **This section moved to `RelatedWork.md`** (§2.1–2.6 and Appendix A, about
> 1,230 lines). Section numbers are unchanged, so references like "§2.5" and
> "Appendix A.3" in this document resolve there directly.

One-paragraph summary — **the three axes were chosen** because they touch
different resources (§2.1), and each axis is laid out as **origin → what it
does → how it came to VLA → what has been tried → what nobody measured**
(§2.2–2.4). **§2.5 matters most** — inside the published tables of the prior
papers themselves, `pick coke can` rises and `move near` falls in all 12
settings, and none of the texts mentions it; they all report only the 4-task
average. §2.6 places us in the evaluation-methodology genre and lists the
closest neighbors — **in the range we checked, no prior work crossed the
backbone axis and the benchmark axis with paired episodes.**

# §3. Experimental setup

## 3.0 Implementation spec of the three interventions

**Without this section there is no Method.** §2 says what the literature
does, and §3.5 lists the layers actually deleted, but nowhere was **the
exact operation we applied** written down in one place. Here it is — and the
guarantee that it is **the same operation** on all three backbones is the
precondition for comparing grid cells.

| Intervention | Exactly what it does | Parameters | Held fixed |
|---|---|---|---|
| **action repeat k** | Call the policy **once every k environment steps** and **execute that one prediction k times** (`for _ in range(k): env.step(a)`). k=1 is the original loop. | k ∈ {2, 4} | the policy and the whole observation pipeline |
| **foveation (log-polar)** | Warp the image **to log-polar → resample at keep% density → interpolate back to size → inverse-warp.** The warp round trip **always** happens, regardless of keep (§4.3 b). | keep ∈ {10, 20, 40, 100}%, center = image center, always on | image size and token count (which is why no compute is saved) |
| **foveation (blur)** | **Preserves geometry.** A central disc (area ≈ keep%) stays **bit-for-bit original**; outside it, a Gaussian blends in more strongly with radius. | keep = 20%, center = image center | same as above |
| **depth pruning k** | Rank decoder layers by **Block Influence** (`1 − cos(layer input, layer output)`) and **bypass** the k most redundant. Weights untouched. | k ∈ {1,2,3,4,8}, candidate window (`min-layer`), adjacency rule (`min-gap`) | the selection rule — **same rule** per backbone, different outcome |

> ⚠️ **`keep` means different things in the two variants.** Blur's keep is
> **the area fraction of the central disc left original**; log-polar's keep
> is **the fraction of samples kept in the warped image.** Same name,
> different referent — write "keep 20%" without saying which and the two
> conditions read as one. This is also why keep=100% is not the original
> image: discard nothing and the warp round trip still happens (§4.3 b).

**Calibration is once per run.** BI is measured on that run's **first
observation** (via a hook), the layer set is chosen, and it stays fixed to
the end. It is not re-measured per episode (the §2.4 correction). Layers
differ per task because each task is a separate process calibrating on its
own first observation.

> ⚠️ **How far "the same intervention on three backbones" actually holds.**
> Action repeat and foveation are operations outside the policy, so the
> three implementations are literally identical. **Depth pruning shares only
> the rule; the implementations are three** — argument meaning (fraction vs.
> count), last-layer protection, and how action tokens terminate all differ
> (§3.5.1). So §4.4's comparison is "the same rule applied to each
> backbone," not "the same layers deleted."

## 3.1 The three backbones

| Backbone | Architecture | Decoder layers | Latency per observation |
|---|---|---|---|
| OpenVLA | Llama 2 7B-based (7B) | 32 | — |
| SpatialVLA | PaliGemma 2-based (Gemma 2 decoder) | 26 | ~0.9 s |
| UniVLA | Emu3-based, 8.5B | 32 | **2.81 s** (L4) |

## 3.2 The two benchmarks

| Benchmark | Tasks | Episodes |
|---|---|---|
| SimplerEnv **WidowX-Bridge** | eggplant, carrot, stack cube, spoon | 4 × 24 = **96** |
| SimplerEnv **Google Robot / Fractal** | coke can, 3 poses × 25 + `move_near_v0` × 60 | **135** |

Each task runs **every episode the protocol defines** (`--n-episodes 0`). We
never truncate — `move_near`'s episode ids are grouped by object triplet, so
cutting from the front gives not a smaller sample but a **biased** one. That
is what once made `move_near` read 91.7% at n=24.

> ⚠️ **We did not use all of Fractal's task families.** The harness
> (`simpler_fractal_protocol.py`) registers **eight** Google Robot tasks; our
> grid uses **four.**
>
> | Used (4) | Not used (4) |
> |---|---|
> | coke can ×3 (25 each), `move_near_v0` (60) | `open_drawer` (24), `close_drawer` (24), `place_in_closed_drawer` (24), and `move_near` (the non-v0 variant — same family as our `move_near_v0`, so not run separately) |
>
> The three drawer tasks are **articulated-object manipulation** — a third
> ability type that belongs to neither of our two families (single-target
> picking / picking out a named target). And the SpatialVLA paper's Table I
> reports Google Robot in **three categories** (pick coke can, move near,
> **open/close drawer**) — so we used **two of the three** categories of the
> standard protocol.
>
> **Why not.** Grid uniformity (§3.6). Adding drawers means adding them to
> **all eight conditions** of both Fractal columns, and their per-episode
> step caps (113–200) are much longer than coke can's (80) — over 15 hours
> for the two columns. We spent that time on the depth-selection contrast
> (§4.4 c) and the §6 mechanism measurement instead.
>
> **So the strongest sentence this work can say is "in Fractal's two task
> families."** Not "confirmed across diverse tasks." Left open in §7 ③.

## 3.3 The pairing protocol — this is part of the contribution

We do not put two success rates side by side and subtract. **We pair the
same episode indices and count only the pairs whose outcome changed.**

- **McNemar's exact test** on the **discordant pairs** only. Episodes where
  both sides succeed, or both fail, carry no information and drop out.
- Whether two cells respond differently (the interaction) is a **Fisher
  exact 2×2.**
- Multiple comparisons are corrected with **Bonferroni.** **The grid
  actually runs 38 paired tests**, so α = 0.05/38 ≈ 0.0013.
  > **Why 38.** Five cells × seven conditions = 35, plus `depth prune 8`
  > run in two cells (OpenVLA/Bridge · UniVLA/Bridge) and `depth prune 2 +
  > action repeat 2` run in one (SpatialVLA/Fractal). This number is not
  > counted by hand — **the row count of `build_grid_report.py`'s paired
  > table is the denominator.**
  >
  > **The family size does not change the conclusion.** **Eight cells
  > pass**, and rank 8 sits at p = 1.07 × 10⁻³ while rank 9 sits at
  > p = 5.10 × 10⁻³ — the gap between them is empty, so **any denominator
  > among 35, 38, and 42** lands the cutoff inside that gap. (Earlier
  > drafts said 15 tests with α ≈ 0.0033, then 42, then 35. **None of the
  > three was the number of tests actually run** — the last one missed
  > `prune8` and the combined condition.)
  Off-grid runs (the capacity sweep, the window sweep, the mechanism
  measurement, the determinism re-checks) count as **separate test
  families** (different sense from the preamble's "values from outside the
  grid," which is about provenance; this is about **where the multiple-
  comparison family is drawn**). They are follow-ups explaining one cell,
  not cell-to-cell comparisons. Which family a number belongs to is stated
  at each citation.
- A condition that only partly finished is paired **only on episodes present
  on both sides**, so it is compared against the matching part of the
  baseline, not the whole protocol.

## 3.4 Determinism — why the p-value is the entire uncertainty

With `do_sample=False` and a seeded environment, a re-run must be
bit-identical — and it is.

- We re-ran the SpatialVLA/Fractal baseline on two tasks **with different
  environment classes**: **85/85 episodes** identical down to
  success/failure, step count, and grasp.
- UniVLA, in a separate check, was **24/24 identical** (success and step
  count both).

**What we checked directly is two columns** — SpatialVLA/Fractal and
UniVLA/Bridge. We did not run the re-check for OpenVLA, so for that column
it is an **inference** that the same decoding settings (`do_sample=False`)
and seeded environment behave the same. Within what we checked, **Δ carries
no run-to-run component**, and the only remaining uncertainty is which
episodes the protocol includes. That is exactly what paired McNemar
measures. So the reported p-value is not part of the uncertainty — **it is
all of it.** No need to run several seeds and report mean ± std.

> ⚠️ **One qualifier: with hardware held fixed.** Change the GPU and the
> floating point changes, and the closed loop amplifies it. Measured
> directly, when UniVLA/Bridge foveation was re-run on a different card (the
> early Colab L4 run vs. `results/univla_bridge_0805/`):
>
> | Condition | Mean | Per task |
> |---|---|---|
> | log-polar | **86.5% → 86.5%** (identical to the decimal) | **all four tasks moved** (−4.2 / +4.2 / +8.3 / −8.3) |
> | blur | 76.0% → 72.9% (**−3.1**) | three moved; `spoon` by **−20.8** |
> | *(control)* baseline | 78.1% → 78.1% | **0/4 — not one episode changed** |
>
> **A matching mean is not evidence of reproduction** — the log-polar row is
> the counterexample. All four tasks moved, the moves canceled, and the mean
> matched to the decimal. It is the same failure mode we point out in others
> in §2.5, appearing once in our own data — which is why
> `experiments/compare_runs.py` compares episode vectors, not success rates.
>
> **The baseline row matters.** Under the same card change, the original
> policy stayed 96/96. So what wobbled was not execution in general but
> **the foveation path**, and whether the cause is the GPU or a build
> difference in `cv2`'s warp we could not tell. **Either way the conclusion
> is the same — a condition and its baseline must come from the same
> environment.**

### 3.4.0 But whether the grid actually held hardware fixed cannot be proven from the records

Apply the qualifier above to our own grid and one thing remains to check.
**The result files do not record the GPU.** Inference time cannot identify
it either — the two UniVLA baselines below differ in 11 episodes while their
`ms/infer` reads 2801 vs. 2811, practically the same. By commit dates, one
cell's conditions span several sessions (e.g., SpatialVLA/Bridge: baseline
on 08-05, conditions 08-06 through 08-10). **So "all five cells ran on the
same card" is not a sentence we can prove.**

**What we can do instead is measure the size of the effect.** UniVLA/Bridge
has **two baselines with identical settings** — `baseline` from 08-05, and
`baseline_l4`, re-taken on a matched card, from 08-10. They differ in **11
of 96 episodes, 3.1 points on the mean** (p = 0.5488, indistinguishable from
chance). We attached every condition in that cell **to each baseline in
turn.**

| Condition | vs `baseline` | vs `baseline_l4` (what the grid uses) | Conclusion changes? |
|---|---:|---:|---|
| action repeat 2 | −66.7 (p = 1.8e−18) | **−69.8** (p = 2.4e−19) | no |
| action repeat 4 | −78.1 | **−81.2** | no |
| `prune4_mid` | −76.0 | **−79.2** | no |
| depth prune 1 | ±0.0 (p = 1.0) | **−3.1** (p = 0.63) | **sign changes, but both are meaningless** |
| depth prune 4 | +1.0 (p = 1.0) | **−2.1** (p = 0.80) | **same as above** |
| foveation log-polar | +8.3 (p = 0.13) | **+5.2** (p = 0.36) | no |
| foveation blur | −5.2 (p = 0.38) | **−8.3** (p = 0.15) | no |

**Every Δ shifts in parallel by exactly 3.1 points; no cell's significance
flips.** The two whose sign changes (prune 1, prune 4) are cells with
p ≥ 0.6 to begin with — cells where we claim no direction.

> **So the size of this limitation can be bounded at "about 3.1 points."**
> The values the grid actually argues from — −69.8, −40.0, −28.1, −19.3,
> +18.8, +15.6 — are all an order of magnitude larger. **So hardware
> confounding cannot flip the conclusions** — but the sentence we can defend
> is not "we fixed the card"; it is **"even if we failed to, the effect does
> not exceed this size."** Left open in §7 ⑤.

### 3.4.1 Then what is the p-value the probability of? — stating the premises

**"If it is deterministic, what is left for a p-value to measure?" is a fair
question.** The answer lies in the premises, and since we chose them, we
write them down.

McNemar's null hypothesis is **"this intervention changes no episode's
outcome."** Under it, the episodes whose outcome did change (the discordant
pairs) should split evenly in either direction, and what we measure is **how
far the observed broke/fixed split deviates from that symmetry.** No model
randomness is needed for this — and indeed there is none.

So the question this p-value answers is:

> **Over the episodes of this protocol, if the intervention had no
> directional effect at all, what is the probability of a split this
> lopsided?**

**Two premises, stated.**

**① We treat the episode set as a sample of the task distribution.**
SimplerEnv's protocol is fixed — Bridge: 24 per task; Fractal: coke can
25 × 3 and move near 60. We did not choose it; the benchmark did, and **the
sample cannot be enlarged.** So the p-value is about "if the split were
symmetric within these episodes," not "if the protocol were redefined."
**It makes no claim that generalizes to other sets of initial states.**

**② We treat episodes within a task as exchangeable.** `move_near`'s 60 are
object-triplet-and-placement combinations; coke can's 25 are a 5×5 position
grid. We know they differ in difficulty (§4.4 d's family split is the
evidence). The test **does not model individual difficulty** — it cancels it
by pairing: both conditions share the same episode, so difficulty drops out
of both arms together. That is why pairing is required, and why subtracting
two success rates destroys the cancellation.

**Two claims we therefore do not make.** First, **no confidence
intervals** — with zero run-to-run variance, "run it many times and get this
range" has no referent. Second, we never write **"this intervention gains X
points on this task on average."** What we can write is at most: **"on
these episodes of this protocol it moved X points, and the probability that
lopsidedness is chance is p."**

> The same distinction carries into §5's Fisher tests. There, two cells
> share no episodes, so nothing can be paired; what is compared is **the
> ratio of the discordant splits.**
> So §5 tests not "the Δs differ" but "**the break-and-fix patterns
> differ.**"

## 3.5 The layers depth pruning actually deleted

The depth axis sets a **candidate range** via `--depth-min-layer` /
`--llm-prune-min-layer` and picks the top-k by BI inside it. But this option
is **read differently per harness** — a **fraction** in OpenVLA and UniVLA
(0.5 → from half the layer count) and a **count** in SpatialVLA (2 → from
layer 2). The shared name fooled us once too, and produced the false
counterexample in §6. So instead of condition names, we record **the layers
actually deleted.**

| Campaign | Condition | Candidate range | Layers actually deleted | Region |
|---|---|---|---|---|
| OpenVLA / Bridge | prune 1 | `0.5` → 16..31 | L23 | back |
| OpenVLA / Bridge | prune 4 | `0.5` | L17, 20, 23, 27 (stack: …26) | back |
| OpenVLA / Bridge | prune 8 | `0.5` | L17,19,20,23,25,27,29,31 (small per-task differences) | back |
| OpenVLA / Fractal | prune 1 | `0.5` → 16..31 | L23 | back |
| OpenVLA / Fractal | prune 2 | `0.5` | L23, 25 (move_near) / L23, 26 (pick) | back |
| OpenVLA / Fractal | prune 4 | `0.5` | L17, 23, 25, 27 (move_near) / L17, 20, 23, 26 (two picks) / L17, 21, 23, 26 (pick standing) | back |
| **OpenVLA / Fractal** | **prune 4 early** | **`0.08` → 2..31** | **L2, 4, 6, 23** (move_near) / **L2, 4, 23, 26** (pick) | **2–3 front + 1–2 back** |
| SpatialVLA / Bridge | prune 1 | `2` → 2..25 | L10 (carrot·eggplant) / L17 (spoon) / L9 (stack) | middle |
| SpatialVLA / Bridge | prune 2 | `2` | L10,19 / L8,10 / L12,17 / L9,10 (by task) | middle |
| SpatialVLA / Fractal | prune 1 | `2` | L10 (three) / L9 (pick vertical) | middle |
| SpatialVLA / Fractal | prune 4 | `2` | L8, 9, 10, 20 (move_near) / L8, 9, 10, 19 (three picks) | middle |
| UniVLA / Bridge | prune 1 | `0.5` → 16..31 | L26 (three) / L30 (spoon) | back |
| UniVLA / Bridge | prune 4 | `0.5` | L21,24,26,30 / L21,23,26,30 / L21,25,27,30 / L20,22,26,30 | back |
| UniVLA / Bridge | prune 8 | `0.5` | 8 of L16–L30 (small per-task differences) | back |
| **UniVLA / Bridge** | **prune 4 mid** | **`0.08` → 2..31** | **L2, 4, 26, 30** (spoon: …25,30) | **2 front + 2 back** |

All three harnesses record the deleted layers in the result files — OpenVLA
in `depth.bypassed_layers`, SpatialVLA in `depth_prune.pruned`, UniVLA in
`llm_pruning.active_layers`. The table above was read entirely from those
fields.

**Even within one condition, the selection differs slightly per task**,
because the ranking is computed from each run's actual observation (the
difference from ShortGPT, §2.4). The table lists a representative value and
notes divergences where they occur.

**Three kinds are mixed here.** ① OpenVLA and UniVLA at defaults have only
the back half as candidates, so even if BI preferred a front layer it could
not pick one. ② SpatialVLA's candidates run 2..25, yet what BI actually
picked is L8–L12 and L17–L20 — **the middle band, 31–77% of depth in a
26-layer stack.** The front was open down to L2 and was never chosen.
③ The two open-range runs are **not pure front but front-back mixtures.**

**So "prune 4" does not specify an experiment** — someone reading only that
phrase cannot reproduce it. Same four layers: back-only on UniVLA gives
Δ −2.1, include the front and it gives Δ −79.2. Prior work reports only
counts (EfficientVLA: `L=28, L=22`; MoLe-VLA: "50% layers"). **In our
measurements, comparisons specified only by count split by double digits.**

### 3.5.1 Three implementation differences — we ran into all three

The `min-layer` meaning (fraction vs. count) is above. Widening the window
sweep (§4.4 c) to five columns surfaced **two more.** All three tell the
same story: "the same intervention" does not survive crossing
implementations.

| | OpenVLA · UniVLA | SpatialVLA |
|---|---|---|
| ① the `min-layer` argument | **fraction** (0.5 → back half) | **count** (2 → from L2) |
| ② the last layer | **not** protected | **always protected** |
| ③ action token count | fixed | **decided by EOS**, cap 256 |

**② silently shifts the capacity.** SpatialVLA's selection code is
`protected = set(range(min_layer)) | {self.n - 1}`. With 26 layers and
`min_layer=22`, the candidates are just `{22,23,24}` — L25, being the last
layer, is protected — so even with `--depth-prune 4` it **deletes only
three.** No error is raised. OpenVLA's `window875`, by contrast, deleted
`[28,29,30,31]`, and L31 *is* the last layer. **The same phrase "the last 4
layers" means different things in the two implementations.**

### 3.5.2 ★ The failure mode ③ produces — an intervention meant to save runs 4× slower

Observed when running SpatialVLA/Fractal with `min_layer=22` (in practice,
the three layers L22–24).

```
baseline                 937 ms/infer      (135-episode mean; 936.8 in the records)
depth prune 1 / 2 / 4    865 / 847 / 788   (faster as layers are removed)
L22–24 removed           5466 / 3702 / 3390   <- ep00/01/02, observed on the console
```

The suspect is `modeling_spatialvla.py:491`:

```python
generation_outputs = self.generate(**model_inputs, max_new_tokens=256, do_sample=False)
```

Normally the model writes a few action tokens and stops at EOS. **Delete the
layers just before the output and it can fail to emit EOS, running
generation to the 256 cap.** The size fits the 4× delay — our
`chunk_exec.py` profile measured **about 12 decode tokens** per normal step
(1 action = 3 tokens × 4 chunks), so 12 growing to 256 at roughly 11 ms per
token adds **about 2.8 seconds** (244 tokens × 11.3 ms = 2,765 ms; add
937 ms and you get ep01's 3,702 ms).

> ⚠️ **But this is a size check, not a verification.** Two unknowns (prefill
> P, per-token d) and two measurement points mean **any values solve it
> exactly** — that it fits is not evidence for the hypothesis. And indeed
> the answer moves with which episode you fit: ep01 gives d ≈ 11 ms, ep02
> 10 ms, **ep00 19 ms.**
>
> Also, **the three episodes spreading 5466 / 3702 / 3390 — a 1.61× range —
> is something this explanation cannot cover.** If all three hit the 256
> cap, their times should match. That they spread means some episode did not
> hit the cap, or another factor moved with it — and **we did not log
> generated-token counts, so we cannot tell which.**
>
> What is established is only the observation: **"inference slowed 3.6–5.8×
> in this condition."** That missing EOS is the cause is **a likely
> explanation that fits the code structure and the sizes** — not a measured
> mechanism. Confirming it needs a re-run that logs tokens per step (open,
> §7).

**So this condition, far from saving compute, uses 4× more.** §5.4 says
"the amount saved does not predict the outcome" — here **even the sign of
the saving flips.** And OpenVLA's matching contrast (`window875`) was fine
at −10.6% — its action token count is fixed, so it does not depend on EOS.

> **Status of this observation.** Seen on three episodes; filling all 135
> would take about 11 hours (80 steps × 3.7 s × 135). The policy failed
> three episodes in a row without even terminating, so we judged there was
> nothing more to learn and stopped. **Recorded as "a failure mode observed
> during execution," not as a measured grid cell.** No success rate is
> reported.
>
> **And the three ms values in this subsection have no result files.** The
> run was stopped, nothing remains under `results/`, and the values were
> copied from the console. **One of the three values from outside the
> grid** (the other two: §3.8's right column and §2.2 (c)'s chunk numbers) —
> and of those three, **the only ones that cannot be checked against any
> original either.** The baseline and prune values above, in contrast, were
> all recomputed from records.

## 3.6 How uniform does it have to be

This campaign contains two kinds of runs with different natures, and **their
uniformity requirements differ.**

| | The grid | Control & diagnostic runs |
|---|---|---|
| Examples | original policy, repeat {2,4}, foveation {log-polar, blur}, depth prune {1,2,4} | `baseline_rerun`, `depth_prune4_early`, `depth_prune4_mid`, `move_near_v1`, determinism re-checks |
| What it claims | **comparison between cells** | **one question the grid raised** |
| Must it be uniform? | **yes** — without uniformity the comparison does not hold | **no** — run it where the question lives |
| In the grid? | yes | **no** (`EXCLUDE`) |

That is why we re-measured the legacy runs, re-took the baseline on a
matched card, and dropped chunk execution — whose meaning changes per
backbone — from the axes. **Grid uniformity is not negotiable.** Conversely,
running a diagnostic in all five cells is waste — you run it where the
counterexample lives.

> ⚠️ **One thing currently sits on the boundary.** The region control (the
> two bold rows of §3.5) began as a diagnostic, but UniVLA's −79.2 is so
> large that **it is becoming a claim in its own right** ("not how many
> layers — which layers"). Used as a claim, it falls under grid uniformity,
> not diagnostic freedom.
>
> **As of 2026-08-10, the within-backbone region comparison exists on all
> three backbones, and the answer was "both."**

**① Region matters within a backbone.** Fix the capacity and move only the
region, and results split.

| Backbone | Same capacity | Back | Including front |
|---|---|---|---|
| UniVLA / Bridge | 4 layers | all ≥ L20 (e.g. [21,24,26,30]) → **−2.1** | all include L2·L4 (e.g. [2,4,26,30]) → **−79.2** |
| OpenVLA / Fractal | 4 layers | `move_near`'s [17,23,25,27] → **70.0** | `move_near`'s [2,4,6,23] → **50.0** (p = 0.0169) |

> Bracketed sets are one representative task each. The sets differ slightly
> per task (table before §4.4), so the conclusion rests on the **region**,
> not individual layers.

**② But control the region and the backbone still remains.** SpatialVLA
drops sharply even when only the back half is deleted.

| Backbone (Bridge) | Deleting from the back half only |
|---|---|
| OpenVLA (32 layers) | 8 layers removed: **±0.0** |
| UniVLA (32 layers) | 8 layers removed: **−4.2** (p = 0.42) |
| **SpatialVLA (26 layers)** | **4 layers removed: −30.2** (29 broke / 0 fixed) |

So **SpatialVLA has no spare layers.** The other two get the last 8 layers
nearly for free, while this backbone is already at −30.2 with 4 back-half
layers. §2.4 (e)'s "the sign is opposite on SpatialVLA" is not a region
artifact but **a real backbone property**, surviving after regions are
matched.

### ③ At dose 1, region also does not carry the effect (2026-08-11)

The contrast in ② was at dose 4, but the number §0 cites is **prune 1's
−10.4.** The layers behind that −10.4 differ per task — **two L10s, one L17,
one L9** — the **middle**, 35–65% of depth in 26 layers. Write "even one
layer costs −10.4" and a reader hears the price of one layer, when it may be
"one *middle* layer specifically." So we ran the same one layer chosen from
the back half (L13–25).

| Condition | Deleted layer (by task) | Depth | Success | Δ | broke/fixed | p |
|---|---|---|---:|---:|---:|---:|
| baseline | — | — | 30.2% | — | — | — |
| `depth_prune1` | 10 / 10 / **17** / 9 | 35–65% | 19.8% | −10.4 | 19/9 | 0.0872 |
| `depth_prune1_back` | 19 / 20 / **17** / 19 | 65–77% | 17.7% | −12.5 | 17/5 | 0.0169 |

Compared head to head (same 96 episodes, paired):

> `prune1` vs `prune1_back` — 19.8% → 17.7%, **Δ −2.1, 9 broke / 7 fixed,
> p = 0.8036.** Indistinguishable from chance.

> ⚠️ **This contrast holds on only three of the four tasks.** On
> `spoon_on_towel`, BI picked **the same L17** even when restricted to the
> back half — the originally chosen layer was already there. So that task's
> 24 episodes are **the same setting in both conditions, and the results are
> bit-identical** (0/24 both sides). McNemar uses only discordant pairs, so
> those 24 contribute nothing and the p effectively comes from the other 72
> — but **"we moved the layer back" is not true of all four tasks**, and
> that must be said.

**So §0's −10.4 is not a position artifact.** There are no grounds for a
qualifier.

This contrast carries one control we got by luck. The two layers' **Block
Influence is practically equal** — 0.939 vs. 0.938 on `carrot`, and
0.925–0.939 vs. 0.920–0.938 across all four tasks. So with "they differed in
redundancy" ruled out at the source, **only the depth position** differed —
38% vs. 73%. We did not design this; it fell out when we raised
`--depth-prune-min-layer` to 13 — so it is not a control we built but **a
property we checked and now report.**

**Same size is not same damage.** Of the episodes the baseline solved,
`prune1` breaks 19 and `prune1_back` breaks 17 — but **only 14 overlap.**
Each layer breaks a slightly different set of episodes while the totals
happen to match. Do not read this as "the layers are interchangeable."

### ④ And the low-dose region is neither monotone nor significant in the first place

A side result of measuring ③ — and it is actually this side that required
fixing §0.

| Condition | Deleted layers (representative task) | Success | Δ | broke/fixed | p |
|---|---|---:|---:|---:|---:|
| baseline | — | 30.2% | — | — | — |
| `depth_prune1` | [10] *(10 / 10 / 17 / 9)* | 19.8% | −10.4 | 19/9 | 0.0872 |
| `depth_prune1_back` | [19] *(19 / 20 / 17 / 19)* | 17.7% | −12.5 | 17/5 | 0.0169 |
| `depth_prune2` | [10,19] *(varies by task)* | 20.8% | −9.4 | 15/6 | 0.0784 |
| `depth_prune4` | [9,10,17,19] *(varies by task)* | 2.1% | −28.1 | 27/0 | <0.0001 |
| `depth_prune4_back` | [17,18,19,20] *(other three: [13,17,19,20])* | 0.0% | −30.2 | 29/0 | <0.0001 |

> Brackets show `carrot_on_plate`; parentheses show all four tasks — the
> ranking is recomputed per task from that task's first observation (§3.0).
> This section's argument rests on the **band** (middle vs. back half), not
> on individual layer numbers.

**Deleting two beats deleting one** (`prune1` → `prune2` +1.0, p = 1.0000;
`prune1_back` → `prune2` +3.1, p = 0.6636) — which of course also means the
three are mutually indistinguishable. The honest reading: **the 1–2-layer
range is a flat shelf near −10**, and **the steep drop sits between 2 and 4
layers** — `prune2` → `prune4` is **−18.8, 20 broke / 2 fixed, p = 0.0001.**
At 4 layers, region becomes irrelevant again (`prune4` vs. `prune4_back`:
−2.1, p = 0.50, both at the floor).

**And the low-dose Δs are not significant after correction.** Bonferroni
over the grid's 38 tests gives α ≈ 0.0013, and both `prune1`'s p = 0.0872
and `prune1_back`'s p = 0.0169 sit above it. So **"even one layer costs
−10.4" cannot be asserted.** Precisely: **a loss in the −10 to −12 range is
visible but, after correction, indistinguishable from chance** — and the
only firm statement for this backbone is **the large drop at the 4-layer
point.** §0 was rewritten accordingly.

> This item came close to §7.1 ② (us getting ahead of the data). The run
> was made to measure ③, but once run, the bigger problem turned out to be
> **the significance of the number §0 itself was citing.** Had we not asked
> the region question, we would still be writing −10.4 as a plain assertion.

## 3.7 Reproducing

```bash
python experiments/build_grid_report.py           # every table
python experiments/build_grid_report.py --json    # machine-readable

# one individual comparison -- 2x2 contingency table and per-task detail
python adaptive_sparse_vla/paired_test.py \
  results/spatialvla_fractal_0806/baseline \
  results/spatialvla_fractal_0806/action_repeat4

# whether a re-run really reproduced, episode by episode
python experiments/compare_runs.py <baseline-condition-dir> <new-condition-dir>
```

> ⚠️ **Never use `git add results/`.** The execution environment re-clones
> the repository each time it starts, and if the previous campaigns'
> records are not all present, **missing files get staged as deletions.**
> This actually happened on 2026-08-12 — the commit uploading UniVLA
> prune4_last (`f2b25d5`) deleted three of the four files under
> `univla_bridge_0805/foveate_logpolar` and moved one to a new path. The
> grid cell turned into `--`, and **no error was raised.** Recovered with
> `git show f2b25d5^ -- <path>`.
>
> When uploading, name **only the directory that run produced.**
> ```bash
> git add results/<this-run's-directory>/     # not results/ wholesale
> git status --short | grep '^ D' && echo "a deletion is staged -- STOP"
> ```
> The grid is regenerated wholesale, so a vanished cell just quietly shows
> `--`. **To notice something is gone, you have to count.**

## 3.8 How far our baselines sit from the authors' reported values

**Without this section there is no answer to "isn't your setup just
broken?"** Our whole argument is Δs, and Δs sit on top of baselines — so
first we write down how far our baselines are from everyone else's.

> **Both columns of this section have been checked against originals and
> records.** The left (our values) was recomputed from `results/`; the right
> (authors' values) was checked against **the SpatialVLA, UniVLA, and
> OpenVLA PDFs** — all three match to the decimal. We verified SpatialVLA
> Table II's zero-shot success `20.8 / 20.8 / 25.0 / 70.8 → 34.4`, Table I's
> `81.0 / 69.6`, UniVLA Table 3's `83.3 / 66.7 / 33.3 / 95.8 → 69.8`, and
> that the two papers' OpenVLA rows agree in all eight cells.

### (a) To state first — the OpenVLA paper contains no SimplerEnv numbers

Neither the OpenVLA paper nor OpenVLA-OFT contains **any SimplerEnv
evaluation.** The original's evaluations are real WidowX / Google robots,
and **the only simulation is LIBERO, in Appendix E** (an earlier draft said
"both evaluate only on real robots," missing LIBERO). So every **SimplerEnv**
number for OpenVLA was **run by a third party.**

> ⚠️ **And that third party is one source — confirmed from both papers.**
> SpatialVLA Table II's OpenVLA row and UniVLA Table 3's OpenVLA row agree
> **in all eight cells across the four tasks**:
> `4.1 / 0 · 33.3 / 0 · 12.5 / 0 · 8.3 / 4.1`, mean **1.0%** (each pair is
> grasp rate / success rate). Not two independent measurements — **one
> source cited twice.** The same structure we point out about EfficientVLA
> and FastV in §2.5; we nearly counted it as "two papers report it" too.

### (b) The comparison table

Our baselines all use **the authors' released checkpoints with the authors'
settings**, run without any intervention.

**WidowX / Bridge (24 episodes per task, 96 total)**

| | spoon | carrot | stack | eggplant | mean |
|---|---:|---:|---:|---:|---:|
| **OpenVLA — ours** | 8.3% | 16.7% | 12.5% | 25.0% | **15.6%** |
| OpenVLA — reported (SpatialVLA Table II = UniVLA Table 3) | 0% | 0% | 0% | 4.1% | **1.0%** |
| **SpatialVLA — ours** | 8.3% | 25.0% | 29.2% | 58.3% | **30.2%** |
| SpatialVLA — its paper's Table II (zero-shot) | 20.8% | 20.8% | 25.0% | 70.8% | **34.4%** |
| **UniVLA — ours** | 87.5% | 62.5% | 75.0% | 100.0% | **81.2%** |
| UniVLA — its paper's Table 3 | 83.3% | 66.7% | 33.3% | 95.8% | **69.8%** |

**Google Robot / Fractal (visual matching; coke can sums the three poses)**

| | pick coke can | move near |
|---|---:|---:|
| **OpenVLA — ours** | 20.0% (15/75) | 61.7% (37/60) |
| OpenVLA — reported (SpatialVLA Table I) | 16.3% | 46.2% |
| **SpatialVLA — ours** | 85.3% (64/75) | 83.3% (50/60) |
| SpatialVLA — its paper's Table I (zero-shot) | 81.0% | 69.6% |

> We do not compare the papers' "averages" — their task sets differ from
> ours (SpatialVLA Table I includes Open/Close Drawer, which our grid does
> not have). The table above puts **matching tasks only** side by side.

### (c) Three readings

**① Four of the five cells are above the reported values; one is 4.2 points
below.**

| Cell | Ours | Authors | Difference |
|---|---:|---:|---:|
| OpenVLA / Bridge | 15.6% | 1.0% | **+14.6** |
| OpenVLA / Fractal (pick / move) | 20.0 / 61.7% | 16.3 / 46.2% | **+3.7 / +15.5** |
| **SpatialVLA / Bridge** | **30.2%** | **34.4%** | **−4.2** |
| SpatialVLA / Fractal (pick / move) | 85.3 / 83.3% | 81.0 / 69.6% | **+4.3 / +13.7** |
| UniVLA / Bridge | 81.2% | 69.8% | **+11.4** |

**The objection "your setup is broken and the policies underperform" does
not fit this table** — in that case our values should be *consistently* low,
but four are high and only one is low, and that one by 4.2 points. Still,
**we cannot write "nowhere lower."** An earlier draft did, while ② right
below it recorded the counterexample itself (§7.1 ②).

**② SpatialVLA is close to its authors' values on both benchmarks.** Bridge
30.2% vs. 34.4% (−4.2); Fractal coke can 85.3% vs. 81.0% (+4.3), move near
83.3% vs. 69.6% (+13.7). Per task, spoon (8.3 vs. 20.8) and eggplant (58.3
vs. 70.8) deviate downward, the rest upward. **A matching total does not
mean matching tasks** — the very point we stress in §3.4 shows up again in
this comparison with someone else's table.

**③ In two places we are far above. The reasons differ.**

- **OpenVLA / Bridge: 15.6% vs. 1.0%.** Fifteen-fold. But the 1.0% has one
  independent source, as (a) showed, and **OpenVLA's training mixture
  contains 13.3% Bridge** (its paper's Appendix A, Table 3; Fractal is
  12.7%). A model that saw that much Bridge scoring 1% on the Bridge sim
  casts suspicion less on the policy than on **the evaluation setup.** We
  did not reproduce that setup, so we do not judge which is right. **What we
  can write stops at: "the widely cited 1.0% differs from our 15.6%, and
  that 1.0% has a single independent source."**
- **UniVLA / Bridge: 81.2% vs. 69.8%.** Most of the gap is one task,
  stack_cube (75.0% vs. 33.3%; the other three are within ±4.2). Our
  checkpoint is `UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K`, **a Bridge-specific
  follow-up training**, and which checkpoint the paper's table used we could
  not determine. **We suspect a checkpoint difference, but have not
  confirmed it.**

### (d) What this table does for the argument

Our claim is **not "our baselines are the right ones."** Our claim is that
Δs do not hold across conditions, and that is measured **by pairing within
each column** (§3.3). Even if our baseline heights differ from others', the
pairing inside a column is untouched.

Baseline height does, however, set **the range Δ can move in.**
OpenVLA/Bridge at 15.6% can only fall 15.6; SpatialVLA/Fractal at 84.4% can
only rise 15.6. **This constraint is written in §4.1 and revisited every
time a large + is compared with a large −.**

---

# §4. Results by intervention

> ⚠️ **In this section's tables, bold means `p < 0.05` — the uncorrected
> bar.** The grid table in `Overview.md` bolds only what passes
> **multiple-comparison correction (α ≈ 0.0013)**, so **the same value can
> be bold in one document and not the other.** For example,
> SpatialVLA/Bridge's action repeat 2 (+12.5, p = 0.043) is bold here and
> not in the Overview. **Twelve cells uncorrected, eight corrected — and
> the argument rests on the corrected eight** (§3.3).
>
> **So there is one place where boldness must not be read as a result.** In
> the same cell, repeat 2 and repeat 4 are **+12.5 and −12.5 — equal
> magnitude** — yet their p-values straddle 0.05 at 0.0428 and 0.0501. A
> difference of 0.0073 makes one bold and the other not. Neither passes the
> correction (α ≈ 0.0013), so **in the argument the two cells play the same
> role** (neither supports a claim).

Every number is generated straight from the records by
`python experiments/build_grid_report.py`. No table is copied by hand — in
this campaign, nearly every wrong sentence happened at the point where a
human transcribed a table (§7.1 ②).

## 4.1 The original policies — and the floor/ceiling problem

| | OpenVLA Bridge | OpenVLA Fractal | SpatialVLA Bridge | SpatialVLA Fractal | UniVLA Bridge |
|---|---:|---:|---:|---:|---:|
| success | **15.6%** | **38.5%** | **30.2%** | **84.4%** | **81.2%** |
| n | 96 | 135 | 96 | 135 | 96 |
| ms / env-step | 518 | 515 | 902 | 937 | 604 |

**That these five baselines sit at different heights is the first constraint
on reading the grid.** OpenVLA/Bridge, at 15.6%, is near the floor — 84
points of headroom. SpatialVLA/Fractal (84.4%) and UniVLA/Bridge (81.2%) are
near the ceiling — 16 and 19 points of headroom respectively.

So **"a big + appeared in this cell" must not be read as one method beating
another.** That the campaign's largest gains all came from OpenVLA/Bridge
(+18.8, +19.8, keep 100%'s +30.2) is partly because that was the only cell
with room to rise. Conversely, SpatialVLA/Fractal's largest gain stopping at
**+8.1** (depth prune 1) may not mean the intervention is weak — from 84.4%
there were only 15.6 points to gain.

This constraint **does not touch the sign-flip claims.** A ceiling makes a +
smaller; it does not make it a −. That our thesis is "does the same method
keep its sign," not "which method is best," is justified once more here.

## 4.2 The time axis — action repeat

| | OpenVLA Bridge | OpenVLA Fractal | SpatialVLA Bridge | SpatialVLA Fractal | UniVLA Bridge |
|---|---:|---:|---:|---:|---:|
| repeat 2 | −8.3 | +5.2 | **+12.5** (p=0.043) | ±0.0 | **−69.8** (p<0.0001) |
| repeat 4 | **−11.5** (p=0.0010) | −1.5 | −12.5 (p=0.0501) | **−40.0** (p<0.0001) | **−81.2** (p<0.0001) |
| saved (repeat 2 / 4) | −50% / −75% | −50% / −75% | −51% / −75% | −52% / −76% | −52% / −77% |

**Three readings.**

**① k = 4 loses in all five cells.** It is the only condition whose sign
holds — and the sign is negative. UniVLA at −81.2 **reaches 0.0% success** —
all four tasks at 0/24.

**② k = 2 splits.** SpatialVLA/Bridge +12.5, UniVLA/Bridge −69.8. Same
benchmark, same intervention, same k — an 82-point spread. The difference is
**p < 0.0001** by Fisher's exact test (§5.2).

**③ The saving is pinned at −50% / −75% regardless of backbone.** So on
this axis, what depends on the backbone is not "how much you save" but only
**"what you lose in exchange."**

> **Why UniVLA falls this far, we have no measured answer.** We record the
> observation only — UniVLA emits actions as discrete tokens through the
> Emu3 VQ tokenizer, and executing one prediction for two steps costs this
> backbone unusually much. Whether the tokenizer or the training-time action
> distribution is responsible, our data cannot distinguish.

## 4.3 The vision axis — foveation

### (a) The grid cells (keep 20%)

| | OpenVLA Bridge | OpenVLA Fractal | SpatialVLA Bridge | SpatialVLA Fractal | UniVLA Bridge |
|---|---:|---:|---:|---:|---:|
| log-polar | **+18.8** (p=0.0051) | **−19.3** (p=0.0004) | −8.3 | +0.7 | +5.2 |
| blur | **+17.7** (p=0.0060) | −8.9 | ±0.0 | −1.5 | −8.3 |
| saved | ≈0% | −1.7% / −0.8% | +2.7% / −0.2% | ±0.0% / −3.1% | +0.8% / −2.5% |

**The saving is effectively zero.** Neither variant changes image size or
token count — they only change the preprocessing — so there is no structural
reason for compute to shrink. The measured values across the five cells sit
between −3.1% and +2.7%, which is measurement noise. **Our measurements give
no grounds to classify this as an efficiency intervention.**

**Yet the campaign's largest gain comes from here** (OpenVLA/Bridge +18.8).
And the same intervention on the same model's other benchmark is −19.3.
That pair is the first sentence of §0.

### (b) ★ Dose-response — the gain does not come from compression

What does `keep-percent` do? The code:

```python
logpolar     = cv2.warpPolar(frame, ...)                    # always warps, regardless of keep
sample_ys, sample_xs = _uniform_sample_grid(h, w, keep)     # sparse sampling AFTER the warp
interpolated = cv2.resize(sampled, (w, h))
restored     = cv2.warpPolar(interpolated, ..., INVERSE)     # warp back
```

**`keep` only sets the sample density after the warp. The log-polar round
trip itself always happens.** So `keep = 100%` is not the original image but
**"discard no samples, still take the warp round trip."**

![the four log-polar stages](figures/logpolar_stages.png)

*The four stages applied to a real Bridge observation*
(`experiments/make_logpolar_figure.py`). In the top row, column 3 is
**byte-identical** to column 2 — at keep=100% the resampling is a no-op, so
all that remains is the warp and its inverse. Yet the difference map in
column 5 is not empty. In the bottom row (keep 20%), column 3 shrinks to
286×215 and the difference map spreads past object outlines into the whole
background.

**The round trip loses information because the log-polar axis is the log of
the radius.** Columns are assigned by the radius *ratio*, not pixel count —
so one pixel near the center spreads over dozens of columns while several
edge pixels get averaged into one. **Edge detail is gone before the inverse
warp even runs.**

How we measured it: pass an image **whose pixel value equals its radius**
through the warp, and read the column → radius mapping straight off the
result. The warp arguments are not typed by hand — we intercept what the
shared module actually passes (`dsize=(640,480)`, `center=(320,240)`,
`maxRadius=400.0`, `flags=265`). 640×480 is the WidowX camera's actual
render resolution (`ManiSkill2_real2sim/agents/configs/widowx/defaults.py`);
the processor's 224 downscale happens **after** foveation.

| Radius band (px) | Columns assigned | Source width (px) | **Source px per column** |
|---:|---:|---:|---:|
| 1–2 | 49 | 1 | **0.02** |
| 2–4 | 56 | 2 | 0.04 |
| 4–8 | 63 | 4 | 0.06 |
| 8–16 | 68 | 8 | 0.12 |
| 16–32 | 71 | 16 | 0.23 |
| 32–64 | 72 | 32 | 0.44 |
| 64–128 | 74 | 64 | 0.86 |
| 128–256 | 73 | 128 | 1.75 |
| 256–400 | 47 | 144 | **3.06** |

**At the center, one pixel is spread across 49 columns (0.02); at the edge,
three pixels are averaged into one column (3.06) — a 150-fold difference.**
This is the mechanism behind "center preserved, periphery lost," and `keep`
plays no part in it. Check: the map is monotone and the last column reaches
395.3 px (maxRadius 400).

> **This magnify/compress exists only in the intermediate representation;
> the output's geometry is restored.** The inverse warp undoes the forward
> warp, so **objects do not look stretched** — place a marker and run the
> round trip, and dead center returns to the **0.00 px** spot, and edges
> come back **within 1 px** (0.5–1.0 px by the brightness centroid of a 3×3
> marker; it moves with marker position, size, and estimator, so we claim
> only "sub-pixel," not a single digit. A single-pixel marker can vanish
> entirely at the far edge, averaged into its neighbors — which is exactly
> the information loss of the next paragraph).
>
> **What does not return is information, and the loss is one-sided.** The
> center spreads one value over 49 columns and gathers it back — nothing
> lost. The edge averages three values into one and spreads it back — the
> differences among the three are gone. Run five values through it:
> `[10,200,30,220,40]` comes back unchanged on the 49× path, and comes back
> as `[153,155,164,173,175]` on the 3:1 path. **Stretching is lossless;
> squeezing is irreversible** — which is why the loss piles up only in the
> periphery.

![center vs. edge, magnified](figures/logpolar_zoom.png)

*keep=100% input and output, magnified 3×* (`make_logpolar_figure.py
zoom`). The center (carrot and plate, r=20px) is practically
indistinguishable from the input, with differences only thinly tracing the
plate's rim (mean 0.68). The edge (the wall, r=319px) **loses its texture
entirely** (mean 6.66). Geometry is intact on both.

> **We note a place we once got wrong.** An earlier draft hand-computed
> this table with `rho = exp(x/M)` and wrote "74 columns per band,
> uniformly." OpenCV actually uses `rho = exp(x/M) − 1`, so **at small radii
> the counts fall short of 74 — 49, 56, 63, …** The conclusion (center
> oversampled, periphery crushed) stands, but the uniformity claim does
> not. The table above is now entirely measured.

**We measured it directly.** One real Bridge observation (640×480 — the
exact resolution the harness foveates at), mean absolute pixel difference
(0–255) before and after the round trip, split into center (the middle 1/4
of the area) and periphery:

| keep | Samples after warp | Center | Periphery | Periphery/center |
|---:|---:|---:|---:|---:|
| **100%** | **307,200 (= 640×480, same as original)** | **1.2** | **3.3** | **2.7×** |
| 40% | 123,120 | 2.6 | 5.1 | 2.0× |
| 20% | 61,490 | 2.9 | 6.1 | 2.1× |
| 10% | 30,704 | 3.5 | 7.6 | 2.2× |

Two things emerge. First, **even at keep=100%, which discards nothing, the
image changes — and the periphery degrades 2.7× more than the center.** So
keep=100% is not "foveation off" but **already foveation** — consistent
with the gain peaking there. Second, **lowering keep degrades the center
too** (1.2 → 3.5). It does not just discard more periphery — it cuts into
the very center the policy uses, and the compute saved in exchange is zero
(last column of the table above). This is consistent with the curve being
monotone.

> **Limits of this measurement.** One observation; the absolute sizes
> depend on how much fine texture the image has — on checkerboard/noise
> test images the same measure runs up to **19–89.** So **we quote
> direction and ratio only, never absolute values.**
>
> The direction held in **all 28 cells** — **seven images** (three
> synthetic + four Bridge observations) × four keeps — the periphery
> degraded more than the center in every one. At keep=100% the ratio is
> **1.5–2.8×**; across all keeps, 1.1–2.8×. To reproduce:
> `python experiments/measure_foveation_roundtrip.py experiments/figures/obs_*_raw.png`.
>
> **Linking this to success rates is interpretation, not measurement.**

**Quoting only means makes the change look small — so we give the
distribution too.** The 1.2 and 3.3 above are means, dominated by flat
regions like the background. Re-measure after downscaling to the 224×224
the model actually receives (what the processor does):

| keep | Pixels changed | Changed by **more than 2** | Mean | Max |
|---:|---:|---:|---:|---:|
| **100%** | **71.0%** | 19.3% | 2.06 | **187** |
| 40% | 88.4% | 43.6% | 4.09 | 216 |
| 20% | 90.3% | 49.4% | 5.16 | 218 |
| 10% | 92.6% | 57.2% | 6.73 | 232 |

> **One rule for this table.** A pixel's change is defined as **the maximum
> over the three channels** (it counts if even one channel moved that
> much). The second column is **strictly more than 2**, not "2 or more" —
> an earlier draft wrote "2 or more" while the actual count used `> 2`, in
> that column only. And the draft's "mean" column alone was computed as a
> **channel average** (1.75 / 3.44 / 4.32 / 5.61), so one table carried two
> rules. **All values above are unified on the channel-maximum rule.**

**Even at keep=100%, 71% of pixels change and the maximum change is
187/255.** Take the center alone and the median error is 1, but **2.5% of
pixels move by 10 or more, up to 91.** So the loss is **resampling error
concentrated on edges**, not a global blur. Lower keep and the share of
center pixels moving by 10 or more grows **2.5% → 8.3%.**

> **Those two values are measured at the original 640×480, not at 224** —
> downscale the center to 224 and one more resampling mixes in, giving
> 0.6% → 5.5%. Different basis from the 224 table above; do not read them
> side by side. Also, the draft's "2.2% → 7.2%" was counted with `> 10`;
> the values above are re-counted as the sentence says, **10 or more**
> (`>= 10`).

> **"Shouldn't keep=100% be the original?" is a fair question, and the
> answer is "by design it should be — but this variant is not."**
> `foveate_image_blur` returns `frame.copy()` at `keep_ratio >= 1.0` — truly
> a no-op. Only log-polar takes the warp round trip regardless of `keep`,
> leaving interpolation loss. So keep=100% is **not a foveation anyone
> designed on purpose — it is the condition where irreversible round-trip
> loss happens to pile up in the periphery.** And this section's claim is
> not "foveation is good" but **"this gain cannot be explained by
> compression."**

This structure lets us separate two hypotheses. Measured on OpenVLA/Bridge:

| keep | Success | Δ | broke/fixed | p | Compute saved |
|---:|---:|---:|---:|---:|---:|
| *(original)* | 15.6% | — | — | — | — |
| 10% | 19.8% | +4.2 | 12/16 | 0.5716 | +0.3% |
| 20% | 34.4% | **+18.8** | 10/28 | 0.0051 | — |
| 40% | 35.4% | **+19.8** | 7/26 | **0.0013** | −0.6% |
| **100%** | **45.8%** | **+30.2** | **3/32** | **4.2 × 10⁻⁷** | −0.5% |

All four points are 96 episodes.

> ⚠️ **Only the keep 20% row comes from a different tree.** 10, 40, and
> 100% ran in `results/openvla_bridge_foveate_sweep/`, but 20% is the grid
> cell, from the pre-campaign
> `RetinaBased/GoogleColab/results_reproduction_eager/` run. There is **no**
> OpenVLA/Bridge log-polar keep=20% under `results/` (which is also why its
> compute-saved cell is `—`: that tree does not record timing in the same
> format).
>
> **So we first checked the two trees are the same computation.** Matching
> the two trees' **baseline conditions** episode by episode: **96/96
> identical down to success and step count** (both 15/96 = 15.6%). Same
> policy, same seeds, same determinism — so placing this row beside the
> other three is legitimate. But **the provenance difference itself must be
> disclosed**, hence this note.
>
> **A stronger check — the conclusion stands without mixing trees.** We
> tested using only comparisons that never cross trees:
>
> | Confound-free comparison | Δ | broke/fixed | p |
> |---|---:|:--:|---:|
> | **within the old tree**: baseline → keep20 | **+18.8** | 10/28 | **0.0051** |
> | **within the new campaign**: keep10 → keep100 | **+26.0** | 12/37 | **0.00047** |
>
> **Both survive.** So the keep-axis effect is not an artifact of harness or
> environment. Only the table's +30.2 crosses the trees, and it is backed by
> the two above.

**The setting that discards nothing is the largest and the most significant
in the whole campaign.** So this gain does not come from **discarding the
periphery** — it comes from **the log-polar geometry itself.** Compression
is not the cause of the gain, and pushed to 10% it actually erases it
(+4.2, p = 0.57).

**The total curve is monotone** — 10 → 20 → 40 → 100, **the more you
discard, the worse** (+4.2 → +18.8 → +19.8 → +30.2).

> ⚠️ **But unfold it per task and it is not monotone — the split lives in
> our table too.**
>
> | Task | baseline | keep10 | keep20 | keep40 | keep100 |
> |---|---:|---:|---:|---:|---:|
> | `carrot_on_plate` | 4/24 | 3 | 4 | 6 | **8** |
> | `put_eggplant_in_basket` | 6/24 | 4 | 8 | 13 | **19** |
> | `spoon_on_towel` | 2/24 | 6 | **10** | 9 | 8 |
> | `stack_cube` | 3/24 | 6 | **11** | 6 | 9 |
>
> **keep 100% is the maximum on two of the four tasks**; the other two bend
> at keep 20%. The draft said "three of four" — counting again, it is two.
>
> **The total is monotone while half the tasks are not** — exactly the
> shape we point out in others' tables in §2.5, happening in our own. So
> what this section can defend is **"the total curve is monotone and peaks
> at keep 100%"** — not that every task does. **The conclusion that
> compression is not the source of the gain stands on the total alone** —
> the least-discarding setting is the total's maximum, at p = 4.2 × 10⁻⁷.

The biggest riser is `put_eggplant_in_basket`, 6/24 → **19/24** — more than
three times the original.

> **So the name "foveation" does not fit this condition.** The point of
> foveal vision is to save budget by giving up peripheral resolution — and
> the setting that wins here gives up nothing. What we actually measured is
> that **log-polar resampling is a better policy input than the original
> pixel grid** — a claim about representation, not efficiency. That the
> prior work cited in §2.3 does not separate the two is picked up again in
> §5.4.

> **This section originally began as "let's see whether 20% is the sweet
> spot."** And the author (us) judged the upper side (40 → 100) "pointless,
> since Δ → 0 is forced by design." Reading the code proved that wrong —
> the upper side was exactly where the answer lay. Logged in §7.1 ②.

### (c) The choice of variant changes the outcome too

**The two variants differ already in what `keep` means** (the ⚠️ in §3.0).
Blur sets a **central disc of area keep** via `r0 = sqrt(keep·H·W/π)` and
preserves it bit for bit. Log-polar sets only **the fraction of samples kept
in the warped image** via `sample_scale = sqrt(keep)` — it designates no
region at all. And **only log-polar moves pixels** (a geometric transform);
blur preserves geometry and merely blurs the periphery.

The starkest difference is keep = 100%. `foveate_image_blur` returns
`frame.copy()` at `keep_ratio >= 1.0` — **literally a no-op.** At the same
setting, log-polar still changes the image, as (b) showed. So "keep 100%"
is the original on one side and the campaign's largest-gain condition on
the other.

The two variants **run the same episodes, so they can be paired with each
other.** Instead of subtracting two Δs against baseline, we pair the two
conditions directly — that is §5.3's table, where which variant wins flips
per cell (UniVLA/Bridge: log-polar by +13.5; OpenVLA/Fractal: blur by
+10.4). But **no cell passes the multiple-comparison correction** — details
in §5.3.

## 4.4 The compute axis — depth pruning

### (a) The capacity curve

| Layers deleted | OpenVLA Bridge | OpenVLA Fractal | SpatialVLA Bridge | SpatialVLA Fractal | UniVLA Bridge |
|---:|---:|---:|---:|---:|---:|
| 1 | +2.1 | +0.7 | −10.4 | **+8.1** (p=0.013) | −3.1 |
| 2 | ±0.0 | ±0.0 | −9.4 | +3.0 | −4.2 |
| 3 | — | **+11.1** (p=0.017) | — | — | — |
| 4 | +1.0 | **+15.6** (p=0.0011) | **−28.1** (p<0.0001) | **−17.8** (p=0.0002) | −2.1 |
| 8 | ±0.0 | *(not run)* | *(not run)* | *(not run)* | −4.2 |
| saved (4 layers) | −11.2% | −11.9% | −12.0% | −15.9% | −11.2% |

> **⚠️ The "4 layers" cells of this table are not single values.** On
> OpenVLA/Fractal, the same four layers chosen by a different rule give
> anywhere from **+15.6 to −30.4** (§4.4 c). This table shows each column's
> **default setting**, and that the default decides the outcome is this
> section's point. Whether the other columns swing as widely, we did not
> measure.

> **`depth prune 8` is not a grid row.** It exists in two of five columns.
> The depth axis's grid is {1, 2, 4}, present in all five. The 8-layer
> point is read only as an **extra capacity point** in two columns (§3.6's
> uniformity rule). SpatialVLA is already at the floor (Bridge 2.1%) at 4
> layers, so there is nothing to measure at 8.

**Three readings.**

**① The sign is opposite by backbone.** At 4 layers, OpenVLA/Fractal is
**+15.6** (though its interpretation is left open in §4.4 c) and
SpatialVLA/Bridge is **−28.1.** Both pass the correction. Same
intervention, same capacity — one is the campaign's largest gain and the
other its largest drop. Fisher: **p < 0.0001** (§5.2).

**② SpatialVLA has no spare layers.** The other two backbones get the last
8 layers nearly free (OpenVLA ±0.0, UniVLA −4.2); this backbone is at
−28.1 already at 4. The property survives region control in §3.6 ②·③.

**③ The low-dose range is neither significant nor monotone.** On
SpatialVLA/Bridge, 1 layer is −10.4 (p=0.087) and 2 layers −9.4 (p=0.078)
— **two layers beat one.** The steep drop sits between 2 and 4 layers
(−18.8, p = 0.0001). §3.6 ④.

**④ And the values in this table swing widely with layer choice — by a
different amount per cell.** That is §4.4 (c).

### (b) `--depth-min-layer` means different things per implementation

This is not a result but **a caution about an implementation difference** —
unrecorded, the next person hits the same thing.

| Implementation | What `--depth-min-layer 2` means |
|---|---|
| OpenVLA · UniVLA | **a fraction** — 0.5 means "from the back half" |
| SpatialVLA | **a count** — 2 means "from L2" |

The same flag value deletes completely different layer sets. This is why
§3.5 records **the layers each run actually deleted**, and one early
counterexample in §6 came from this confusion.

**What this flag sets is the candidate range (the window).** Layer selection
is two steps — ① fix the window, ② pick the top-k by BI inside it. The
contrast in (c) moves only ①. The actual windows, read from the `eligible`
line of the run logs:

| Backbone | Layers | Default window | Shifted-back window | Other windows run |
|---|---:|---|---|---|
| OpenVLA | 32 | **L16–31** (0.5, 16 candidates) | **L28–31** (0.875, **4** candidates) | L8–31 (0.25), L2–31 |
| UniVLA | 32 | L16–31 (0.5) | L28–31 (0.875) | L2–31 (`prune4_mid`) |
| SpatialVLA | 26 | L2–24 (count 2, 23 candidates) | **L13–24** (count 13, 12 candidates) | — |

> **In the shifted-back window, OpenVLA and UniVLA have exactly 4
> candidates and delete 4.** In that condition **the BI ranking does
> nothing** and the last four layers are deleted by force. `window875`'s
> −30.4 is not "the worst set BI picked" but the value of "a condition with
> no choice at all."
>
> **Only SpatialVLA could not use the same setting.** Its last layer is
> always protected (§3.5.1 ②), so narrowing to 4 candidates actually
> deletes only 3 — and it runs into §3.5.2's generation-length blow-up
> besides. So we substituted **top of the back half**, and this
> non-uniformity is stated in (c).

### (c) ★ The swing layer choice makes at equal capacity and equal cost — and the swing itself depends on the cell

The grid in §4.4 (a) shows each column's **default setting.**
OpenVLA/Fractal's +15.6 was the campaign's largest gain, so we shook that
default — **capacity (4 layers) and rule (top BI, adjacency) fixed, only
the candidate window moved.** The result is this section.

### The eight conditions (baseline 52/135 = 38.5%)

| Condition | Candidate window | Layers | Success | Δ | broke/fixed | p | Compute |
|---|---|---:|---:|---:|---:|---:|---:|
| `prune1` | L16–31 (`0.5`) | 1 | 39.3% | +0.7 | 11/12 | 1.0000 | −3.3% |
| `prune2` | L16–31 | 2 | 38.5% | ±0.0 | 17/17 | 1.0000 | −6.0% |
| `prune3` | L16–31 | 3 | 49.6% | **+11.1** | 10/25 | 0.0167 | −7.9% |
| **`prune4`** | **L16–31** | **4** | **54.1%** | **+15.6** | 9/30 | **0.0011** | **−11.9%** |
| `window25` | L8–31 (`0.25`) | 4 | 54.1% | +15.6 | 9/30 | 0.0011 | −10.7% † |
| `prune4_gap3` | L16–31, gap 3 | 4 | 40.0% | +1.5 | 17/19 | 0.8679 | −10.9% |
| `prune4_early` | L2–31 (`0.08`) | 4 | 44.4% | +5.9 | 19/27 | 0.3020 | −10.9% |
| **`window875`** | **L28–31 (`0.875`)** | **4** | **8.1%** | **−30.4** | 43/2 | **5.9 × 10⁻¹¹** | **−10.6%** |

> † `window25` and `prune4` have **identical deleted layers and identical
> results** (① below) — yet their measured savings disagree by 1.2 points,
> −10.7% vs. −11.9%. **Two runs of the same computation wobble by that
> much**, so decimal differences in this column must not be read as
> effects. It is also the basis for saying below that "the four conditions'
> savings are effectively equal."

### ★ Same capacity, same compute, 45.9 points

The four 4-layer conditions **save effectively the same compute (−10.6% to
−11.9%)** while their Δs run **from +15.6 to −30.4.**

```
[17,23,25,27] / [17,20,23,26]  ->  +15.6   (−11.9%)   <- per task; standing-only [17,21,23,26]
[2,4,23,26]                    ->   +5.9   (−10.9%)
[17,23,27,31]                  ->   +1.5   (−10.9%)
[28,29,30,31]                  ->  −30.4   (−10.6%)
```

**Backbone, benchmark, method, capacity, cost — all equal. What changed is
the candidate-window flag (`--depth-min-layer` 0.5 / 0.08 / 0.875 — rows
1, 2, 4) and one selection spacing rule (`gap 3` — row 3, window at
default).** So a number a paper reports as "depth pruning, 4 layers,
+15.6" is not the method's performance but **a function of where that
implementation put its candidate window.** It is the most direct
demonstration of §0's thesis, with more variables controlled than the
benchmark-axis sign flip.

### ★★ But the swing itself depends on the cell — and follows the benchmark

We ran the same contrast — **default-window 4 layers vs. shifted-back 4
layers** — in all five cells. Capacity (4 layers) and compute saved (−10.6
to −16%) are fixed everywhere.

| Cell | baseline | Default window | Shifted-back window | **Paired gap** |
|---|---:|---:|---:|---:|
| **SpatialVLA / Fractal** | 84.4% | −17.8 (p=1.8e−4) | **−68.1** (p=1.2e−25) | **50.4** |
| **OpenVLA / Fractal** | 38.5% | **+15.6** (p=1.1e−3) | **−30.4** (p=5.9e−11) | **45.9** |
| UniVLA / Bridge | 81.2% | −2.1 (p=0.8036) | −8.3 (p=0.0963) | **6.3** |
| OpenVLA / Bridge | 15.6% | +1.0 (p=1.0000) | −4.2 (p=0.5413) | **5.2** |
| SpatialVLA / Bridge | 30.2% | −28.1 (p=1.5e−8) | −30.2 (p=3.7e−9) | **2.1** |

**The gap follows the benchmark, not the backbone.** The two Fractal cells
are 45.9 and 50.4; the three Bridge cells are 2.1, 5.2, 6.3. **About 40
points sit empty between the two groups, with no cell inside** (the two
closest, 6.3 and 45.9, are 39.6 apart).

> **The gap must not be computed by subtracting two rounded Δs.** It comes
> from the raw counts — OpenVLA/Fractal is (73−11)/135 = **45.93**, while
> adding the rounded +15.6 and −30.4 gives 46.0, off by 0.1 (logged in
> §7.1 ②). UniVLA/Bridge is 6/96 = **6.25**, exactly on the rounding
> boundary, standardized to 6.3. The two backbones each span both
> benchmarks and both point the same way.

And the split is **not explained by baseline height.** The Bridge baselines
are scattered — 15.6%, 30.2%, 81.2% — yet all three gaps are small. The
Fractal ones are 38.5% and 84.4%, and both are large.

Floor effects, cell by cell for the three Bridge cells:

| Cell | Room to the floor | Actual gap | Does the floor explain it? |
|---|---:|---:|---|
| OpenVLA / Bridge | 15.6 | 5.2 | **partly** — it can fall at most 15.6 |
| SpatialVLA / Bridge | 30.2 | 2.1 | **yes** — both conditions are at the floor, 2.1% and 0.0% |
| **UniVLA / Bridge** | **81.2** | **6.3** | **no** — 81 points of room, gap of 6.3 |

**UniVLA/Bridge breaks the floor-effect explanation.** It starts at nearly
the same height as SpatialVLA/Fractal (84.4%), yet the gaps differ
eight-fold, 50.4 vs. 6.3. So "Bridge cells have small gaps because they
have no room" does not hold.

So this section's claim is not "depth pruning is sensitive to layer
choice."

> **The sensitivity to selection is itself a property of the condition.** In
> some cells, layer choice makes 46–50 points; in others it cannot make even
> 2–6. And in our grid that split **follows the benchmark** — with 40 empty
> points between the two groups.

This lifts §0's thesis one level. **A paper that concluded "this method is
robust to layer choice" on one benchmark could swing 45.9 points on
another.** Even robustness claims do not hold from a single condition.

### How to count this "swing" — do not mix the two definitions

The number of selections we tried **differs per cell.** So we keep two
numbers apart.

| | Meaning | Comparable across cells? |
|---|---|---|
| **Paired gap** | default window vs. shifted-back window — **exactly two conditions** | **yes** — the table above |
| **Observed range** | max−min over **all** 4-layer selections tried in that cell | **no** — the condition counts differ |

By observed range the ordering changes:

| Cell | Selections tried | Observed range | Condition producing the lowest value |
|---|---:|---:|---|
| **UniVLA / Bridge** | 3 | **77.1** | front-including window L2–31 → **−79.2** |
| SpatialVLA / Fractal | 2 | 50.4 | shifted-back window |
| OpenVLA / Fractal | 4 | 45.9 | shifted-back window |
| OpenVLA / Bridge | 2 | 5.2 | shifted-back window |
| SpatialVLA / Bridge | 2 | 2.1 | shifted-back window |

**UniVLA/Bridge has the largest observed range (77.1) while its paired gap
is among the smallest (6.3).** The difference comes entirely from one
condition, `prune4_mid` (front-including window L2–31, `[2,4,26,30]`,
**−79.2**). That condition ran in only **two** of the five cells, with
opposite results — on OpenVLA/Fractal the same front-including window,
`[2,4,23,26]`, gives **+5.9.**

> **So the sentence "this backbone is the most sensitive" must not be
> written.** The observed range reflects what we happened to try in that
> cell. The cross-cell number is standardized to **the paired gap** alone;
> the observed range is quoted only as "the lowest value we saw in this
> cell."

> **The contrast is not perfectly uniform.** The two OpenVLA cells and
> UniVLA used "the last 4 layers" ([28,29,30,31]); the two SpatialVLA cells
> hit §3.5.2's generation-length blow-up under that condition, so we
> substituted **the top of the back half.** Each cell compares **its
> default selection against the back-most selection that cell can run**,
> not the same layer numbers across five cells (stated per §3.6's
> uniformity rule).

### Four checks

**① Widening the window by itself does nothing.** `window25` (L8–31) is
**bit-identical across 135 episodes** to `prune4` (L16–31)
(`compare_runs.py` → `reproduced exactly`). The top four most-redundant
layers were all ≥ L17 anyway, so widening the window changed no selection.
**The window changes the outcome only when it actually changes the
selection** — not by the mere fact of touching the flag.

**② The steep change sits between 2 and 3 layers.** 1 layer +0.7 → 2
layers ±0.0 → **3 layers +11.1** → 4 layers +15.6. An earlier draft said
"between 2 and 4"; measuring 3 shows the rise already there. 2 → 3 → 4 is a
gentle climb.

**③ The last four layers, L28–31, are essential.** `window875` has only
four candidates, forcing `[28,29,30,31]`.

| | Success | Grasp | Mean steps |
|---|---:|---:|---:|
| `move_near` | 11/60 | 0 | 71.2 |
| `pick_horizontal` | **0/25** | 0 | **80.0** (cap) |
| `pick_standing` | **0/25** | 2 | **80.0** |
| `pick_vertical` | **0/25** | 0 | **80.0** |

The three pick tasks are all at zero and all 75 episodes run to the step
cap. The policy stops acting. **"The back is safe" does not hold** — even at
the back, *which* back decides.

**④ A layer-level pattern — an observation, not a conclusion.** Sorting by
`pick_horizontal` alone (baseline 7/25, grasp 17/25):

| Deleted layers | Success | Includes L26 |
|---|---:|:---:|
| `[23]` | 5/25 | ✗ |
| `[23, 26]` | 6/25 | ✓ |
| `[17, 23, 26]` | 8/25 | ✓ |
| `[17, 20, 23, 26]` | 11/25 | ✓ |
| `[2, 4, 23, 26]` | **14/25** | ✓ |
| `[17, 23, 27, 31]` | 5/25 | ✗ |
| `[28, 29, 30, 31]` | 0/25 | ✗ |

Every combination that beats baseline contains L26, and every one without
it does worse. But `[23,26]` alone is 6/25, below baseline, so **L26 by
itself is not enough.** And n = 25 per task. **Recorded as a hypothesis
only.**
> Between `gap3` and `prune4`, two layers change at once on pick
> (`{20,26}` → `{27,31}`), so **L31 alone cannot be blamed.** What can be
> defended stops at "deleting L28–31 wholesale is very costly."

### And the baseline is a "grasps but cannot finish" state

| | baseline success | baseline grasp | `prune4` success | `prune4` grasp | Mean steps |
|---|---:|---:|---:|---:|---|
| `pick_horizontal` | 7/25 | **17/25** | 11/25 | 17/25 | 70.6 → 59.7 |
| `pick_standing` | 5/25 | **14/25** | 13/25 | 19/25 | 73.8 → 66.4 |
| `pick_vertical` | 3/25 | 4/25 | 7/25 | 9/25 | 75.0 → 66.8 |

On `pick_horizontal` the grasp count stays at 17 **while success alone goes
7 → 11.** The grasping ability did not improve — the rate of finishing
after a grasp did. The baseline's mean steps sit at 70–75, pressed against
the cap (80), then drop to 60–67.

**So the defensible sentence for this cell is:**

> OpenVLA/Fractal's original policy often grasps the object and still
> fails to finish the episode, running to the step cap. Deleting the
> **right** four layers reduces those failures and raises success. Which
> four are right is decided by the candidate window, and a wrong pick stops
> the policy altogether.

**Not "depth pruning improves VLA policies."** It is a case study in how a
large Δ in one cell can reflect that cell's baseline condition and
implementation details together — our own table put under the same
criticism we make of others' (§2.5).

> **Qualifier.** This window sweep ran in **one cell, OpenVLA/Fractal,
> only.** Whether the other four columns swing as widely, we did not
> measure. So we cannot write "selection sensitivity is a property of all
> VLAs" — we write **"the swing differed in every cell we measured."**

### (d) Split by task family, and what the average hides appears

Fractal's four tasks are two families — `move_near` must pick **the named
one** of three objects; the three `pick_coke_can` tasks have one
instruction and one target.

**SpatialVLA / Fractal, 4 layers removed**

| Family | Δ | broke/fixed | p |
|---|---:|---:|---:|
| named-target (`move_near`, n=60) | **−31.7** | 22/3 | 0.0002 |
| single-target (`pick_coke_can`, n=75) | −6.7 | 10/5 | 0.3018 |

The average, −17.8, hides this split. **§6 starts here.**

---

# §5. Results across cells

§4 looked inside cells. This section looks **between** them. Two runs share
no episodes, so nothing can be paired; what is compared is the **discordant
split** — whether "how many fixed, how many broken" differs between two
cells, asked with a **Fisher exact 2×2.** **This family has 43 tests**, so
Bonferroni gives α = 0.05/43 ≈ 0.0012.

> **Why 43.** The benchmark axis: two backbones that run both benchmarks ×
> seven conditions = 14. The backbone axis: on Bridge, three backbone pairs
> = 22 (only OpenVLA×UniVLA, which both run `prune8`, has eight; the other
> two pairs have seven), and on Fractal, one pair × seven = 7. Total 43. A
> **different family** from the grid's McNemar 38.
>
> ⚠️ **This number was wrong once, and wrong in the bad direction.** The
> generator's backbone-axis loop had the benchmark hardcoded to
> `"Bridge"`, so **the seven tests of Fractal's only backbone pair
> (OpenVLA vs. SpatialVLA) never ran at all.** The denominator is
> `len(rows)`, so missing tests make **α larger and the bar looser** — the
> omission made passing easier, not harder. Fixed and re-run, **two of
> those seven pass the correction, and one is a sign flip** (§5.2). In
> other words, the omission had been understating our own thesis. The 36
> and 42 in earlier drafts were, respectively, the bugged value and a
> hand-counted one.

## 5.1 Change only the benchmark and results split

Code and checkpoint untouched; only the benchmark changes.

| Backbone | Intervention | Bridge (fixed/broke) | Fractal (fixed/broke) | Sign | p |
|---|---|---|---|:--:|---:|
| OpenVLA | foveation log-polar | 28/10 (**+18.8**) | 13/39 (**−19.3**) | **flips** | **<0.0001** |
| OpenVLA | foveation blur | 26/9 (+17.7) | 27/39 (−8.9) | **flips** | 0.0017 |
| SpatialVLA | depth prune 1 | 9/19 (−10.4) | 14/3 (+8.1) | **flips** | 0.0018 |
| OpenVLA | action repeat 4 | 0/11 (−11.5) | 20/22 (−1.5) | same | 0.0038 |
| SpatialVLA | action repeat 4 | 10/22 (−12.5) | 6/60 (−40.0) | same | 0.0085 |
| SpatialVLA | depth prune 4 | 0/27 (−28.1) | 8/32 (−17.8) | same | 0.0176 |

**Two things must be read apart.** What Fisher asks is "does the
break-and-fix pattern differ between the two benchmarks," not "did the sign
flip." Of the six rows, **the sign actually flips in three**; the other
three keep the sign but **differ widely in size** (−11.5 vs. −1.5, −12.5
vs. −40.0, −28.1 vs. −17.8).

**And only the first row passes the correction (43 tests, α ≈ 0.0012).**
OpenVLA's log-polar foveation: +18.8 on Bridge, −19.3 on Fractal, with the
difference at p < 0.0001. The other five pass p < 0.05 but not the
correction.

So the precise sentence is: **one benchmark-axis sign flip firmly shown**,
**two more observed**, and **three cases of same sign but widely different
size.** §0 is written that way.

## 5.2 Fix the benchmark, change only the backbone — results split too

Same benchmark, backbone changed. Bridge has all three backbones, so three
pairs; Fractal has two, so one pair.

**Bridge (three backbones, 96-episode protocol)**

| Intervention | Left | Right | p |
|---|---|---|---:|
| action repeat 2 | SpatialVLA 21/9 (**+12.5**) | UniVLA 1/68 (**−69.8**) | **<0.0001** (1.7 × 10⁻¹³) |
| action repeat 4 | SpatialVLA 10/22 | UniVLA 0/78 | **<0.0001** (1.4 × 10⁻⁶) |
| depth prune 4 | OpenVLA 11/10 (+1.0) | SpatialVLA 0/27 (**−28.1**) | **<0.0001** (1.6 × 10⁻⁵) |
| depth prune 4 | SpatialVLA 0/27 | UniVLA 7/9 (−2.1) | **0.0004** |
| foveation blur | OpenVLA 26/9 (+17.7) | UniVLA 8/16 (−8.3) | 0.0029 |
| foveation log-polar | OpenVLA 28/10 | SpatialVLA 11/19 | 0.0031 |
| action repeat 2 | OpenVLA 3/11 | SpatialVLA 21/9 | 0.0037 |

**Fractal (two backbones, 135-episode protocol)**

| Intervention | Left | Right | p |
|---|---|---|---:|
| depth prune 4 | OpenVLA 30/9 (**+15.6**) | SpatialVLA 8/32 (**−17.8**) | **<0.0001** (4.3 × 10⁻⁷) |
| action repeat 4 | OpenVLA 20/22 (−1.5) | SpatialVLA 6/60 (**−40.0**) | **<0.0001** (1.4 × 10⁻⁵) |

**The backbone axis is stronger than the benchmark axis.** Six comparisons
pass the correction (α ≈ 0.0012), five of them at p < 10⁻⁴. All three
backbones take part, and the axis passes on **both benchmarks.**

**Fractal's first row is the strongest single line in this section.**
`depth prune 4`, same benchmark, same 135 episodes: **+15.6 on OpenVLA,
−17.8 on SpatialVLA** — the only cell pair where **both sides individually
pass the correction with opposite signs**, and the difference itself is
p = 4.3 × 10⁻⁷. Evidence **of the same strength as §5.1's benchmark-axis
flip (OpenVLA log-polar) exists on the backbone axis too.**

The asymmetry is itself a result. **"This method works on VLA" cannot be
carried by one backbone** — in our grid, changing the backbone shakes the
outcome harder than changing the benchmark.

> ⚠️ **The two axes use different cell sets.** UniVLA is Bridge-only, so it
> cannot join the benchmark axis, and Fractal's backbone axis has one pair.
> So **the benchmark axis rests on two backbones, and the backbone axis
> leans toward Bridge.** This is one reason a third benchmark is needed
> (§7 ①).
>
> **The two Fractal rows appeared for the first time in this audit** — the
> generator had been building the backbone axis for Bridge only (the ⚠️ at
> the head of §5). The values are the existing grid's; no new runs were
> made.

## 5.3 Which *variant* wins also changes the answer

**The two variants can be paired directly.** Log-polar and blur run the
same episodes under the same protocol, so instead of subtracting two Δs
against baseline, we pair the two conditions against each other and run
McNemar. An early draft did the former — exactly the move §3.3 tells people
not to make.

| Cell | log-polar | blur | Difference | lp-only wins / blur-only wins | p |
|---|---:|---:|---:|:--:|---:|
| **UniVLA / Bridge** | 86.5% | 72.9% | **+13.5** | 19 / 6 | **0.0146** |
| SpatialVLA / Fractal | 85.2% | 83.0% | +2.2 | 13 / 10 | 0.6776 |
| OpenVLA / Bridge | 34.4% | 33.3% | +1.0 | 19 / 18 | 1.0000 |
| SpatialVLA / Bridge | 21.9% | 30.2% | −8.3 | 8 / 16 | 0.1516 |
| **OpenVLA / Fractal** | 19.3% | 29.6% | **−10.4** | 17 / 31 | 0.0595 |

**Which variant is better flips per cell.** On UniVLA/Bridge log-polar wins
by 13.5 points; on OpenVLA/Fractal blur wins by 10.4. The two ends are 24
points apart.

> ⚠️ **The OpenVLA/Bridge row crosses the two trees.** In that cell only
> the log-polar comes from the legacy `RetinaBased/` tree while blur comes
> from `results/` (the same situation as §4.3 b). The two trees' baselines
> are 96/96 bit-identical, so the juxtaposition is legitimate, but this row
> alone mixes provenance — noted. As it happens the row claims nothing
> (+1.0 / p = 1.0000), so the argument is unaffected.
>
> ⚠️ **No cell passes correction, though.** There are five comparisons, so
> α = 0.05/5 = 0.01, and the strongest (UniVLA/Bridge) is p = 0.0146.
> **"Variant choice changes the outcome" is an observation, not an
> established result.** What is established is §4.3 (b)'s keep axis —
> within one variant, moving keep runs from +4.2 to +30.2 with the top end
> at p = 4.2 × 10⁻⁷.

**So writing "we applied foveation" in a paper decides nothing.** The
variant, the keep value, and the backbone all have to be stated — and of
those, **only "the keep value moves the outcome most" is something we
established.**

## 5.4 The amount saved does not predict the outcome

| Axis | Saved | Success in our grid |
|---|---:|---|
| action repeat 4 | **−75%** | negative in all five cells; largest drop −81.2 |
| action repeat 2 | −50% | splits, −69.8 to +12.5 |
| depth prune 4 | −11 to −16% | splits, −30.4 to +15.6 |
| foveation | **≈0%** | the campaign's largest gains — +18.8 at keep 20%, **+30.2 at keep 100%, which discards nothing** |

**Across axes, the axis that saves the most loses the most, and the axis
that gains the most saves nothing.** §4.3 (b) pushed this one step
further — even within foveation, **the setting that discards the least
gains the most** (keep 100% at +30.2).

**And within an axis there is no correlation at all.** The four conditions
of §4.4 (c) are decisive:

| Condition | Saved | Δ |
|---|---:|---:|
| `prune4` | −11.9% | **+15.6** |
| `prune4_early` | −10.9% | +5.9 |
| `prune4_gap3` | −10.9% | +1.5 |
| `window875` | −10.6% | **−30.4** |

**The savings all sit within 1.3 points of each other while the Δs spread
45.9 points.** So "how much was saved" cannot predict "what was lost."
There is no trade-off curve — at least not in the cells we measured.

**And even the saving itself is decided by the selection.** On
SpatialVLA/Fractal, the same 4 layers save −15.9% under the default window
and only **−4.2%** under the shifted-back window (788 ms vs. 898 ms) — as
the policy's success drops, it emits more action tokens. Pick worse layers
and the sign flips — deleting the layers just before the output (L22–24)
turned 937 ms into 3390–5466 ms, **4× slower** (§3.5.2): the policy cannot
emit the stop token and generation runs to the cap. **An intervention
installed to save compute, given the wrong layers, spends four times
more.** The matching contrast on OpenVLA was fine at −10.6% — that
backbone's action token count is fixed.

> So even the sentence "this method saves X% compute" does not hold without
> naming the backbone and the layer selection. The **cost half** of the
> premise §1 set out to test also fails here — not just the accuracy half.

This is our answer to the premise §1 said it would test ("give up a little
accuracy, save a lot of compute"). **In our grid, that exchange does not
hold.** Precisely: there is **no cell that yields a significant gain on
both benchmarks while saving compute.**

> The closest is OpenVLA depth prune 4. Its sign holds on both benchmarks
> (Bridge +1.0, Fractal +15.6) and it saves −11%. But the Bridge +1.0 is
> p = 1.0000, indistinguishable from chance, and the Fractal +15.6, as
> §4.4 (c) shows, **falls to −30.4 when only the candidate window changes
> at equal capacity and cost.** **We cannot write "no cell saves while
> keeping its sign"; we must write "no cell saves while being significant
> on both benchmarks."**

---

# §6. Mechanism — the record of building one hypothesis and rejecting it with our own data

> **Conclusion first.** §6 originally set out to claim that cutting
> capacity **damages the ability to pick out the named object first.**
> Opening up the task **from the inside**, in the backbone and condition
> where the drop actually happened, **we found no evidence supporting that
> hypothesis.** What was damaged was not the pointing but **the doing.**
> This section keeps the full record — hypothesis, test, rejection —
> because deleting a result for pointing the wrong way is exactly the
> behavior §7.1 ② warns about.

## 6.1 The evidence so far is a **between-task** comparison

The observation: on SpatialVLA/Fractal, deleting the last 4 layers drops
`move_near` sharply, 83.3 → 51.7, while the same run's three coke-can tasks
move much less — 84.0 → 88.0, 84.0 → 80.0, 88.0 → 68.0. Switch the task
version to `move_near_v1` and it is the same direction, 86.7 → 58.3. From
this came the reading "**the ability to pick out the named object is
damaged first.**"

But that reading sits on the **difference** between `move_near` and
`pick_coke_can`, and those tasks differ in several ways at once —
instruction complexity, number of objects in the scene, episode length,
success criteria, and above all `move_near` must point out two objects
while pick points out one. Which difference produced the drop, this
comparison cannot separate. So §6 was, until here, **a hypothesis** —
consistent with the observation, but so are other stories.

## 6.2 The tool for measuring **inside** the task was already in the environment

`move_near_in_scene.py` computes these five every step and exports them as
`info["episode_stats"]`:

| Field | Definition |
|---|---|
| `moved_correct_obj` | the named object moved more than 3cm, and more than any other object |
| `moved_wrong_obj` | some other object moved more than 3cm, and more than the named one |
| `near_tgt_obj` | the named object ended up next to the target (bbox-based radius) |
| `is_closest_to_tgt` | the named object is closer to the target than any other |
| `all_obj_keep_height` | nothing fell off the table |

Success is the AND of four of these. Failure is the rest — and **how** it
failed remains. We split failures into four buckets
(`experiments/mechanism_move_near.py`):

| Bucket | Meaning | Reading |
|---|---|---|
| `wrong_object` | moved an object we did not name | the arm works; the **pointing** is wrong |
| `misplaced` | moved the right object to the wrong place | source pointing fine; **destination pointing** wrong |
| `no_contact` | nothing moved more than 3cm | the **doing** failed |
| `dropped` | something fell off the table | — |

`wrong_object` and `no_contact` make opposite predictions. If pointing is
damaged the first should grow; if control is damaged, the second. **Both
are measured inside `move_near`**, so §6.1's confounds do not apply here.

## 6.3 On OpenVLA/Fractal it was already recorded — and the answer is not the pointing side

The OpenVLA harness saves the terminal `info` wholesale. So **the answer
was already inside episodes we had run — not one new simulation was
needed.** Re-reading 9 conditions × 60 episodes:

| Condition | Success | dropped | **wrong_object** | misplaced | **no_contact** |
|---|---:|---:|---:|---:|---:|
| baseline | 37 | 1 | **0** | 6 | **16** |
| depth_prune1 | 42 | 2 | **0** | 5 | **11** |
| depth_prune2 | 37 | 6 | **0** | 8 | **9** |
| depth_prune4 | 42 | 5 | **2** | 7 | **4** |
| depth_prune4_early | 30 | 1 | **1** | 9 | **19** |
| foveate (log-polar) | 12 | 6 | **1** | 6 | **35** |
| foveate_blur | 18 | 5 | **3** | 15 | **19** |
| action_repeat2 | 37 | 3 | **0** | 10 | **10** |
| action_repeat4 | 31 | 4 | **1** | 9 | **15** |

Paired — looking only at episodes the baseline solved and the condition
lost:

| Condition | Lost | dropped | **wrong_object** | misplaced | **no_contact** |
|---|---:|---:|---:|---:|---:|
| depth_prune1 | 6 | 1 | **0** | 3 | 2 |
| depth_prune2 | 7 | 3 | **0** | 2 | 2 |
| depth_prune4 | 5 | 0 | **1** | 3 | 1 |
| depth_prune4_early | 12 | 1 | **0** | 2 | 9 |
| foveate | 28 | 4 | **0** | 4 | **20** |
| foveate_blur | 28 | 3 | **1** | 11 | 13 |
| action_repeat2 | 8 | 1 | **0** | 3 | 4 |
| action_repeat4 | 12 | 1 | **0** | 4 | 7 |

**How to read it — as shares of failures.** Failure counts run from 18 to
49 per condition, so raw counts side by side misread the arithmetic growth
as an effect (the same reason as the correction in §6.5 ②).

| Condition | Success | Failures | `no_contact` | `wrong_object` |
|---|---:|---:|---:|---:|
| baseline | 37 | 23 | 16 (69.6%) | 0 (0.0%) |
| prune 1 | 42 | 18 | 11 (61.1%) | 0 (0.0%) |
| prune 2 | 37 | 23 | 9 (39.1%) | 0 (0.0%) |
| prune 4 | 42 | 18 | 4 (**22.2%**) | 2 (11.1%) |
| prune 4 early | 30 | 30 | 19 (63.3%) | 1 (3.3%) |
| foveate log-polar | 12 | 48 | 35 (**72.9%**) | 1 (2.1%) |
| foveate blur | 18 | 42 | 19 (45.2%) | 3 (7.1%) |
| action repeat 2 | 37 | 23 | 10 (43.5%) | 0 (0.0%) |
| action repeat 4 | 31 | 29 | 15 (51.7%) | 1 (3.4%) |

**① `wrong_object` barely appears on this backbone.** Across all nine
conditions it is **0–11%** of failures, and the baseline has zero. The
interventions do not create this failure type.

**② Depth pruning lowers the `no_contact` share.** 69.6% → 61.1% → 39.1%
→ **22.2%** as layers are removed — consistent with success rising
37 → 42. Nothing here supports §6.1's reading.

**③ But foveation cannot be pinned as "execution damage" either.**
Log-polar's `no_contact` more than doubles **in count**, 16 → 35, but **as
a share it barely moves, 69.6% → 72.9%.** Failures grew from 23 to 48; the
**composition** of failure did not change. The draft wrote "clearly the
execution side" here; divide by the denominator and that basis disappears.
What can be said stops at **"failures grew a lot and their composition
resembles baseline's"** (§7.1 ②).

**But none of this measures §6.1's claim.** On OpenVLA/Fractal, depth
pruning did not drop `move_near` at all (37 → 42 — it rose). We had run the
tool on a backbone with no drop to explain. The backbone that did drop —
SpatialVLA — was running on a harness that discarded `episode_stats`
(§6.6). So we fixed the harness and re-ran exactly those two conditions.

## 6.4 ★ The decisive measurement — SpatialVLA / Fractal (2026-08-11)

The very backbone and condition where §6.1's drop happened. With the fixed
harness we re-ran `baseline` and `depth_prune4` on `move_near`.

**First, we confirmed it is the same run.** Episode-level comparison with
`compare_runs.py`:

```
baseline      ref 50/60  new 50/60  -> identical    VERDICT: reproduced exactly
depth_prune4  ref 31/60  new 31/60  -> identical    VERDICT: reproduced exactly
```

The deleted layers are `[8, 9, 10, 20]`, same as on August 6. So **not one
episode's success/failure changed — only more was recorded.** The exact
same pair as §6.1's −31.7.

| | Success | dropped | **wrong_object** | misplaced | **no_contact** |
|---|---:|---:|---:|---:|---:|
| baseline | 50 | 4 | **1** | 5 | **0** |
| depth_prune4 | 31 | 7 | **4** | 6 | **12** |

Paired per bucket (same 60 episodes):

| Bucket | baseline → prune4 | new / resolved | p |
|---|---|---:|---:|
| **`no_contact`** | **0 → 12** | 12 / 0 | **0.0005** |
| `wrong_object` | 1 → 4 | 4 / 1 | 0.3750 |
| `misplaced` | 5 → 6 | 5 / 4 | 1.0000 |
| `dropped` | 4 → 7 | 6 / 3 | 0.5078 |

Looking only at the 22 episodes the baseline solved and prune4 lost:
`no_contact` 10, `misplaced` 5, `dropped` 5, **`wrong_object` 2.**

**Only one thing moved: `no_contact`.** And it moves from a baseline of
**exactly zero** to 12 — the original policy moved something by more than
3cm in all 60 episodes, and with 4 layers deleted, 12 episodes end without
touching anything. `wrong_object` grows 1 → 4 but is indistinguishable from
chance (p = 0.375).

**So §6.1's hypothesis is not supported by our measurement.** Opening up
the very cell where the drop happened, from inside the task, what was
damaged is **not the pointing but the doing.**

## 6.5 So what is §6, in the end

**① What was rejected.** The reading "cut capacity and the ability to pick
out the named target is damaged first" is not supported by our data. Its
only evidence was the between-task comparison (§6.1), and replacing that
with a within-task measurement produced no trace of the predicted signal.

**② What remains — but never counted in raw numbers.**

> ⚠️ **The draft of this item used the wrong method.** Failure counts
> differ per condition (from 10 to 56), yet the draft lined up the buckets'
> **raw counts** and read "success drops, `no_contact` grows." When
> failures grow, every bucket's count grows arithmetically — much of that
> observation is automatic. **We told others to look at the split behind
> the average, then failed to divide by the denominator ourselves** —
> logged in §7.1 ②. Below is the recount, **as shares of failures.**

One task, `move_near` (n=60), all eleven conditions:

| Backbone / condition | Success | Failures | `wrong_object` (share of failures) | `no_contact` (share of failures) |
|---|---:|---:|---:|---:|
| OpenVLA baseline | 37 | 23 | 0 (**0.0%**) | 16 (**69.6%**) |
| OpenVLA prune 4 | **42** | 18 | 2 (11.1%) | 4 (**22.2%**) |
| OpenVLA prune 3 | **42** | 18 | 2 (11.1%) | 7 (38.9%) |
| OpenVLA `gap3` | 38 | 22 | 1 (4.5%) | 6 (27.3%) |
| OpenVLA `early` | 30 | 30 | 1 (3.3%) | 19 (63.3%) |
| OpenVLA blur | 18 | 42 | 3 (7.1%) | 19 (45.2%) |
| OpenVLA log-polar | 12 | 48 | 1 (2.1%) | 35 (72.9%) |
| OpenVLA `window875` | 11 | 49 | 0 (**0.0%**) | 41 (**83.7%**) |
| SpatialVLA baseline | 50 | 10 | 1 (10.0%) | 0 (**0.0%**) |
| SpatialVLA prune 4 | 31 | 29 | 4 (13.8%) | 12 (41.4%) |
| SpatialVLA `prune4_back` | **4** | 56 | 7 (**12.5%**) | 42 (**75.0%**) |

**Two things we can say.**

**(i) `wrong_object` is a minority of failures everywhere** — between 0.0%
and 13.8% — and **it does not grow with intervention strength.** In
OpenVLA's most-failing condition (`window875`, 49 failures) it is **0.0%**,
and in the campaign's most-failing condition (SpatialVLA `prune4_back`, 56
failures) it is 12.5% — while that backbone's **baseline is already
10.0%.** The interventions do not create this failure type.

**(ii) On SpatialVLA, the `no_contact` share rises with intervention
strength** — 0.0% → 41.4% → 75.0%. Three points, monotone.

**One thing we cannot say.** **On OpenVLA the relation is not monotone.**
The baseline is already at 69.6%; prune 4, which *raises* success, brings
it down to 22.2%; and blur (18 successes) sits at 45.2%, below both
log-polar (12 successes, 72.9%) and baseline (37 successes, 69.6%). The
draft's **"no exceptions" is not a fact.**

So the sentence §6 leaves behind is:

> **Traces of damaged target selection are a minority of failures in every
> condition and do not grow with intervention strength. Failures
> concentrate instead in "touched nothing," and the relation to strength is
> monotone on one backbone — SpatialVLA, the one with three points.**

**③ What this tool can and cannot rule out.** `moved_wrong_obj` fires only
when another object actually **moves** more than 3cm. An episode where the
arm reached toward the wrong object but failed to push it lands in
`no_contact`, not `wrong_object`. So **we cannot write "the pointing is
intact."** What we can write:

> **We found no trace of wrong pointing carried through to manipulation;
> instead, failures that manipulate nothing at all grew from 0 to 12.**

One fact narrows this qualifier: the baseline's `no_contact` was **zero.**
The original policy moved something in all 60 episodes. If only the
pointing had broken while execution stayed intact, those 12 episodes
should still have moved *something*.

**④ So §6.1's between-task drop is still unexplained.** That `move_near`
is −31.7 while `pick_coke_can` is −6.7 is real and significant (§4.4 d).
What we showed is that **the difference is not explained by a difference in
pointing ability** — not what does explain it. The remaining candidates —
trajectory length, required precision, scene complexity from object count
— are not separable with our data. **§6 stands as a negative result.**

> There was a temptation to delete this section. It was not the direction
> we wanted, and a whole section amounts to "our hypothesis was wrong." We
> keep it for two reasons. First, this paper's thesis is **"a number
> measured in one condition cannot carry a claim"** — and claiming a
> mechanism from one between-task comparison is exactly that error; there
> has to be a record of us applying our own standard to ourselves. Second,
> the concentration into `no_contact` is **less interesting but more
> certain** than what §6.1 tried to claim.

## 6.6 The harness change

We added a function to
`SpatialVLA/experiments/tome/tome_spatialvla_eval.py` that records only the
keys the task actually reported:

```python
_OUTCOME_KEYS = (
    "moved_correct_obj", "moved_wrong_obj", "near_tgt_obj", "is_closest_to_tgt",
    "is_src_obj_grasped", "consecutive_grasp", "src_on_target",
)

def outcome_detail(final_info) -> dict:
    """-> {"env_<key>": value} for whatever this task's env actually reported."""
    stats = (final_info or {}).get("episode_stats") or {}
    return {f"env_{k}": bool(stats[k]) for k in _OUTCOME_KEYS if k in stats}
```

Keys a task does not define are **not written.** Pick tasks have no "wrong
object," and writing `False` there would read as if it were measured.

UniVLA (`adaptive_sparse_vla/eval.py`) and OpenVLA (`simple_eval.py`)
already save `final_info` wholesale, so nothing needed changing. But UniVLA
is Bridge-only and has no `move_near` — on this axis UniVLA is out from the
start (§7 ②).

---

# §7. What is settled and what is open

## 7.0 Campaign status — what has finished running

**The grid is full.** Five columns (backbone × benchmark) × **eight
conditions** — one original policy + seven interventions (repeat 2/4,
foveation ×2, prune 1/2/4) — with **no gaps.** (The 38 **paired tests**
§3.3 counts exclude the baseline — the 35 of this table's five columns ×
seven conditions, plus the two `prune8` cells and one `prune2+repeat2` —
while the eight here are **runs**, baseline included.)

| Column | Original | repeat 2/4 | foveation ×2 | prune 1/2/4 | Status |
|---|---|---|---|---|---|
| OpenVLA / Bridge | ✓ | ✓ | ✓ | ✓ (+8 layers) | **done** |
| OpenVLA / Fractal | ✓ | ✓ | ✓ | ✓ | **done** |
| SpatialVLA / Bridge | ✓ | ✓ | ✓ | ✓ | **done** |
| SpatialVLA / Fractal | ✓ | ✓ | ✓ | ✓ | **done** |
| UniVLA / Bridge | ✓ | ✓ | ✓ | ✓ (+8 layers) | **done** |

On top of this come the diagnostics — determinism re-checks
(`baseline_rerun`, 85/85 and 24/24), region controls (`depth_prune4_early`
/ `_mid` / `_back`, `depth_prune1_back`), the task-version contrast
(`move_near_v1`), one combination (`prune2 + repeat2`), the foveation
capacity sweep (**keep 10 / 40 / 100** — keep 20 is already a grid cell,
so these three complete the four-point curve), and the §6 mechanism
measurement.

**Every run is finished.** The foveation capacity sweep (keep
10/20/40/100, all four points at 96 episodes, §4.3 b), OpenVLA/Fractal's
depth window sweep (`prune3`, `prune4_gap3`, `window25`, `window875`, 135
episodes), and the widening of that contrast **to all five cells**
(§4.4 c) are complete. The §6 mechanism measurement and the determinism
re-checks are done too.

**What remains open cannot be opened by more simulation.** A third
benchmark (①), a UniVLA/Fractal checkpoint (②), a named-target task other
than `move_near` (③) — all three need a new benchmark or a new
checkpoint.

> **One optional item.** OpenVLA / Fractal at **8 layers removed.** Not a
> grid row (see the note at §4.4 a), so its absence breaks no uniformity —
> but that cell's 4-layer point is the campaign's largest gain at +15.6,
> so whether it keeps rising or bends at 8 would strengthen §4.4's
> capacity curve. SpatialVLA is already at the floor at 4 layers (Bridge
> 2.1%), so there is nothing to measure at 8.

---

The three big items.

**① A third benchmark.** With two, "those two are peculiar" is still
sayable; with three, it becomes a pattern. The biggest open item — and
**the only one more simulation can close** — no amount of filling other
conditions changes the number of benchmarks.

**② UniVLA/Fractal cannot be filled — which actually enlarges ①.**
UniVLA's authors evaluated SimplerEnv on Bridge only and released a
Bridge-only checkpoint. So the benchmark axis's sign-flip evidence comes
**from OpenVLA and SpatialVLA only.** The backbone axis is carried by all
three backbones and passes correction on **both benchmarks** (repeat 2 at
+12.5 vs. −69.8 on Bridge; `prune 4` at +15.6 vs. −17.8 on Fractal). Even
so, four of the backbone axis's six passes are on Bridge, so the weight
leans one way — and the fact that the two axes use different cell sets
must be stated in the paper, not hidden. One more reason a third
benchmark is needed: right now the benchmark axis rests on two backbones.
> One alternative. **If UniVLA has a LIBERO checkpoint**, UniVLA gains a
> second benchmark and the campaign gains its third. Worth checking.

**③ Only two task families — and this one simulation can open.** §6's
decisive run happened and did not support the hypothesis (§6.4). So **why
`move_near`'s −31.7 is larger than `pick_coke_can`'s −6.7 remains
unexplained.** The remaining candidates — trajectory length, required
precision, scene complexity — cannot be separated by the current grid,
because Fractal has only **two** families in it.

**The three drawer tasks open that door** (the warning box in §3.2).
`open_drawer`, `close_drawer`, and `place_in_closed_drawer` are already
registered in the harness and are **articulated-object manipulation** — a
third ability type belonging to neither family. With a third family,
"does it fall because it is a named-target task, or just because it is
harder" becomes separable for the first time.

| Scope | Runs | Cost (est.) | What it buys |
|---|---|---|---|
| **Into the grid** | Fractal's two columns × 8 conditions × 3 drawer tasks | **over 15 hours** | fills all three categories of the standard protocol |
| **Diagnostic only** | two columns × {baseline, prune4, foveate} × `open`+`close` | **about 6 hours** | a third point on §6's family axis |

Under §3.6's rule **diagnostics need not be uniform**, so the second
option is available. Numbers obtained that way are used only as §6's
supporting evidence, never as grid cells. **Lower priority than the third
benchmark (①)** — ① extends the thesis's axes; this raises one section's
resolution.

**⑤ We did not record the execution environment.** Result files carry no
GPU, and inference times cannot identify the card (§3.4.0). So "all five
cells ran on the same hardware" **cannot be proven from the records.** The
size is bounded at about 3.1 points (§3.4.0's two-baseline contrast), but
that is an after-the-fact estimate. **The next campaign must stamp GPU,
driver, and library versions into the result files** — one line, and this
open item disappears entirely.

**④ What we measured is "success rate," not "ability."** §6.5 ③'s limit
generalizes here. `episode_stats` gives the terminal state; which internal
failure produced that state is not observed. Every sentence we can defend
has the form **"in this condition, this metric moved this much"** — and
"what broke" is written only to the degree of §6.5 ② — that is, only at
the level of correlation.

---

## 7.1 What moved and what did not

Sentences change every time results come in. We record which changes are
normal and which are not — without that distinction, "the story keeps
changing" and "an experiment is in progress" look the same.

**What did not move (the thesis)**

- The same intervention's effect **does not keep its sign across backbones
  and benchmarks.**
- Therefore **a number measured in one condition cannot carry a claim about
  the method.**
- **Episodes must be paired instead of subtracting two success rates.**
- **Averages hide the task-family split.**

Each new result made these four stronger, not weaker. Re-measuring UniVLA
foveation on matched hardware gave the same conclusion; pairing the
SpatialVLA legacy runs produced one more "cell that cannot carry a claim";
and ShortGPT's Limitation confirmed the fourth item one domain up.

**What moved ①: because new measurements came in — normal**

| What | Before | After | Why |
|---|---|---|---|
| UniVLA foveation log-polar | +8.3 | +5.2 | re-measured the baseline on the same card |
| UniVLA/Bridge depth axis | 1·2 layers only | 1·2·4·8 layers + range contrast | ran more conditions |
| SpatialVLA/Bridge foveation | †legacy | paired −8.3 (p = 0.20) | re-measured with records kept |
| §2.4 (e) "only the backbone differs" | asserted | region confound stated | checked the layers actually deleted |
| §3.6 region control | at dose 4 only | region also irrelevant at dose 1 (Δ −2.1, p = 0.80) | ran `depth_prune1_back` |
| §6 mechanism | "pointing dies first" (hypothesis) | **rejected.** What is damaged is execution (`no_contact` 0 → 12, p = 0.0005) | measured inside the task, in the cell that dropped |
| §6.5 ② | "no exceptions," from raw bucket counts | recounted **as shares of failures**; OpenVLA is not monotone | failure counts ran 10–56, so raw-count comparison did not hold |
| §4.3 foveation | "it helps by discarding the periphery" | **compression is not the source of the gain** — keep 100% is the maximum | ran the capacity sweep |
| §4.3 (b) "the round trip itself is lossy" | claimed from reading code | **measured in pixels** — at keep 100% the periphery degrades 2.7× more than the center | passed a real Bridge observation through the transform |
| §4.4 OpenVLA/Fractal +15.6 | "removing 4 layers helps" | **one window flag spans +15.6 to −30.4** | ran the four-condition window sweep |
| §4.4 selection sensitivity | "depth pruning is sensitive to layer choice" | **the sensitivity itself depends on the cell, and follows the benchmark** (Fractal 45.9·50.4 vs. Bridge 2.1·5.2·6.3) | ran the same contrast in five cells |
| §4.4 depth drop-off (**OpenVLA/Fractal**) | "between 2 and 4 layers" | **between 2 and 3** (already +11.1 at 3) | ran `prune3`. **SpatialVLA/Bridge's "between 2 and 4" stands** — that cell never ran 3 layers, so it cannot be narrowed |
| §3.5 implementation differences | one (`min-layer` meaning) | **three** — argument meaning · last-layer protection · EOS dependence | stepped on two more while widening the window sweep to five columns |
| §5.4 cost | "each intervention has a fixed saving" | **selection changes even the sign of the saving** (−10.6% vs. 4× more) | observed the generation-length blow-up on SpatialVLA |
| §3.8 baseline check | did not exist | **all five cells compared against the authors' values** — four higher, one −4.2 | opened the three backbone papers and checked the right column too |
| the UniVLA log-polar cell | empty as `--` in the grid | **recovered.** +5.2 (p = 0.36) | found in history that `git add results/` had committed missing files as deletions |
| §4.4 "swing" | nearly compared one number across cells that tried different condition counts | separated **paired gap** (for comparison) and **observed range** (within a cell) | UniVLA was 1st by observed range (77.1) and 4th by paired gap (6.3) |

**What moved ②: because we got ahead of the data — not normal**

| What | What was wrong |
|---|---|
| "the tables of **three** prior papers" | Two papers. The FastV rows are not an independent source |
| "EfficientVLA FLOPs **−28.9%**" | reduced **to** 28.9% = a 71.1% cut |
| "all 12 settings" | did not disclose that the 2 Random Dropping rows were excluded |
| "MoLe's gain comes from the **router**" | Table 5 says the opposite; distillation makes it |
| "**no cell** saves compute while keeping its sign" | OpenVLA depth prune 4 is the counterexample |
| §0's depth-pruning sign flip (−10.4 / +8.1) | wrote Δ but **no p.** Neither passes correction (0.0872 / 0.0127) |
| §0 "**both** foveation values pass correction" | the Bridge +18.8 does not (p = 0.0051). The generator printed `**`, transcription turned it into `***` |
| Bonferroni "15 comparisons, α ≈ 0.0033" → then "**42**" → then "**35**" | **None of the three is the number of tests actually run.** The generator's paired table has **38 rows** — 5 cells × 7 conditions plus two `prune8` cells and one `prune2+repeat2`. α is 0.0013. Fortunately **the same eight cells pass under 35, 38, or 42** — rank 8 at 1.07 × 10⁻³ and rank 9 at 5.10 × 10⁻³ leave the gap empty. The "ranks 10·11" wording was wrong too (those p-values are ranks 8·9). **Fixed three times, hand-counted three times, wrong three times** — the denominator is now the table's row count |
| §0 UniVLA action repeat 2 "**−70.8**" | re-paired against the L4 baseline it is −69.8 |
| "sweeping keep **upward is pointless** — Δ→0 is forced by design" | written without reading the code. `keep` only sets sample density **after** the warp; the round trip always happens. keep 100% is not the original, and that direction was exactly the one that decided the question |
| "**whichever four** you pick, front or back, success rises" | `window875` is −30.4. Two of four conditions rise. Generalized from the three diagnostics (`prune4`, `prune4_early`, `gap3`) alone |
| asserting `prune4`'s layers as `[17,23,25,27]` | that is `move_near`'s set alone. The ranking is per-task, so the pick sets are `[17,20,23,26]` / `[17,21,23,26]` |
| §2.5 "the **basis** of §6" · A.3 "**proposed** in §6" | sentences from before §6 was rejected were still standing. The **fact** of the split in others' tables stands; **why it splits is open** |
| §2.4·A.3 "the ranking is recalibrated **per episode**" | the code calibrates **once** per run (`calibrated` flag). Layers differ per task because each task is a separate process |
| A.1 "the authors **do not see it**" | only "it is not in the text" is verifiable |
| A.3 ShortGPT headline "MMLU **55.0 → 52.2**" | the introduction's value. **Table 2's same setting is 55.00 → 54.69**, 2.5 points apart. An internal inconsistency of the paper; we use the table value |
| §6.3 ② "foveation's damage is **clearly execution-side**" | counts go 16 → 35, but **as shares, 69.6% → 72.9%** — almost unchanged. Failure composition did not move — the same denominator error as §6.5 |
| §5.3 variant comparison | **subtracted two Δs** against baseline. The variants run the same episodes, so they must be **paired directly** — the very move §3.3 forbids |
| §5.1 "the other five point the same way" | of six rows, **three flip sign** and three keep it (differing in size) |
| §4.1 "on SpatialVLA/Fractal no intervention **exceeds +5**" | `depth prune 1` is **+8.1** |
| §0 "§5's sign-instability pattern exists in prior papers" | what §2.5 shows is the **task split** (move_near vs. pick), not a benchmark sign flip |
| Appendix A.6 "these **six** papers" | five were read closely. The table's sixth row is this work |
| §4.3 (c) "the two variants **share the same `keep` meaning**" | they do not. Blur's keep is **the area of the central disc left original**; log-polar's keep is **the fraction of samples kept after the warp.** At keep=100%, blur is a no-op (`frame.copy()`) and log-polar is the campaign's largest-gain condition |
| §4.3 (b)·Overview §② "the center is **barely touched** (1.2)" | quoted the mean only. Re-measured at 224×224, even keep=100% changes **71% of pixels with a max of 187/255**, and 2.2% of the center moves by 10+. The mean is small because most of the image is background |
| §4's tables never said **what bold means** | §4 bolds `p < 0.05` (uncorrected, 12 cells); `Overview.md` bolds **corrected** (8 cells). Different rules, neither stated — so **the same +12.5 was bold in one document and not the other.** The rule and an example now head §4 |
| §7.0 "the grid is full … **the only gaps are these three**" | self-contradictory within one sentence; the table right below has **no gaps.** A sentence from the era when the grid had holes. Also explained why "eight conditions" (runs incl. baseline) differs from §3.3's "seven" (tests excl. baseline) |
| §7.0 "capacity sweep (keep 10 / 40 / 100)" vs. "keep 10/20/40/100" four lines later | both true, but three vs. four within four lines — keep 20 is a grid cell, so only three more were run as diagnostics. The reason is now in parentheses |
| §4.3 (b) "keep 100% is the maximum in **three of four** tasks" | counting: **two** (`carrot` 8/24, `eggplant` 19/24). `spoon` and `stack` bend at keep 20% (10/24, 11/24). **The total is monotone while half the tasks are not** — the very shape we criticize in §2.5, in our own table. Added the per-task table and narrowed the conclusion to the total |
| §4.3 (b) keep axis | presented +30.2 alone, but that value **crosses two trees.** Added the two tree-pure comparisons (old tree +18.8 p=0.0051; new campaign keep10→keep100 +26.0 p=0.00047), closing the confound |
| RelatedWork §2.2 (b) "**three backbones** agree to the decimal" | the three cited are OpenVLA/Bridge, SpatialVLA/Bridge, **SpatialVLA/Fractal** — **three cells of two backbones** — and the UniVLA values (−52.3 / −76.6) were missing. All five cells now listed |
| RelatedWork §2.4 (e) "on SpatialVLA: −10.4 at 1 layer, −17.8 at 4" | −10.4 is **Bridge's** 1 layer; −17.8 is **Fractal's** 4 — **two cells mixed.** And Fractal's 1 layer is **+8.1**, opposite in sign |
| RelatedWork §2.4 (e) "Δ from −2.1 to **−77.1**" | **−79.2** (81.2% → 2.1%). 77.1 is `depth_prune8`'s **success rate**, not a Δ |
| RelatedWork A.6 · §2.6 "Gaze-Reg: **1 backbone · 1 LIBERO**" | the paper runs **LIBERO's four suites (OpenVLA) + Gym-Aloha + Pi-0 real robot.** Our conclusion (no axis crossing) stands, but we had shrunk someone's experimental scope |
| RelatedWork A.6 "VLA-Cache: **2 benchmarks**" | LIBERO and SIMPLER plus a **Kinova Jaco2 real robot**, and LIBERO runs on both OpenVLA and **OpenVLA-OFT** |
| RelatedWork A.6 "MoLe-VLA: **RLBench only**" | there are also **3 Franka FR3 real-robot tasks** (present in §2.6's table, missing from A.6) |
| RelatedWork A.3 — quoting ShortGPT's Limitation | the original says *"XSum **and C3** deceases to nearly zero"*, but in the same paper's Table 2, **C3 is 43.56 → 39.62 / 64.55 → 56.33** — not near zero. Only XSum falls near zero, so we split the citation |
| RelatedWork §2.6 "zero competitors" — the supporting text | Gaze-Reg runs **two benchmarks on one backbone**, MoLe-VLA runs **two backbones on one benchmark** — **each has one axis.** The conclusion (nobody put both axes together with paired tests) stands, but the support must say so |
| RelatedWork §2.3 (d) "the success-rate **literature converges on 'harmful'**" | it does not converge. **Look, Focus, Act** (2507.15833) reports 94% ViT compute cut, 3× inference, and **success gains on some high-precision tasks.** The split with Gaze-Reg is **whether the token count is reduced or only pixels blurred** |
| RelatedWork §2.3 (e) "we found **no report corresponding to** +18.8" | reports that foveation helps exist. What we could not find is a gain under **our combination — token count unchanged, fixed center, no training.** Narrowed accordingly |
| Overview §① paired gap "**46.0**" | **45.9.** 46.0 comes from adding two rounded Δs; from raw counts it is (73−11)/135 = 45.93. Report was fixed but **not propagated to Overview** |
| Overview §①'s five numbers | never said what they were, so they read as success rates. Now stated as **the gap between two conditions' Δs within one cell**, with the note that deleted layers differ per cell |
| §4.4 (c) text "the three Bridge cells are 2.1, 5.2, **6.2**" | the same section's table says **6.3.** 6/96 = 6.25, exactly on the boundary — standardized to 6.3 |
| §4.3 (b) the keep 20% row | never disclosed that this row alone comes from the legacy `RetinaBased/` tree, not `results/`. The 96/96-identical baselines justify the juxtaposition, but provenance must be stated |
| Overview §② "lowering keep **barely changes the periphery**" | true only on the checkerboard test image (its periphery is already saturated). On real Bridge observations both degrade: center 1.2 → 3.5, periphery 3.3 → 7.6 |
| §3.5.2 "back-solving from two measurements **works out**" | two unknowns, two points — **anything solves it exactly**; fitting is not evidence. And the draft assumed 8 normal tokens where our `chunk_exec.py` profile measures **about 12** (3 × 4 chunks). d swings 10–19 ms with the episode you fit |
| §3.5.2 "generation runs to the 256 cap" (asserted as mechanism) | if all three hit the cap their times should match, yet they spread **1.61×** (5466/3702/3390). Token counts were not logged — downgraded to **a likely explanation** |
| RelatedWork §2.3 (b) "the two variants' **difference is the geometric-distortion share**" | does not hold. The same section's table refutes it — blur keeps the center at 100% and erases the periphery; log-polar cuts the center to 39% and keeps more periphery. **Different amounts in different places** — subtracting does not isolate geometry |
| the preamble "all numbers are generated from `results/`. **Nothing is hand-transcribed**" | §3.5.2 in the same document said "copied from the console." Values from outside the grid number **three** — §3.5.2's ms values, §3.8's right column, §2.2 (c)'s chunk numbers. The preamble now names all three and points to §7.2 |
| RelatedWork preamble "the PDF check surfaced **two things**" | only ShortGPT's internal inconsistency and "scope written too narrowly." Our errors were actually **four** (three scopes + Gaze-Reg backbones reversed + LFA conditions omitted + calibration data amount). `Report.md` §7.2 had been fixed; this one had not |
| §7.2 "human transcription is the **only** place errors occurred" · §7.1 "② is **entirely** pre-checking writing" · "Appendix A's **only** discrepancy is not ours" | all three falsified by the full read. ② has **three causes** with different detection methods — caught by source comparison, caught only by **re-deriving the recipe** (`>` vs. `>=`, channel aggregation), caught only by **reading through** (§3.8 ① contradicting ②). Appendix A had our errors too (scope, attribution). All three sentences fixed; cause table added |
| §0 ④ "foveation saves **0%** compute" | §4.3 (a) says −3.1% to +2.7% and §0 ⑥ says "≈0%." Only ④ was categorical |
| §3.8 (c) ① "our baselines are **nowhere lower. All five equal or higher**" | **wrong, and ② right below recorded the counterexample itself** — SpatialVLA/Bridge is 30.2% vs. 34.4%, **4.2 points lower.** Checked against the three backbone papers. Precisely: four higher, one at −4.2 — and the defense ("a broken setup would be consistently low") stands. Replaced with the five-cell table |
| §3.8 (a) "the OpenVLA paper and OFT both evaluate **only on real robots**" | the original has **LIBERO simulation in Appendix E.** The core claim (no SimplerEnv numbers) stands, but "only real robots" is not a fact |
| §3.8 citation keys `[30]` · `[59, re-cited 55]` | **mixed reference numbers from different papers** — `[30]` is SpatialVLA's number for OpenVLA; `[59]` is UniVLA's for SpatialVLA. Not our document's keys; replaced with paper names |
| §3.1 backbone table "Gemma2-based" · "8B" | SpatialVLA's backbone is **PaliGemma 2** (its decoder is Gemma 2), and UniVLA is **8.5B.** Layer counts 26·32·32 match the result files |
| §2 preamble "moved to RelatedWork (about **1,195 lines**)" | the header table was updated but **this one line was not.** It is 1,230 lines |
| §4.3 (b) 224 table "pixels changed by **2 or more**" | actually counted with **`> 2`** (strictly more). And within the same table, the "mean" column alone used **channel averages** while "max" used channel maxima — two rules in one table. **Unified on channel maxima** (means become 1.75→2.06 etc.) and fixed the threshold wording |
| §4.3 (b) "**2.2%** of center pixels move 10+ … **2.2% → 7.2%**" | the same `>` vs. `>=` issue. Counted as written (**10 or more**): **2.5% → 8.3%.** Also these two values are at the original **640×480** while the table above is at 224 (at 224: 0.6% → 5.5%). Both now stated |
| §3.2 "**eight** Google Robot tasks registered" | 8 is right, but the table listed 4 used + 3 unused = 7, one short. The missing one is `move_near` (non-v0), same family as our `move_near_v0`, hence not run. Added to the table |
| RelatedWork §2.5 "the direction is **monotone in capacity**" (as if for all 12) | only **EfficientVLA's four configurations** have capacity ladders, giving four ladders over two settings. All four are monotone, but the other four rows (FastV, VLA-Cache) are **single points where monotonicity cannot be asked.** Split as **direction 12/12, monotonicity 8/12** |
| §3.3·§0·§3.6·Overview "the grid runs **42** paired tests" → corrected to "**35**" | **the correction was wrong too.** 35 counts the grid body only, missing `prune8` (two cells) and `prune2+repeat2` (one). Actually run: **38**, α ≈ 0.0013. The passing eight are the same under any denominator |
| §5 "cell-to-cell comparisons: **36**" → corrected to "**42**" | **both wrong, and this is the heaviest row of the audit.** 36 was what the generator actually printed — but **the generator had a bug**: the backbone-axis loop was hardcoded to `"Bridge"`, so **Fractal's only backbone pair (OpenVLA vs. SpatialVLA) — seven tests — never ran.** 42 was hand-counted, one short (the `prune8` pair has eight). The true value after the fix: **43**, α ≈ 0.0012 |
| §5 — what that bug did to the results | **the denominator is `len(rows)`, so missing tests loosen α** — the omission made passing easier. Even so, **two of the seven missing tests pass correction** (`depth prune 4` p = 4.3 × 10⁻⁷, `action repeat 4` p = 1.4 × 10⁻⁵). The first is OpenVLA +15.6 vs. SpatialVLA −17.8 — **the only pair where both sides individually pass with opposite signs.** So §5.2's passes number **six**, not four — we had been **understating our own thesis.** No new runs; a question the same grid had never been asked |
| §7.2 "the §5 Fisher count is **36, matching the document**" (recorded as a passing machine check) | **the document and the script shared the same bug, so the comparison passed.** Machine verification verifies only when the two sources are independent — a number copied from the script proves nothing. The caveat now sits in §7.2 |
| §6.4·Overview "the original policy moved something by **3cm or more** in all 60 episodes" | the environment code is `source_obj_xy_move_dist > 0.03` — **strictly more than 3cm** (`move_near_in_scene.py`). The same type as §4.3 (b)'s "2 or more vs. `> 2`", in another section — meaning threshold sentences need an exhaustive list, which we built (`audit_claims.py` class 5, all 29 sentences checked) |
| Overview "**80 points** apart in one row" vs. "**82 points** of difference" in the same document | +12.5 vs. −69.8 is **82.3 points.** The same value rounded two ways within one document — same type as 46 vs. 45.9. Unified on 82 |
| RelatedWork A.6 "**5.6×** vs. EfficientSAM" | computing from the very table quoted above it: 78.6/13.7 = **5.7×.** A number at odds with its own table. Which row (EfficientSAM-S) and the arithmetic are now stated |
| §4.3 (b)·Overview "edges return to the **0.37 px** spot" | re-measured: **it swings 0.0–1.0 px with marker position, size, and estimator**, and a single-pixel marker can vanish entirely at the far edge. 0.37 was one setting's value, quoted without its recipe — **a violation of the rule we set in §7.1 ourselves** ("state what you measured with and against"). Lowered to "within 1 px (sub-pixel)" with conditions |
| RelatedWork §2.3 (b) "shrink to 256×256 and 21% becomes **39%**" | 39% is **INTER_LINEAR only.** INTER_AREA gives **55%**, INTER_CUBIC 30% — 25 points of swing from the interpolation choice, with no recipe stated. The direction (it rises) holds under any of them, so the range and condition are now written |
| §4.4 ★·Overview ① "the only thing changed is the `--depth-min-layer` **value**" / "we only touched **① (the window)**" | of the four conditions, **only three moved the window** (0.5 / 0.08 / 0.875); `gap3` **kept the default window and added a spacing-3 rule to the selection.** The same section's condition table had it right ("L16–31, gap 3") — the summary flattened its own table. Overview's window table also showed only two windows, making [2,4,…] read as the default window's product. Rewritten as three windows + one rule |
| Overview status table "correction log **42 entries**", "notes on **5 papers**", "discrepancies: **0**" | corrections were 66 at the time; the notes are 7 papers (Look Focus Act and Segment This Thing added); and "0 discrepancies" must be written as **"every discrepancy found in checking was fixed"** — it read as "there never were any" |
| RelatedWork A.3 "extends to non-transformers in **§4.4**" | ShortGPT's §4.4 — but our Report also has a §4.4, so it read as ours. Now "that paper's §4.4" |
| RelatedWork §2.4 (a)·A.3 "calibration is **once on both sides; the only difference is what you look at**" | re-reading the original proved this wrong. ShortGPT averages BI **across** *"a calibration set, which is a **set** of unlabelled text samples such as PG19"* — that is the `E[ ]`. We use **one forward pass of one frame**, averaging only over token positions (`_sum`/`_cnt` in `depth_prune_gemma2.py`). **Two differences, not one** (what you look at / how much) — and the smaller-sample side is **us.** Written as our open question |
| RelatedWork A.5·§2.6·A.7 Gaze-Reg's backbone arrangement | **written backwards.** The main backbone is **Pi-0**, running LIBERO's four suites + Gym-Aloha + the real robot (Tables 2·4); **OpenVLA runs LIBERO only** as a transfer check (Table 3). We had written "OpenVLA runs both benchmarks and Pi-0 is real-robot only." Corrected — and it turns out **Gaze-Reg has our grid's shape**: both axes present, crossing cell (OpenVLA × Gym-Aloha) empty. The conclusion (no crossing) survives, but as our nearest neighbor it must be written that way |
| §3.4 hardware re-run "log-polar: **three of four** tasks moved" | **all four moved** (−4.2 / +4.2 / +8.3 / −8.3). The mean stayed at 86.5% to the decimal while all four moved — so **the original claim gets stronger.** Also missing: under the same card change the baseline stayed 96/96 (only the foveation path wobbled) |
| §3.4 "**with hardware fixed**, run variance is zero" — applied to the grid | result files carry no GPU and timings cannot identify the card. **We cannot prove the five cells shared one card.** Instead the two UniVLA baselines (11/96 episodes apart, 3.1 points) **bound the size at 3.1 points**, and attaching every Δ to both baselines flips no significance (§3.4.0, new). Added to §7 ⑤ |
| the ④ table under §3.6 ③'s "the value of deleting L10" | the ③ table had been fixed per task, but **the ④ table right under it still showed one task's values.** All five rows now carry all four tasks (`prune4_back` is `[13,17,19,20]` on three of four) |
| §3.6 ① "OpenVLA/Fractal 4 layers **[17,20,23,26] → `move_near` 70.0**" | layers and task mismatched. **[17,20,23,26] is pick's set**; the set that produced `move_near`'s 70.0 is **[17,23,25,27].** The right-hand cell ([2,4,6,23]) was correct for move_near, so one row mixed two bases |
| §4.3 (b) "in the **five** images tested, periphery/center **1.5–4.8×**" | does not reproduce. Three synthetic + four Bridge observations = **seven** images, and at keep=100% the ratio is **1.5–2.8×** (1.1–2.8× over all keeps). 4.8 appears only for **a file that was already foveated fed back in.** Replaced with the stronger fact that **the direction holds in all 28 cells.** The synthetic-image range is also **19–89**, not "20–80" |
| §4.4 (c) `window25`'s compute "−11.9%" | copied from `prune4`. Measured: **−10.7%.** The two conditions delete the same layers over the same 135 episodes, so **the 1.2 points is the wobble of running the same computation twice** — footnoted as such |
| §3.6 ③ "the two layers' BI: **0.939 vs. 0.938**" | `carrot`'s values alone. Across all four tasks: 0.925–0.939 vs. 0.920–0.938 — same conclusion (they overlap), but it must be a range |
| §5.3's OpenVLA/Bridge row | only that row has log-polar from the legacy tree and blur from `results/`. Disclosed in §4.3 (b) but **not here** |
| preamble "RelatedWork **~985 lines**" · §2 "about **800**" · status "UniVLA 5 conditions **running**" | all three stale (RelatedWork was 1,195 lines then). §7.0 in the same document said "the grid is full." **Length notes go stale with every edit — recount before committing** |
| RelatedWork A.6 "Look, Focus, Act … **success rises**" (unconditioned) | unfolding Tables III/IV, it depends on conditions. The clear gains are **sim + no ViT pre-training**; with MAE pre-training Fine wins two tasks and Fov-UNet three; and **on the real robot uniform tokenization leads in three of four cells** (Ball 64 vs. 62, Toothbrush 24 vs. 18 / 18 vs. 14). Split four ways |
| RelatedWork A.7 "Look, Focus, Act: **3 real-robot tasks**" | **2 tasks** (Ball, 60 episodes; Toothbrush, 78). Sim's 6 tasks were right |
| RelatedWork §2.3 (d) "**VLA-Cache** cut FLOPs 24.5% while time rose 60.6%" | that row is **SparseVLM.** VLA-Cache is the paper that **measured and reported** it; its own method goes 51.91 → 31.83 ms. Subject corrected |
| RelatedWork §2.4 (e)·§4.4 "SpatialVLA/Bridge prune 1 is **L10** (depth 38%)" | differs per task — **two L10s, one L17, one L9.** The very error §7.1 had caught on OpenVLA `prune4`, repeated on the SpatialVLA and UniVLA rows. The conclusion (all middle-band) stands, but as a range |
| RelatedWork §2.4 (e)·§4.4 UniVLA window contrast "`[21,24,26,30]` → `[2,4,26,30]`" | one representative task. The narrow window also produces `[21,23,26,30]`, `[21,25,27,30]`, `[20,22,26,30]`. Writing it as **"all four tasks ≥ L20 → all four include L2·L4"** is both accurate and stronger |
| §4.4 ③ "ran the same one layer **chosen from the back half**" | true of three of four tasks. On `spoon_on_towel` BI picked the same L17 (already in the back half), so **the two conditions are identical there** and its 24 episodes are bit-identical. The test uses discordant pairs only, so Δ −2.1 and p = 0.8036 stand — but the contrast covering 3/4 tasks must be said |
| RelatedWork preamble "the five-paper check found **no numeric errors on our side**" | the copied values were right, but the same check surfaced **three papers' scopes written too narrowly** (Gaze-Reg, VLA-Cache, MoLe-VLA). Not "no errors" — "values right, scope descriptions wrong" |
| RelatedWork §2.1 "one cell stacks **all three axes**" | that cell (SpatialVLA/Fractal `prune2_repeat2`) stacks **two.** No grid condition stacks three |
| RelatedWork §2.3 (b) Look, Focus, Act "**324 → 20 patches**" | judged absent from the original based on search results and deleted it — **opening the PDF, it is right there in Table II and §5 A** (Fine 324 / Foveated 20). Restored. **Trying to refute a primary source with a secondary one** was the mistake |
| RelatedWork §2.3 (d)·A.5 "the two Gaze-Reg conditions work **in our favor** → our version has even less reason to rise" | the same sentence in two places. ① Not "in our favor" — **conditions favoring foveation.** ② More importantly, that conclusion **collides head-on with our +18.8, and the collision went unmentioned.** Now: "our result is not explained by this paper" |
| §7.1's own tallies ("② has **42**", "wrong numbers: **three**", "the other **twenty**") | the table had grown to 44 rows and the paragraphs below never followed; the type table summed to 16. **Reclassified all rows** so the sums match, and noted that classification involves judgment |
| RelatedWork §2.3 (b) "the measured curve: **even dead center only 30–53%**" | reproduces under its own conditions, but **the metric was never named** — same image, same keep: Laplacian 39%, Sobel 68%. It read like an intrinsic property of the transform |
| Overview status table "baseline … **nowhere lower than the papers**" | **Report §3.8 was fixed without propagating to Overview.** The same error lived in both documents; one was fixed, and for days Overview carried an already-refuted sentence. Now "four higher, one −4.2." **There was no procedure of grepping the other documents when fixing an error** — the cross-document version of the repeated-across-backbones hole §7.1 had already recorded |
| Overview §③ "changing the backbone gives four, **three at p < 10⁻⁵**" | the four p-values are 1.736 × 10⁻¹³ · 1.376 × 10⁻⁶ · 1.561 × 10⁻⁵ · 3.55 × 10⁻⁴. **Two are below 10⁻⁵; three are below 10⁻⁴.** Report §5.2 had said "p < 0.0001" correctly all along — only Overview slid the exponent |
| §1·§3.1·Overview "UniVLA takes **3 seconds** per step" (Overview) · "**~2.9 s**" (Report) | measured: `avg_model_ms_per_infer` = 2811.5 ms = **2.81 s.** The two documents carried different roundings, both above the measurement. Unified at three digits with the field name |

② has **three causes, not one**, and each is caught differently.

| Cause | How it gets caught | Example |
|---|---|---|
| **writing before checking the source or code** | going back to the source/code catches it | calibration data amount; Gaze-Reg backbones |
| **the number is right but its label disagrees with the computation** | only **re-deriving the recipe** catches it | "2 or more" over an actual `> 2`; two channel rules in one table |
| **the document contradicts itself** | only **reading it through** catches it | §3.8 ① said "nowhere lower" while ② recorded the counterexample |

**The draft listed only the first cause.** So "we checked the originals"
can never be grounds for closure — the second needs the scripts re-run, and
the third needs reading. If another ② appears, it gets added here.

---

## 7.2 Machine verification — did this document's numbers come from the records?

Every hand-carried number was **checked against the records.** We first
believed human transcription was the only place errors could occur — **that
belief was wrong**: two of the three causes above have nothing to do with
transcription. So the table below records not "how many mismatches" but
**what was verified, and with what.**

| What was verified | With what | Status |
|---|---|---|
| the grid's 35 cells — Δ, p, broke/fixed | recomputed from `results/` episode records | **match** |
| compute savings % (all conditions) | `model_stats` in the result files | **match** |
| §5's 15 Fisher p-values (§5.1: 6 + §5.2: 9) | recomputed from the discordant splits | **match** |
| the two test-family sizes | enumerated the conditions directly | **were wrong → corrected to 38 / 43** |
| keep sweep · window sweep | recomputed | **match** |
| all §6 failure-bucket tables | ran `mechanism_move_near.py` | **match** |
| deleted layer sets (all conditions, all tasks) | the layer fields in the result files | **match** (single-task notation corrected to ranges) |
| four bit-identity claims | episode/step/grasp comparison | **match** |
| foveation pixel measurements | re-ran the measurement script + traced recipes | **threshold wording corrected** |
| code citations (formulas, flags, line numbers) | compared against source | **match** |
| §3.8 **left column** (our baselines) | recomputed | **match** |
| §3.8 **right column** (authors' values) | checked against the SpatialVLA, UniVLA, OpenVLA PDFs | **match** ("nowhere lower" corrected) |
| Appendix A quoted values (**7** PDFs + 3 backbone papers) | direct table/sentence comparison | **match** (the papers' own 3 inconsistencies separate) |
| background citations (arXiv IDs, bibliography) | search comparison | **match** |
| internal tallies, cross-references, lengths | script | **match** |
| universal sentences ("all/only/nowhere") and threshold sentences | `audit_claims.py` **enumerates exhaustively** (counts recounted each run), each traced to evidence | **pass** (one 3cm wording corrected) |
| **all of the above at once** | **`verify_all.py`** — every machine check in this table (record integrity → grid → test families → costs → control runs → mechanism → pixel measurements → document structure → external records → citation arithmetic → sub-suites) in **one command**, printing a failure list. The campaign's lesson is that verification arriving in installments can never be known to be finished, so the entry point is one | **all 189 checks pass** |
| **execution environment (GPU)** | **no record exists** | **impossible; §7 ⑤** |

> **This table is the document's verification boundary.** What could *not*
> be verified is now **only the last row — the GPU record** — absent from
> the data and unrecoverable after the fact (its size is bounded at 3.1
> points in §3.4.0). Everything else was matched by recomputation or
> source comparison. **Whenever we write "verified," this row is never
> omitted.**

**More than 140 numbers match.** §5's Fisher count is 43, matching the
document (while this line said 36, the generator printed 36 too — **the two
shared one bug, so the comparison passed.** When script and document come
from the same source, agreement is not evidence of correctness). The three
determinism re-runs (SpatialVLA/Fractal 85/85, UniVLA 24/24, §6's 60/60 on
both arms) matched down to **step counts**, and the 255 files and 7,198
episodes were recounted and confirmed.

> **One number verification cannot reach.** §3.5.2's three ms values
> (5466/3702/3390) have no result files — the run was aborted and they were
> copied from the console. The baseline/prune values in the same section
> were recomputed from records. **The exception is noted in §3.5.2's text
> as well.**

**Every table value copied in Appendix A matched the originals.** Our
errors came not from values but from **scope and attribution** — three
papers' scopes written too narrowly, Gaze-Reg's backbones reversed,
Look/Focus/Act's gains quoted without conditions (all in §7.1 ②). And the
papers themselves carry three inconsistencies (ShortGPT's introduction vs.
Table 2: MMLU 55.0 → 52.2 vs. 55.00 → 54.69; Table 2 vs. Table 6 on BoolQ;
and the Limitation's "XSum and C3"). The values proper — EfficientVLA
Table 2's 12 settings and eight Δs (PickCan +4.0/+3.4/+2.7/+2.0, MoveNear
−1.7/−2.6/−2.9/−3.7) exact; VLA-Cache Table 3's eight cells identical to
EfficientVLA's citation row to the decimal, **confirming the
independent-source judgment**; MoLe Table 5's `STAR alone 56.3% < baseline
57.2%`; Gaze-Reg Table 11's **10/10 decline and 85.9 → 78.5** — all stood.

**So the remaining risk is sentences, not numbers.** Of ②'s 98 rows,
**twelve are mistranscribed numbers** (−70.8 → −69.8, −77.1 → −79.2,
46.0 → 45.9, 6.2 → 6.3, the −10.4/−17.8 cell mix, "three → two" tasks,
layers-vs-task mismatch, `window25`'s copied saving, the exponent
10⁻⁵ → 10⁻⁴, UniVLA latency 2.9/3 s → 2.81 s, 80 → 82, 5.6× → 5.7×) —
mostly rounding or adjacent-cell slips. **The other eighty-six are
sentences that got ahead of the data**, falling into these types:

> **Do not confuse tables ① and ②.** ① (17 rows) is **sentences changed
> by new measurements** — that is an experiment in progress, not error. ②
> (98 rows) is **us getting ahead of the data.** In the draft, six rows that
> belonged in ② were mixed into ①.

| Type | Count | Examples |
|---|---:|---|
| **writing a one-condition measurement as a general property** | 24 | foveation's periphery behavior (checkerboard image), the detail curve (Laplacian), §3.5.2's back-solve, "difference = geometry share", **one task's values as a whole condition (five)**, LFA's gains without conditions, extending determinism from two columns to the grid's hardware |
| **writing without reading the code or the original** | 26 | calibration data amount, the meaning of `keep`, MoLe's router, FLOPs 28.9%, four scopes of others' experiments, **Gaze-Reg's backbones reversed**, picking the wrong row of someone's table, back-solving a number and writing it as the paper's, **a range our own script does not reproduce (1.5–4.8×)** |
| **not dividing by the denominator / not counting the family / not stating provenance / not propagating across documents** | 32 | §6.3's bucket shares, **the two family sizes swapped (35 ↔ 42)**, `**` → `***`, "six papers", "three axes", §5.3's tree mixing, lengths/status blocks/status tables, another paper's section number as ours, §7.1's own tallies, **a fixed baseline sentence left unfixed in Overview** |
| **mistranscribed numbers** | 12 | the twelve above |
| **the verification script itself wrong** | 2 | the backbone-axis loop hardcoded to Bridge, dropping seven tests; document and script sharing one bug so the machine comparison passed |
| **sentences surviving from before the result** | 2 | §2.5's "the basis of §6", and the Gaze-Reg passage written as if unaware of our +18.8 |

> **The same type repeated three times.** "Writing one task's deleted
> layers as the whole condition's" had already been caught once on OpenVLA
> `prune4` — and was repeated verbatim on the SpatialVLA and UniVLA rows.
> **There was no procedure for checking whether a caught error also lives
> in the other backbone rows** — that is what this table teaches.

> **And the same hole has a cross-document version.** Report §3.8's
> "nowhere lower" was fixed without fixing the same sentence in
> `Overview.md`'s status table. 46.0 → 45.9 and the UniVLA latency have the
> same shape — **a sentence fixed in one document survived in another.** So
> one more rule: **when a value or sentence is fixed, grep all three
> documents for the same value or phrasing before committing.** The last
> three rows of this table are the ones caught late for lack of that
> procedure.

> Type assignment involves judgment, and a few rows sit on boundaries. But
> **pushing them either way changes nothing** — the two largest types are
> "writing without looking" and "generalizing one condition," and pure
> transcription slips are a minority.

**"Writing without looking," the denominators, and the stale sentences all
arose at transcription points, not in computation.** Everything new from
sweeping §2 and Appendix A was the same — §2.5 and A.3 still held pre-
rejection sentences after §6 was rejected, and our calibration procedure
had been written from memory instead of code. This type **is always caught
by going back to the source or the code.**

**"Generalizing one condition" is different in kind, and more dangerous.**
Transcription errors are caught by comparison with records; this is the
case where **the number is right and the sentence is wrong.** The detail
curve reproduces exactly under its own conditions, and §3.5.2's back-solve
is arithmetically sound — what was wrong was **not stating what the value
depends on.** Switch the metric from Laplacian to Sobel and 39% becomes
68%; switch the test image from a checkerboard to a real observation and
the periphery conclusion reverses.

> **Hence one more rule.** When quoting a measurement, **state what it was
> measured with and measured against.** Not "39% at dead center" but "39%
> by Laplacian (a metric sensitive to the finest texture) on a 640×480
> Bridge observation." And **when the number of unknowns equals the number
> of measurement points, the solution is not a verification** — it must be
> labeled as such.

**And one rule of this document, fixed here.** When quoting totals,
**write the denominator with them.** Not "16 → 35" but "16 of 23 failures
→ 35 of 48 failures." Using counts where shares are needed makes the mere
growth of failures read like an effect — the same error we criticize in
§2.5.

---

## 7.3 Where this document goes in the paper

**This document is not the paper — it is the paper's material.** Here is
which section goes where, and **what the paper needs that we do not yet
have.**

### The mapping

| Paper | What it takes from here | Status |
|---|---|---|
| **Abstract** | §0's one sentence + the five results | **not written** |
| **1. Introduction** | all of §1 (the field's claim → the premise we test) | draft exists |
| **2. Related Work** | §2.1–2.6; material in Appendix A | draft exists |
| **3. Method** | **§3.0** (intervention specs) · **§3.3** (pairing protocol) · **§3.4 + 3.4.1** (determinism and the p-value's premises) · §3.6 (grid uniformity rules) | ready |
| **4. Experimental Setup** | §3.1 backbones · §3.2 benchmarks · §3.5 deleted layers · §3.8 baseline comparison · §3.7 reproduction | ready |
| **5. Results** | §4 (per axis) · §5 (across axes) | ready |
| **6. Analysis** | §6 (mechanism — negative result) · §4.4 (c) (selection sensitivity) | ready |
| **7. Limitations** | §7.0 status · §7 ①②③④ | ready |
| **8. Conclusion** | — | **not written** |
| **Appendix** | Appendix A · §3.5.1–3.5.2 · §7.1 · §7.2 | ready |

### Do not confuse what the Method is

**We propose no new method.** So what goes into the Method section is not
an algorithm but **a measurement procedure.** Three parts:

1. **Per-episode pairing** (§3.3) — pair identical initial states and
   count only discordant pairs, instead of subtracting two success rates.
   Impossible to apply to prior work (no episode records released), which
   is itself one of our points.
2. **The determinism check and the statistical reading that follows**
   (§3.4, §3.4.1) — re-run variance is confirmed zero, so the p-value, not
   a confidence interval, is the entire uncertainty. The premise is
   stated.
3. **The grid uniformity rules** (§3.6) — what is a grid cell, what is a
   diagnostic, and why diagnostics need not be uniform.

These three are the Method; **§3.0's intervention specs are what that
Method is applied to.**

### Results has three tiers

| Tier | What | Where |
|---|---|---|
| **the grid** | 5 columns × 8 conditions, paired Δ and p | §4 |
| **tests across axes** | the 43 Fisher tests — does the same intervention act differently per cell | §5.1–5.3 |
| **digging into one cell** | the keep sweep (4 points), the depth window sweep (5 cells), the mechanism measurement | §4.3 (b), §4.4 (c), §6 |

The third tier is **likely to carry the paper's weight.** The grid shows
"the sign splits" but controls few variables; the third tier **fixes
backbone, benchmark, method, capacity, and cost, moves one flag**, and
produces 45.9–50.4 points.

### What the paper needs and we lack

| Missing | Note |
|---|---|
| **Abstract / Conclusion** | the results are settled, so these can now be written |
| **Figures** | tables only so far. At minimum: (a) a grid heatmap, (b) the keep dose-response curve, (c) a five-cell paired-gap bar chart |
| **A third benchmark** | §7 ① — not openable by simulation |
| **The drawer task family** | §7 ③ — openable by simulation; about 6 hours at diagnostic scope |
| **The paper version of the correction log** | §7.1's table does not go in as is. Only §7.2's **three causes and the "totals with denominators" rule** go into the methodology section, as one paragraph |

---
