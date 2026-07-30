# LIBERO Experiments — Are Test-Time Interventions Architecture-Dependent?

Purpose: to report that the results obtained on SimplerEnv do not reproduce on
LIBERO, and to ask for advice on where to take the work next.

> **Before reading.** Every condition is 50 episodes. At that size,
> **differences smaller than roughly 18 points cannot be trusted** (§6.5).
> Where this report says "no difference", please read it as **"not visible at
> this sample size"** rather than "proven to be zero". Only the large values
> (−46, −68, −74) are certain regardless of sample size.

---

## 1. Summary

On a single suite, `libero_spatial`, **three kinds of test-time intervention
acted in opposite directions on the two backbones**. And **foveation, which
helped on SimplerEnv, became a loss on LIBERO for both backbones**.

There are three things I would like to ask about:

1. **How to read this inconsistency** (§5)
2. **Which direction to take the research from here** (§7.2) — what I have is
   the observation that training-free, plug-in interventions behave
   differently across backbones and benchmarks, and I am not sure what story
   to build from it.
3. **Whether you could look at the foveation code** (§7.4) — the LIBERO result
   is extreme enough that I cannot rule out having missed something when
   porting it.

---

## 2. Setup and Terminology

### 2.1 Setup

- Benchmark: `libero_spatial`, 10 tasks × 5 initial states = **50 episodes per
  condition**
- Backbones: `openvla-7b-finetuned-libero-spatial`,
  `UniVLA (Emu3) / UNIVLA_LIBERO_IMG_BS192_8K`
- Both backbones share the **same evaluation code**. There is one loop that
  steps the simulator, feeds the observation to the policy, and applies the
  returned action, and each intervention is applied identically — pixel for
  pixel, rule for rule — to both.
- Rendering: OSMesa (CPU). Episodes end as soon as LIBERO reports success.
- Every condition replays the **same 50 initial states** (see §2.4)

### 2.2 The three interventions — each touches a different resource

By **intervention** I mean changing the input or the computation at inference
time, without retraining. Every experiment in this report is of that kind; the
model weights are never touched.

| intervention | number of model calls | compute per call | information the model sees |
|---|---|---|---|
| action-repeat 2 | **1/2** | unchanged | unchanged |
| foveation (keep 20%) | unchanged | unchanged | **1/5** |
| depth-prune (bypassing layers) | unchanged | **3/4** | unchanged |

#### (a) Temporal axis — `action-repeat 2`, and how it differs from `chunk-exec`

`action-repeat 2` calls the model **once** and feeds the single returned action
to the simulator **twice in a row**. The number of model calls halves, so
latency halves.

`chunk-exec k` is a different thing. Some policies (UniVLA, SpatialVLA) return
**several future actions at once** per call. Executing only the first k of them
and then calling again is chunk-exec.

| | action-repeat 2 | chunk-exec 2 |
|---|---|---|
| what gets executed | the **same action, copied** | **two different actions** the model actually predicted |
| requirement | none | the policy must emit multiple actions |
| information lost | yes (motion becomes stepwise) | comparatively little |

OpenVLA emits **exactly one action per call**, so chunk-exec does not even
apply to it — there is no chunk to truncate. Action-repeat is therefore the
only way to reduce its model calls, which is why the LIBERO table uses
action-repeat for both backbones.

#### (b) Visual axis — `foveation`, and how blur differs from log-polar

Foveation imitates the human eye: sharp at the centre of gaze, blurry in the
periphery. Applied to an image, it keeps the centre and discards information
elsewhere. `keep 20%` means the effective sampling density is reduced to 20% of
the original.

| | blur | log-polar |
|---|---|---|
| what it does | progressively **blurs** with distance from the centre | **resamples**: dense at the centre, sparse outward |
| pixel positions | **unchanged** | **moved** (the periphery is pulled inward) |
| what is lost | peripheral sharpness | peripheral spatial resolution |

In both, the **image size (pixel count) is unchanged**. The model therefore
processes the same number of visual tokens and **latency does not drop**
(§3.1). Foveation is an accuracy technique, not a speed technique.

The distinction mattered for SpatialVLA. It attaches an explicit **3D
position**, computed from the pixel coordinates, to each visual token.
Log-polar physically moves pixels, so after warping every token carries a wrong
3D position. Blur, which does not move pixels, does not have this problem.

#### (c) Depth axis — `depth-prune` and `depth-ctrl`

`depth-prune N`: both backbones have **32 LLM decoder layers**. N of them are
skipped entirely (no computation; the input is passed through to the next
layer). This removes N/32 of the compute, and it is **the only axis on which
latency actually moved**.

Which N layers are chosen:

1. **Once, on the first step of the episode**, measure how similar each layer's
   input and output are, by cosine similarity. A small `1 − cos(in, out)` means
   the layer barely changes the hidden state — it is **idle**.
2. Take the N smallest. Two safeguards: the **first half of the stack is
   protected** (early layers always matter), and the selected layers are spaced
   out so they do not cluster consecutively.
3. The measurement rides on the first inference that has to run anyway, so it
   costs **no extra forward pass**.

Both backbones make this choice with the **same code** (`depth_prune.py`),
since a claim that backbones differ only means something if the selection rule
does not.

`depth-ctrl (2→8)`: change the number of bypassed layers **during the episode**.

```
     reaching for the object · grasping        lifting and transporting
  ├────────────────────────────────────┼────────────────────────────────┤
        bypass only 2 (deep = accurate)      bypass 8 (shallow = fast)
                                       ↑
                     the moment the policy closes its own gripper
```

The switching signal is **the gripper command the policy is already emitting**,
so it costs nothing extra. Once shallow, it never goes back, so there is no
oscillation. The 2 bypassed layers are a subset of the 8, so the stack stays
consistent across the switch.

### 2.3 Other terms

| term | meaning |
|---|---|
| **oracle gaze** | Reading the target object's true 3D coordinates **from inside the simulator** and placing the fovea there. It amounts to knowing the answer in advance, so it cannot be used on a real robot. It is not a method but a **diagnostic that measures the ceiling: how well could a perfect gaze possibly do?** |
| **BDDL** | The file format LIBERO uses to define a task. Besides the initial layout and success condition, its `(:obj_of_interest ...)` field states **which object the task is about**. That is where oracle gaze reads its target from. |
| **prefill / decode** | The two stages of LLM inference: prefill processes the whole input at once, decode emits output tokens one at a time. For UniVLA, decode is 70% of the time. |
| **z** | How large a difference between two success rates is relative to chance. Roughly, \|z\| > 2 means "hard to explain as chance" (p < 0.05). |
| **Δ (delta)** | Change in success rate against baseline, in percentage points. |

### 2.4 Every condition uses the same initial states

The evaluation code is deterministic — rerunning the same setting gives the
same result — and every condition replays the **same 50 initial states**. So
when two conditions are compared, they did not solve different problems; they
each solved **the same 50 problems**, which means episodes can be matched up
one by one. This matters for the statistics in §6.5.

---

## 3. Results

| condition | OpenVLA | Δ | ms (speedup) | UniVLA | Δ | ms (speedup) |
|---|---|---|---|---|---|---|
| baseline | 74.0% | — | 524 (1.00×) | 96.0% | — | 1882 (1.00×) |
| action-repeat 2 | 66.0% | −8.0 | **262 (2.00×)** ¹ | **28.0%** | **−68.0** | **941 (2.00×)** ¹ |
| foveate blur 20% | 58.0% | −16.0 | 518 (1.01×) | 94.0% | −2.0 | 1888 (1.00×) |
| foveate log-polar 20% | **0.0%** | **−74.0** | 518 (1.01×) | 88.0% | −8.0 | 1886 (1.00×) |
| depth-prune 4 | 56.0% | −18.0 | 463 (1.13×) | 86.0% | −10.0 | 1750 (1.08×) |
| depth-prune 8 | **28.0%** | **−46.0** | 403 (1.30×) | 86.0% | −10.0 | 1457 (1.29×) |
| **depth-ctrl (2→8)** | 50.0% | −24.0 | 445 (1.18×) | **96.0%** | **0.0** | **1667 (1.13×)** |
| blur + depth-ctrl | — | — | — | 88.0% | −8.0 | 1658 (1.14×) |
| log-polar + depth-ctrl | — | — | — | 84.0% | −12.0 | 1653 (1.14×) |

ms is the time for **one model call**. Two caveats when reading that column:

¹ **Action-repeat is the one row of a different kind.** The cost of a single
call is unchanged (524 / 1882 ms); what halves is the **number of calls**.
Since the same episode finishes in half the calls, the cell above reports the
figure **amortized over the episode** rather than the per-call cost. In every
other row the call itself became cheaper, so the two numbers coincide.

² **The two backbones' ms should not be compared with each other.** UniVLA
emits about 10 actions per call and executes all of them, so per environment
step it is **about 188 ms**, not 1882. OpenVLA emits one, so its 524 ms *is*
the per-step figure. **UniVLA is in fact the faster policy per step.** The
speedup column is measured within each backbone against its own baseline, so it
is unaffected.

**What the table shows at a glance:** only one cell moved wall-clock without
paying accuracy — **UniVLA's depth-ctrl** (96.0% at 1.13×). Every foveation row
sits at 1.00–1.01×, and action-repeat, the largest speedup at 2.00×, is the one
that costs UniVLA 68 points.

The two backbones split on all three axes.

| axis | UniVLA | OpenVLA |
|---|---|---|
| temporal (number of calls) | ✗ −68 (certain) | ✓ −8 (indistinguishable) |
| visual (information) | ✓ −2 / −8 (indistinguishable) | ✗ −16 (weak) / −74 (certain) |
| depth (compute per call) | ✓ −10 (weak) | ✗ −46 (certain) |

**Only the cells marked "certain" can be trusted independently of sample size.**
That said, the **pattern itself** — the sign reversing on every axis — is hard
to attribute to chance even where individual cells are not significant
(six coin flips landing the same way is unlikely).

### 3.1 Latency moved only on the depth axis

Foveation **reduced latency in none of the three conditions**
(UniVLA 1882→1888 ms, OpenVLA 524→518 ms), because the image size is unchanged
and so is the number of visual tokens. Profiling a UniVLA step gives
**6% VQ encoding / 13% prefill / 70% autoregressive decode**, so the entire
visual path has a ceiling of about 19% no matter how hard it is attacked.

That is why the work moved to **bypassing layers**, which attacks decode
directly — and it is where wall-clock first moved.

### 3.2 The shape of the depth-accuracy curve differs by backbone

| layers bypassed | OpenVLA (Llama-2, 32 layers) | UniVLA (Emu3, 32 layers) |
|---|---|---|
| 0 | 74.0% | 96.0% |
| 4 | 56.0% (1.13×) | 86.0% (1.08×) |
| 8 | 28.0% (1.30×) | 86.0% (1.29×) |

UniVLA is **flat past the first cut**; OpenVLA **keeps falling** at roughly
4.5–7 points per layer. I read this as Emu3 having genuinely idle layers and
Llama-2 not.

That reading rests on **three points**, though. Intermediate settings (12, 16)
were not measured, so how far the flat region extends is unknown. Whether
UniVLA landing on 86% twice reflects a truly flat curve or two coincidences
also cannot be fully separated at n=50 — although the exact agreement is
evidence for flatness.

For reference, SpatialVLA (Gemma2) lost accuracy on 3 of 4 tasks with a
**single** layer bypassed, in earlier experiments. Lined up, the three
backbones order as **Emu3 ≫ Llama-2 > Gemma2**.

### 3.3 The phase-adaptive controller — the only condition that gained

Evaluated as **how far above the static curve it sits at matched latency**:

| | static value at that latency | controller | gain |
|---|---|---|---|
| **UniVLA** (1667 ms) | 86.0% | **96.0%** | **+10.0** |
| OpenVLA (445 ms) | 47.6% | 50.0% | +2.4 |

On UniVLA it ran **1.13× faster with no detectable loss of accuracy.** To be
honest about it, though, **the +10.0 gain itself does not reach conventional
significance** (z≈1.8, p≈0.08 treating the two runs as independent). Matching
episodes pairwise would likely strengthen it, but that calculation has not been
done yet (§6.5). For now the right way to report this is **promising but not
established**.

OpenVLA's +2.4 is indistinguishable from any other point on its curve.

Explained through the curve shapes in §3.2: for UniVLA, going from 4 to 8
bypassed layers is essentially free, so buying depth back only during the grasp
is a net gain; for OpenVLA every layer charges for itself, so mixing two
settings just produces a weighted average. The hypothesis is that
**non-uniform allocation wins only when the curve has a flat region**.

This hypothesis was **recorded as a prediction before the run and then
checked**. From the curve shapes I wrote down that the OpenVLA controller would
land at 40–55% and 1.15–1.20× and would not beat static; it measured 50.0% at
1.18×. A correct prediction does not prove the hypothesis (the interval was
wide), but it does establish that this is not a post-hoc reading.

### 3.4 Applying two interventions together

This question can only be asked on UniVLA, since it is the only place where
both are individually near-free (blur −2.0, controller 0.0).

**How to read the table.** Blur alone costs −2.0; the controller alone costs
0.0. If the two interventions had nothing to do with each other, running them
together should cost just the **sum** of those (−2.0 + 0.0 = −2.0). That is the
"expected if independent" column. What actually happened was −8.0. **That is 6
points worse than expected, and those 6 points are the extra cost of running
both at once.**

| combination | expected if independent | measured together | extra cost of combining |
|---|---|---|---|
| blur + ctrl | −2.0 | **−8.0** (88.0%) | **−6.0** |
| log-polar + ctrl | −8.0 | **−12.0** (84.0%) | **−4.0** |

Two different foveation modes gave an extra cost in the same direction and of
similar size. **But 4–6 points is, by the standard in §6.5, indistinguishable
from chance.** The only support is that it appeared twice in the same
direction, and even that is not a fully independent replication since both
conditions share the controller. **"The costs of two interventions probably
should not be assumed to simply add"** is as much as the data currently
supports.

> UniVLA **holds 84–88% while discarding 80% of its visual information and 25%
> of its decoder layers at the same time, at 1.14×.**

That sentence is read directly off the table. It is 8–12 points down from the
96.0% baseline, so it is not lossless, and those 8–12 points are — for the same
reason as above — not distinguishable from chance.

OpenVLA has no counterpart: foveation alone costs it −16 and depth-8 alone −46,
so there is no combination worth discussing.

---

## 4. Checking alternative explanations

I worked through the possibility that these results come from the setup rather
than from the interventions. **The strength of the evidence varies a great deal
between items**, so it is stated in the last column.

| suspicion | how it was checked | observation | strength |
|---|---|---|---|
| UniVLA is robust to foveation because the wrist camera acts as a backup | degrade **both** cameras | at most 8 points of difference | **weak** — 8 points is indistinguishable from noise at n=50; supports only "no large effect" |
| UniVLA is not using the image at all | both cameras blanked to black | **0/50** | **very strong** — 96% → 0% |
| VQ quantization absorbs the distortion | measure the visual-token change rate directly | 99.4% of tokens change under log-polar | **strong** — measures an internal quantity, not success rate |
| a defect in the FAST action tokenizer | port the authors' patch, rerun every condition | 4.5% → **0.0%**, conclusions unchanged | **strong** — cause removed and rerun |
| the fovea was simply in the wrong place | **oracle gaze** (below) | blur 58% → 50% | **moderate** — see below |
| the cut was simply too aggressive | depth-prune 4 | UniVLA is 86% at both 4 and 8 | **moderate** — two points agree |
| the bypass was implemented differently per backbone | one shared module; two measurement paths agree to zero error | identical by construction | **strong** — guaranteed at code level |

### 4.1 Oracle gaze — the control I spent the most effort on

I read the target object's 3D coordinates from the simulator, projected them
through the camera matrix, and placed the fovea **directly on the target**.

#### First, I checked visually that the gaze really lands on the target

All 10 tasks were rendered. **Left** = the raw frame the policy receives, with
the oracle's computed fovea position marked by a magenta crosshair. **Right** =
what the policy **actually receives** once foveation (keep 20%) is applied
centred on that position.

![oracle gaze, tasks 0-2](figures/oracle_gaze_tasks0-2.png)

The crosshair lands in a **different place every time**. It is not marking a
fixed point; it is following the object the instruction names in each task. In
the right column, only the region around the crosshair stays sharp — that is
the entirety of what the policy sees.

![oracle gaze, tasks 3-5](figures/oracle_gaze_tasks3-5.png)

![oracle gaze, tasks 6-9](figures/oracle_gaze_tasks6-9.png)

**And here the problem becomes visible.** Only a few tasks have their target
near the centre of the frame; many sit at the **edge** — far left, or on top of
the cabinet at the right. But centring the fovea there **destroys the
destination and the gripper instead.** The right column shows what happens on
the far side of the crosshair.

In other words, `libero_spatial` requires the policy to see **the target bowl,
the destination, and the gripper at once**, and there is only one sharp disc,
which cannot cover all three. Centred, it covers all three poorly; moved onto
the target, it gives up the other two. **There is no good place to put it** —
which appears to be why the numbers below come out as they do.

(The figures are rendered at initial state 0. The script warns automatically if
a crosshair falls outside the frame or pins to an edge; all 10 tasks were
clean.)

#### Results

| OpenVLA, keep=20% | fovea fixed at image centre | fovea on the target |
|---|---|---|
| log-polar | 0.0% | 22.0% |
| **blur** | **58.0%** | **50.0%** |
| (no intervention) | 74.0% | |

**The blur row is the one that matters.** Blur does not move pixels, so how
much it degrades is independent of where the fovea sits — which makes it the
row that directly answers "is placement the cause?". No improvement appeared;
it was 8 points lower.

**This needs a careful reading, though.** 50% vs 58% is z=−0.81, which is
evidence of **"not detected"** rather than "no improvement". At n=50, a real
improvement of up to roughly +11 points would still be consistent with this
data. So what can be said with confidence is **"gaze placement is unlikely to
be the main cause of the −16 point loss"**, not "gaze is irrelevant".

Log-polar's 0→22 is best read as escaping a pathological configuration: the
transform concentrates samples near the pole, and with the pole on empty table
that budget was wasted. It still falls short of blur's 58%.

The projected coordinates were cross-checked two ways (reproducing the render
pipeline, and robosuite's own projection), and the choice of target object was
verified against the `obj_of_interest` field of all 10 BDDL files. The figures
above are one further check by eye.

**Tentative conclusion: with any fovea placement I tried, OpenVLA goes from 74%
to at best 58%.** On that basis I set aside the plan to build an
instruction-conditioned gaze from the model's own attention, judging its
ceiling to be low. That judgement stands on the statistical caveat above.

---

## 5. What worries me most — the inconsistency with SimplerEnv

### 5.1 Earlier results on SimplerEnv (for reference)

SimplerEnv WidowX-Bridge, 4 tasks × N=24. RoboVLMs is excluded.

| intervention | OpenVLA | SpatialVLA | UniVLA |
|---|---|---|---|
| baseline | 15.6% | 32.3% | 78.1% |
| chunk-exec | n/a ¹ | **45.9% (+13.6, 1.9× faster)** ✓ | 65.6% (−12.5) ✗ |
| foveate log-polar 20% | **34.4% (+18.8)** ✓ | 25.0% (−7.3) ✗ | **86.5% (+8.3)** ✓ |
| foveate blur 20% | **33.3% (+17.7)** ✓ | ² | 76.0% (−2.1) △ |

¹ OpenVLA predicts one action per call, so "execute part of a predicted chunk"
does not apply (§2.2a).
² SpatialVLA's blur was only ever measured in combination with chunk-exec
(36.5%), so there is no value against baseline alone. What is established is
that it recovered most of the log-polar loss (27.1% → 38.6%). The script for
the standalone measurement is ready but **has not been run yet for lack of
compute** (§7.1). It is a 2–3 hour gap to fill.

Here too, **no intervention worked across all three backbones.** Even the
variant of an intervention (log-polar vs blur) had a different winner per
backbone.

SpatialVLA's log-polar failure is explained (§2.2b), and the fact that blur —
which does not move pixels — recovered the loss supports that explanation.

### 5.2 But the sign flips on LIBERO

The same foveation code has opposite signs on the two benchmarks.

| backbone | SimplerEnv Bridge | LIBERO spatial |
|---|---|---|
| OpenVLA | **+18.8pp** (15.6 → 34.4) | **−74pp** (74.0 → 0.0) |
| UniVLA | **+8.3pp** (78.1 → 86.5) | **−8pp** (96.0 → 88.0) |

What has been narrowed down so far:

- **It does not look like a backbone issue.** The two backbones that gained on
  SimplerEnv **both** lost on LIBERO. The direction is the same.
- **Policy strength alone does not explain it.** The hypothesis that "a weak
  policy benefits from having the background cleaned up" fails to explain
  UniVLA gaining +8.3 on SimplerEnv from an already strong 78.1% baseline.
- **Gaze placement is unlikely to be the main cause** (§4.1, with the caveat
  stated there).

**The most plausible remaining candidate is scene composition.** This is still
an **unverified hypothesis**, though, and the table below lists observed
differences rather than demonstrating a cause.

| | SimplerEnv Bridge | libero_spatial |
|---|---|---|
| target | single, near the centre | **two identical-looking bowls**, at the edge |
| how to tell them apart | there is only one object | **spatial language only** ("between the plate and the ramekin") |
| background | photographic, cluttered | clean synthetic table |

It looks as though foveation removes clutter in one setting and removes signal
in the other. Testing this directly would require **a pair that differs only in
policy strength within one benchmark**, but the general-purpose `openvla-7b`
has no LIBERO action statistics, so whichever `unnorm_key` is chosen mixes
"weak policy" with "wrong action scale". The current setup cannot separate the
two, so this is left open.

---

## 6. Checking the setup itself

### 6.1 What was verified

- Whether the intervention flags actually reached the model is audited
  automatically from the summary JSONs (`verify_runs.py`: episode count, task
  count, number of bypassed layers, controller transitions, decode failure
  rate, duplicated conditions). The audit itself was validated against 7
  deliberately broken cases; it caught all 7.
- The layer-bypass bookkeeping passes 27 checks on CPU. In particular,
  **a cached `generate()` with layers bypassed is exactly identical to an
  uncached greedy decode** — if this were wrong, the KV cache would break
  silently and success rate alone would never reveal it.
- The layer selection rule is identical across backbones by code sharing, and
  the redundancy measurement agrees to zero error between two independent paths
  (direct forward vs forward hook).
- The foveation code is **bit-identical** to the original used in the SimplerEnv
  experiments, verified by 29 checks (3 image sizes × 4 keep ratios).

### 6.2 Reproducibility

The OpenVLA baseline was rerun independently with **10 initial states (100
episodes)**.

| | success |
|---|---|
| initial states 0–4 (rerun) | **37/50 = 74.0%** |
| initial states 5–9 (rerun) | **37/50 = 74.0%** |
| all (n=100) | **74.0%** |
| the original 50-episode run | **74.0%** |

Per-task results reproduce as well. **The evaluation code is deterministic, and
every number recorded so far comes back the same when rerun.**

Incidentally, states 0–4 and 5–9 gave exactly the same value. That supports the
choice of running the grid at n=50, though I would not go as far as saying "5
states are proven sufficient" — the two halves could have coincided.

### 6.3 The baseline is a little below the published number

| | measured here | published |
|---|---|---|
| OpenVLA `libero_spatial` | **74.0%** (n=100) | **84.7%** |

About 10 points lower. I suspected the image preprocessing path and checked it,
but that was not the cause (Appendix C), and **I have not yet found what is.**

That said, every conclusion in this report is a **difference (Δ) measured under
identical conditions**. Whether the baseline is 74% or 84%, "turning foveation
on costs 74 points" is the same number, so I do not think this affects the
conclusions much. My understanding is that it is **something to be careful
about only when comparing absolute values against numbers from other papers.**

### 6.4 Not done yet

- **Replication on a second suite.** Every LIBERO result comes from
  `libero_spatial` alone. "Is this just a property of this suite?" has not been
  ruled out. **This is the largest gap right now.**
- The UniVLA baseline (96.0%) has not been checked against a published number.
- The matched comparison in §6.5 below.

### 6.5 How to read the numbers

Each condition was run 50 times, and **50 is fewer than it sounds.**

Rerunning the same condition shifts the result a little each time. It is much
like flipping a coin 50 times: a condition that came out 37 successes out of 50
(74%) might give 33 or 41 on a rerun. **Luck alone moves it by roughly ±6
points.**

Comparing two conditions, both of them wobble, so the range widens to about ±9
points — and to call something a real difference rather than luck, the gap
needs to be roughly twice that, about **18 points**.

| values in §3 | how to read them |
|---|---|
| −46, −68, −74 / blank image 0% | **Certain.** Too large to be explained by chance |
| −8, −10, −16, −18 | **Take the direction only.** Chance is still a possible explanation |

**That calculation treats my experiment worse than it deserves, though.** As
noted in §2.4, every condition **replays exactly the same 50 initial states**.
The two conditions did not solve different problems; they each solved **the same
50 problems**, so episodes can be matched one to one.

| | condition B succeeds | condition B fails |
|---|---|---|
| **condition A succeeds** | both succeed | A only |
| **condition A fails** | B only | both fail |

Here, **"both succeed" and "both fail" tell you nothing about which condition is
better** — those episodes were simply easy or simply hard. The information lives
entirely in the **two cells where the outcomes disagree**, and counting only
those gives a far sharper answer.

The records needed are all kept, and the script is written
(`adaptive_sparse_vla/paired_test.py`). **The values marked "take the direction
only" above will be settled once that is computed.** The present numbers are
stated on the unfavourable basis, so they can go up afterwards but not down.

---

## 7. Where things stand, and what I would like to ask

**The target is the ICRA deadline on 13 September, about 6 weeks away.**

### 7.1 Progress

Colab compute is currently exhausted, so experiments are paused. I intend to
resolve this myself by arranging payment, so it should not become a real
problem. The experiments below will run in order as soon as compute is
available.

| experiment needed | time | status |
|---|---|---|
| SpatialVLA blur standalone (§5.1 footnote ²) | 2–3 h | script ready |
| `libero_object`, 3 conditions × 2 backbones | ≈9.5 h | pending |
| `libero_10` (if time allows) | ≈9.5 h | pending |

In the meantime I am continuing with work that needs no compute: statistical
tests on the existing results, code cleanup, and a first draft.

**One thing I would like to ask.** Would it be possible to check whether there
is any room to use a lab GPU? On a single Colab session one suite takes about
9.5 hours and can only run serially; being able to run them in parallel would
make the schedule far more comfortable. The other point is the number of
episodes per condition — n=50 is, as described in §6.5, the largest weakness of
this work, and with resources I would like to raise it to 100–150. If it is
difficult, I am fine proceeding as things are.

### 7.2 I would like advice on the research direction

**This is the main reason for writing this report.**

What I have been doing is **training-free interventions plugged into existing
policies**. I started from the view that requiring no retraining is a strength,
but running them has shown that **what works and what does not varies by
model.** An intervention that works well on one backbone degrades another
substantially, and even within one backbone the sign flips when the benchmark
changes (§5.2).

As a result it is hard to say "use this method and things improve" — too many
conditions attach. I am not sure how to resolve this, so I have written out
four directions I have considered. **I would be very grateful for your view on
which looks better, or whether there is a direction I have not seen.**

---

**(A) Extend the current axes to a second suite**

Run `libero_object` under the same three conditions to close the gap in §6.4.

- This has to be done in any case, and whichever way it comes out, it is better
  than the present state.
- But there is nothing new in it. Adding one more suite does not, I think, turn
  "interventions differ by backbone" into a paper.
- Necessary, but I doubt it is sufficient on its own.

**(B) Build a value that predicts in advance whether an intervention will work
on a given backbone**

In §3.2 the **shape** of the depth-accuracy curve differed by backbone, and
that shape determined whether the controller succeeded (§3.3). But measuring
that curve takes hours per backbone.

The **layer redundancy profile** (the distribution of `1 − cos(in, out)` across
layers), by contrast, comes from **a single forward pass**. It is already being
measured every episode for depth-prune.

> The question I would like to ask: **could the shape of this essentially free
> profile predict, before running anything, whether depth pruning will work on
> a given backbone?**

If it could, the present observations might become a **tool**. Each new
backbone could be judged in seconds rather than hours, and the problem I am
having — results that differ per backbone and refuse to consolidate — might
become the content of the paper instead.

- The evidence I have is three backbones (Emu3 = flat, almost no loss to 8
  layers / Llama-2 = steady decline / Gemma2 = collapses at one layer). The
  ordering itself matches intuition.
- But it is **a line drawn through three points**, so the evidence is weak. At
  least 5–6 backbones seem necessary, each costing a profile (nearly free) plus
  a confirming experiment (hours).
- Personally this is the direction I would most like to pursue, but I cannot
  judge whether the evidence is sufficient.

**(C) Demonstrate the latency saving on a real robot**

Put the depth controller on hardware and check whether the wall-clock saving
reproduces.

- I understand real-robot experiments carry different weight at ICRA, and I
  suspect simulator latency alone will not be persuasive.
- I am not confident, though, that a 1.13× speedup is a perceptible difference
  on hardware.
- Since you have offered to help with the platform, I will proceed as soon as
  it is ready.

**(D) Write up the benchmark dependence itself as the result**

Report the fact that +18.8 on SimplerEnv became −74 on LIBERO (§5.2) as the
finding — a point about efficiency techniques being validated on a single
benchmark.

- The advantage is that almost all of it can be written from existing data.
- But negative results are, as I understand it, hard to get accepted, and
  without identifying a cause (the "scene composition" hypothesis in §5.2 is
  still unverified) the argument would be weak.

---

My own thinking is that **(A) has to happen regardless, and if (B) holds up it
might turn the current weakness into a strength** — but the fact that (B) starts
from three backbones keeps bothering me.

And to be honest: **so far I have been proceeding by adding one intervention at
a time and filling in the table, and I am worried that in 6 weeks this will
have produced a bigger table and the same claim.** If the direction needs to
change, this feels like the last point at which it can.

### 7.3 A question about the shape of the paper as well

This continues from §7.2. What I have falls into two strands.

| | content | current state |
|---|---|---|
| foveation | accuracy technique. +18.8 on SimplerEnv / −74 on LIBERO | sign flips per benchmark |
| depth controller | the only axis where latency moved | +10.0 on UniVLA, significance unsettled |

**I find it hard to judge whether these two belong in one paper.** They are
different in character — one about accuracy, one about speed — so I worry they
would look forced together; but each on its own feels thin.

To put it plainly: **I have tried a number of ideas, and the results are not
converging into one visible contribution.** The interventions have been run
across several axes and the data has accumulated, but there is still nothing I
could put forward as "this is my method and this is how much it improves
things". I would be grateful for advice on how to organize this.

### 7.4 Could you look at the foveation code?

**The LIBERO foveation result is extreme enough (OpenVLA 74.0% → 0.0%) that I
cannot rule out having missed something when porting it.**

What I have checked on my side:

- It is **bit-identical** to the original function used in the SimplerEnv
  experiments, verified by 29 checks (3 image sizes × 4 keep ratios).
- Foveation is applied at the same point as in the original — on the raw
  environment frame, **before** the policy's own resize.
- I also confirmed that the extra JPEG round-trip present only on the LIBERO
  path does not damage a foveated frame more than a raw one.

Even so, 74.0% → 0.0% strikes me as too large, so if you have time, **could you
take a look at `adaptive_sparse_vla/foveation.py` and the part of
`eval_libero.py` that applies it?** Coming from the person who wrote the
original, something I have missed might be immediately visible.

### 7.5 A question about the original intent of foveation

I would like to confirm whether foveation was designed as an accuracy
technique. In my measurements latency did not fall in any of the three
conditions (§3.1), and my conclusion is that a latency benefit is only possible
in settings where visual tokens dominate the cost.

If the original intent was on the latency side, I may be applying it at the
wrong point, and I would be grateful if you could check that too.

---

## Appendix A: Claims so far, and how confident I am

| # | claim | confidence |
|---|---|---|
| 1 | The direction of a test-time intervention's effect depends on the backbone | **high** — split on all three axes; the large gaps (−46/−68/−74) are certain |
| 2 | The recorded numbers are reproducible | **high** — an independent rerun matches down to the per-task level (§6.2) |
| 3 | Foveation does not reduce latency | **high** — measured directly, and the profile explains why |
| 4 | Non-uniform depth allocation beats uniform allocation (UniVLA) | **moderate** — +10.0 but p≈0.08, below the usual bar. Matching episodes pairwise (§6.5) could raise it, but that is not yet computed |
| 5 | The condition is "a flat region in the depth-accuracy curve" | **moderate** — predicted then confirmed, but 2 backbones and 3 points on the curve |
| 6 | Foveation hurts on LIBERO because of scene composition | **low** — unverified hypothesis. The alternatives (policy strength, gaze placement) have only been narrowed |
| 7 | The 10.7-point gap between my baseline and the published number is a constant offset | **low** — cause unidentified, and constancy is itself an assumption (§6.3) |

## Appendix B: A prediction registered in advance for `libero_object`

Predicting first and checking afterwards has been useful twice, so I am
recording the next experiment in advance as well. `libero_object` has a
suite-specific fine-tune (published 88.4%), so the policy is presumably
exploiting visual detail, and I expect the **same direction** as `spatial`.

| condition | prediction |
|---|---|
| OpenVLA foveation | loss (−10 to −40) |
| OpenVLA depth-prune 8 | large loss (−30 or more) |
| UniVLA foveation | nearly lossless (within −10) |
| UniVLA depth-prune 8 | small loss (within −15) |

If it holds, the reasoning behind the prediction gets stronger; if not, I learn
that the explanations in §3.2–3.3 are wrong. Either way it is a better state
than "only ever seen on one suite".

## Appendix C: Details on the baseline gap (§6.3)

### Checked and ruled out — image compression

OpenVLA's published implementation compresses each image to JPEG and decodes it
before resizing (the training data was stored as JPEG). My code follows the same
order but uses a different library: the original uses TensorFlow, and importing
TensorFlow after LIBERO's graphics libraries have loaded kills the process, so
PIL was used instead.

To check whether this compression step was the cause, I turned off only the
compression and reran the same 50 episodes.

| image path | success (n=50) |
|---|---|
| JPEG compression + resize — the default | **74.0%** (37/50) |
| resize only, no compression | **72.0%** (36/50) |
| published | 84.7% |

A 2-point difference, one episode. Far too small to explain a 10-point gap, and
in the wrong direction — so **the compression step is not the cause.**

### Not yet checked

- **The resizing method itself.** Both libraries use an algorithm of the same
  name, but the implementations differ in detail. This is the only remaining
  **known** difference.
- **Items never audited**: LIBERO / robosuite versions, checkpoint revision,
  maximum episode length, when success is declared, how many runs the published
  number averages. I have **never reproduced 84.7% by running the published
  implementation itself** — so I know my numbers differ, but I have not narrowed
  down where.
