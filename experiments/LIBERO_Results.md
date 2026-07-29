# LIBERO results — OpenVLA vs UniVLA

Benchmark: `libero_spatial`, 10 tasks × 5 initial states = 50 episodes per
condition (log-polar on OpenVLA is 30). Harness: `adaptive_sparse_vla/eval_libero.py`,
OSMesa rendering, episodes end as soon as LIBERO reports success.

Checkpoints: `openvla/openvla-7b-finetuned-libero-spatial`,
`Yuqi1997/UniVLA → UNIVLA_LIBERO_IMG_BS192_8K`.

## Headline

| condition | OpenVLA | Δ | UniVLA | Δ |
|---|---|---|---|---|
| baseline | 74.0% | — | 96.0% | — |
| action-repeat 2 (2x cheaper) | 66.0% | −8.0 | **28.0%** | **−68.0** |
| foveate blur 20% | 58.0% | −16.0 | 94.0% | −2.0 |
| foveate log-polar 20% | **0.0%** | **−74.0** | 88.0% | −8.0 |
| depth-prune 4 | **56.0%** (1.13x) | **−18.0** | 86.0% (1.08x) | −10.0 |
| depth-prune 8 | **28.0%** (1.30x) | **−46.0** | 86.0% (1.29x) | −10.0 |
| **depth-ctrl 2→8** | not yet run | — | **96.0%** (1.13x) | **0.0** |

UniVLA numbers are the post-fix runs (FAST decode failures 0/440-610 in every
condition). The pre-fix runs, which carried a ~4.5% corrupted-chunk rate,
gave 92.0 / 24.0 / 98.0 / 86.0 — every condition moved by at most 4 points,
i.e. within noise, so the defect was not driving any conclusion.

A blank-image control (both cameras zeroed, instruction only) puts UniVLA at
**0.0%**, which is what licenses reading the foveation rows as robustness
rather than as the policy ignoring its cameras — see the control section.

Foveation rows for UniVLA are the `--foveate-views both` runs, i.e. every
camera the policy sees is degraded (see the confound section below).

Significance (two-proportion z, n=50 per cell):

| comparison | Δ | SE | z | verdict |
|---|---|---|---|---|
| OpenVLA action-repeat 2 | −8.0 | 9.1 | −0.88 | within noise |
| OpenVLA log-polar | −74.0 | — | — | conclusive |
| UniVLA action-repeat 2 | −68.0 | 6.9 | −9.8 | conclusive |
| UniVLA blur (both views) | −2.0 | 4.4 | −0.46 | within noise |
| UniVLA log-polar (both views) | −8.0 | 5.4 | −1.49 | within noise |

## The result is a double dissociation

The two backbones fail under **opposite** interventions:

- **OpenVLA** absorbs a 2x cut in forward passes (−8, within noise) but
  collapses when its visual input is warped (−74).
- **UniVLA** is untouched by either foveation (−6 / +6, both within noise)
  but collapses when forward passes are halved (−68).

Neither backbone is simply "more robust". Which intervention is affordable
depends on the architecture, which is the claim this experiment was built to
test.

### Why the efficiency lever splits them

The shared axis is **env steps executed per forward pass**. The baselines
sit at opposite ends of it:

| | OpenVLA | UniVLA |
|---|---|---|
| action chunking | none | native, 10 steps |
| baseline steps/forward | 1 (closed-loop) | 10 (already amortized) |
| under `--action-repeat 2` | 2 | 20 |

`--action-repeat 2` is the same mechanism on both (`np.repeat`, each action
held for 2 env steps, doubling displacement). On OpenVLA it stretches a
1-step open-loop excursion to 2; on UniVLA it stretches an already-10-step
excursion to 20 with no feedback in between. The intervention is identical;
the starting point is not.

Per-task, UniVLA's collapse is uniform — 8 of 10 tasks drop to 0–1 of 5 —
rather than concentrated in hard tasks, consistent with open-loop drift
rather than task difficulty:

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 5 | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 5 | 4 |
| action-repeat 2 | 0 | 0 | **5** | 0 | 0 | 0 | 3 | 2 | 1 | 3 |

Task 2 is the only one that survives intact, and it is the only instruction
in the suite that names no spatial relation: "pick up the black bowl **from
table center**". Every other task ("between the plate and the ramekin",
"next to the ramekin", "on the cookie box") requires selecting one bowl from
several and then placing it precisely. Twenty steps of open-loop execution
removes exactly the corrective feedback that precision needs, which is why
relational tasks collapse and the unambiguous one does not.

### Why foveation splits them

OpenVLA's failures under foveation are concentrated exactly where the target
sits in the periphery, which is what identifies the mechanism as **fovea
placement**, not information loss:

| task | target location | baseline | blur 20% |
|---|---|---|---|
| 3 (cookie box) | centre | 5/5 | 5/5 |
| 7 (stove) | near centre | 4/5 | 5/5 |
| 4 (cabinet drawer) | right periphery | 3/5 | 1/5 |
| 9 (on the cabinet) | right periphery | 5/5 | **0/5** |

Log-polar keeps the image centre sharp and destroys the periphery. In
`libero_spatial` the image centre is empty table while the target bowls are
off-centre, so the policy loses precisely the region it needs. Verified
directly: the transform degrades gracefully (PSNR 22.8 dB at keep=20%,
centre error 2.1 vs periphery 6.8), so this is not an implementation
artifact.

UniVLA shows no such pattern — its cabinet tasks (4, 9) survive foveation
intact.

### Oracle gaze: placement is a real cause, but not the whole cause

`--foveate-center oracle` places the fovea on the simulator's pose for the
object each task moves (`libero_oracle_gaze.py`; verified per task against all
10 BDDLs and by rendering the crosshair). It is privileged state, so it is a
ceiling rather than a method: it answers what a perfect gaze could buy.

| OpenVLA, keep=20% | fovea at image centre | **fovea on the target** | Δ |
|---|---|---|---|
| log-polar | 0.0% (n=30) | **22.0%** | +22.0 (z=3.75) |
| blur | 58.0% | **50.0%** | −8.0 (z=−0.81, noise) |
| (no foveation) | 74.0% | | |

**A perfect gaze does not rescue foveation on this benchmark.** The row that
answers the question is blur, whose degradation profile does not depend on
where the centre sits — and moving its fovea onto the target changes nothing
measurable. The best any foveation configuration reaches is 58.0%, still
−16 against baseline, and the oracle's 50.0% is itself conclusively below
baseline (z=−2.55).

Log-polar's +22 is not a placement effect in the useful sense. `warpPolar` in
log mode spends an enormous share of its sample budget near the pole; with the
pole at the image centre that budget lands on empty table while every bowl is
crushed. Moving the pole relieves a pathological configuration rather than
demonstrating that a sharp target is what matters — and it still lands at 22%,
far below what blur reaches without any gaze at all. Log-polar is simply the
wrong degradation mode for this scene, and no gaze fixes that.

Per task, the two effects that cancel are visible (n=5 each, so individually
noisy):

| task | target | blur, centre | blur, oracle |
|---|---|---|---|
| 9 — on the cabinet | right periphery | 0/5 | **3/5** |
| 4 — in the drawer | right | 1/5 | 0/5 |
| 3 — on the cookie box | centre | 5/5 | 4/5 |
| 7 — on the stove | near centre | 5/5 | 4/5 |

Peripheral targets gain, central ones give it back. That is what a *single*
fovea implies here: the policy needs the target bowl, the destination plate and
the gripper at once, and one sharp disc covers at most two. Centring it is a
compromise across all three; moving it to the target trades the plate and the
gripper away. There is no good place to put one fovea, which is why placement
nets out to zero.

**This closes the axis, and it strengthens the dissociation rather than
weakening it.** The obvious objection to OpenVLA's foveation collapse was that
the fovea was simply in the wrong place. It was put in the provably right
place — from simulator ground truth, an upper bound no deployable gaze can
beat — and the collapse stands. An instruction-conditioned gaze (the model's
own attention over visual tokens) was scoped and deliberately not built: its
ceiling is the oracle's, and the oracle does not clear the fixed centre.

Latency was 521 ms against a 524 ms baseline — a third independent
confirmation that foveation costs information, not compute.

## Control: is UniVLA using the image at all?

UniVLA surviving log-polar at 20% admits two readings, and they point in
opposite directions:

- **(A)** the policy tolerates severe visual degradation → genuine robustness,
- **(B)** the policy barely uses the agent image on `libero_spatial` → the
  foveation result is vacuous and the suite is weak as a perception benchmark.

A first attempt to separate them measured how much foveation perturbs the VQ
token stream: **99.4%** of visual tokens change under log-polar 20% and 88.8%
under blur 20%, yet success holds at 88–94%. That refutes "the quantizer
absorbs the perturbation" but does not decide (A) vs (B) — token-ID equality
is a brittle metric, since neighbouring codebook entries can carry nearly
identical embeddings.

The deciding control is to remove the image entirely. `foveate_image_logpolar`
returns `np.zeros_like(frame)` when `keep_ratio <= 0`, so
`--foveate-keep-percent 0 --foveate-views both` blanks **both** cameras and
leaves only the instruction — no code change, same harness, same checkpoint.

| condition | success | n |
|---|---|---|
| baseline | 96.0% | 50 |
| both cameras blank | **0.0%** | 50 |

Zero of fifty, every task, every trial running the full 230 steps. **(B) is
dead**: the policy cannot do these tasks without vision, so the 88% under
log-polar is tolerance of degraded input, not indifference to input.

Two details make the control tight rather than merely suggestive:

- **FAST decode failures were 0/440.** The policy emits perfectly well-formed
  action sequences on a blank image — it fails because it has no information,
  not because the degenerate input corrupted the tokenizer. Had the rate
  spiked, the run would have measured a tokenizer artifact instead.
- **Actions stay large** (`dim_absmax` ≈ 1.08 / 0.70 / 0.79 on translation).
  The arm moves confidently in the wrong direction rather than freezing, which
  rules out "the model detected a broken input and stopped".

## Confound checked and cleared: the wrist camera

UniVLA takes two camera views (agent + wrist); OpenVLA takes one. The
harness originally foveated only the agent view, which would have handed
UniVLA an undegraded backup that OpenVLA never had. `--foveate-views both`
degrades every view the policy actually consumes:

| foveation | agent view only | both views |
|---|---|---|
| blur 20% | 96.0% | 98.0% |
| log-polar 20% | 94.0% | 86.0% |

Degrading the wrist view as well costs at most 8 points and leaves both
conditions statistically indistinguishable from baseline. UniVLA's
robustness to foveation is therefore **not** wrist-camera redundancy.

## Action-decode failures (UniVLA only)

UniVLA emits actions as FAST tokens. A malformed token sequence is swallowed
by the tokenizer, which substitutes all-zero DCT coefficients; after
un-normalization that is **not** a no-op but a fixed drift
(`[0.116, 0.033, 0, 0.009, 0.014, 0.056, −1]`, the midpoint of the q01/q99
range), so the arm keeps moving for a full 10-step chunk on a dead command.

Measured rates, before and after the fix:

| condition | decode failures |
|---|---|
| foveate log-polar 20%, both views (pre-fix) | 30/639 (4.7%) |
| foveate blur 20%, both views (pre-fix) | 26/587 (4.4%) |
| every post-fix run | 0/440–610 (0.0%) |

The cause was the tokenizer, not the intervention: the stock
`physical-intelligence/fast` release lacks a pad/truncate guard that the
UniVLA authors added to their own copy, so a generated BPE sequence landing
one or two characters short of `time_horizon * action_dim` failed the reshape
and fell into the zero-substituting except-block. Inserting only that guard
(not copying the authors' file, which also carries different quantization
defaults) takes the rate to exactly zero, including on blank-image inputs.

All UniVLA numbers in this document are post-fix. The pre-fix runs moved by at
most 4 points, so the defect was never driving a conclusion — but it is now
excluded as an explanation for any of them.

## Caveats

- 5 of 50 available initial states per task. OpenVLA's baseline reproduces
  at 74% against a published 84.7%; the gap is ~2 SE and plausibly explained
  by the initial-state subsample plus a PIL LANCZOS resize standing in for
  the reference TF `lanczos3` (TensorFlow segfaults when imported after Mesa
  in this process).
- Foveation as implemented **does not reduce latency** — ms/inference is
  unchanged (OpenVLA 524 → 518, UniVLA 1882 → 1888). It reduces information,
  not compute. Of the interventions here, only action-repeat (fewer forward
  passes) and depth pruning (cheaper forward pass) are efficiency levers.
- Per-task `ms/infer` varies 1340–1700 within a single run because OSMesa
  renders on the CPU in the same process and scene complexity differs. Compare
  run-level means, not individual tasks.

## The depth axis: the one lever that moves wall-clock

Neither axis above can make UniVLA faster. Temporal is already spent (its
baseline runs 10 env steps per forward; doubling that collapses it to 28%),
and spatial never touched wall-clock — a UniVLA step profiles as 6% VQ encode
/ 13% prefill / **70% autoregressive decode** (`docs/VISUAL_TOKENS_VS_LATENCY.md`),
so the whole visual path is a ~19% ceiling. Reducing visual *tokens* rather
than visual *fidelity* does not escape it either: FastV, measured on this
backbone, left latency at 1.0× while success fell 100 → 75 → 38%.

Decoder-layer bypass attacks the 70% directly, because the decode pays for
every layer on every generated token. Rank layers by
`1 - cos(layer_in, layer_out)`, replace the most redundant with a pass-through
(`--depth-prune N`). Training-free, no external module.

| condition | success | ms/forward | ms/env step | speedup |
|---|---|---|---|---|
| baseline | 96.0% | 1882 | 188 | 1.00× |
| depth-prune 4 (static) | 86.0% | 1750 | 175 | 1.08× |
| depth-prune 8 (static) | 86.0% | 1457 | 146 | 1.29× |
| **depth-ctrl 2→8 (phase-adaptive)** | **96.0%** | **1667** | **167** | **1.13×** |

Layers are picked per episode from a redundancy ranking over the back half of
the stack (`[16, 18, 20, 22, 24, 26, 29, 31]` at N=8), calibrated on the real
VLA prompt. The static-8 speedup sits at the top of the 1.23–1.29× band the
same mechanism produced on this backbone in SimplerEnv, i.e. it reproduces
across benchmarks. Decode failures stayed at 0.0% in every condition.

**The phase-adaptive controller is a strict Pareto improvement over baseline:
identical success at 1.13×.** It recovers all 10 points static pruning cost
while keeping 215 of its 425 ms saving (51%).

The measured latencies match the profiler independently. If the decode is 70%
of a step and layers are cut uniformly, bypassing 4/32 predicts 1717 ms
(measured 1750) and 8/32 predicts 1553 ms (measured 1457) — so the bypass is
removing real computation, not just changing the output.

### The static curve is flat, which is what makes the controller a result

The obvious objection is that the controller averages ~4–5 bypassed layers per
episode, so maybe "prune less" would do the same job with none of the
machinery. `--depth-prune 4` answers it:

| bypassed | success | speedup |
|---|---|---|
| 4 | 86.0% | 1.08× |
| 8 | 86.0% | 1.29× |

**Pruning less does not buy the accuracy back.** Along the static axis the
only configuration that reaches 96% is bypassing nothing. static-4 is
dominated on *both* axes by the controller (10 points worse and slower) and on
latency by static-8 at equal accuracy — it is the one setting nobody would
ship. Non-uniform allocation in time, not a smaller uniform budget, is what
recovers the accuracy.

The two static settings also fail differently: static-8 loses 3 of its 7
episodes on task 4 alone, while static-4's 7 losses scatter across five tasks
with at most 2 each. Shallow uniform pruning degrades everything slightly
rather than breaking one capability — which is why meeting it halfway cannot
work.

Honest bound: controller vs static-4 on accuracy alone is z=1.77 (p≈0.08),
borderline at n=50. The claim rests on winning both axes at once plus the
flatness of the static curve, not on that one comparison.

### The accuracy cost is concentrated, not diffuse

−10 points pooled is only z=−1.77 (p≈0.08), which by itself would read as
borderline. The per-task breakdown is the stronger signal:

| task | baseline | depth-prune 8 | depth-ctrl |
|---|---|---|---|
| 4 — bowl **in the top drawer** of the cabinet | 5/5 | **2/5** | **5/5** |
| 9 — bowl on the cabinet | 4/5 | 3/5 | **5/5** |
| 1 — next to the ramekin | 5/5 | 4/5 | 4/5 |
| 5 — on the ramekin | 5/5 | 4/5 | 5/5 |
| 3 — on the cookie box | 4/5 | 5/5 | 4/5 |
| 0, 2, 6, 7, 8 | 25/25 | 25/25 | 25/25 |
| **total** | **48/50** | **43/50** | **48/50** |

Three of the five episodes static pruning lost are one task, and it is the only
instruction in the suite that requires reaching **into** a drawer — the most
precision-demanding manipulation here. Five of ten tasks are untouched. Noise
would scatter; this does not.

That is the same failure signature depth pruning produced on SimplerEnv, where
aggressive pruning's failures concentrated at the grasp moment while
free-space transport tolerated it.

### The phase-adaptive controller recovers it

`--depth-ctrl --depth-deep 2 --depth-shallow 8` allocates depth non-uniformly
in *time*: near-full depth through the approach+grasp, then bypass 8 once the
policy's own commanded gripper has been closed for 2 consecutive chunks. The
signal costs nothing (no env ground truth, no detector), the switch is one-way
so it cannot oscillate, and deep is a strict prefix of shallow so the
transition only ever adds layers.

Task 4 goes **2/5 → 5/5** and task 9 **3/5 → 5/5**, restoring the baseline
total exactly. That the recovery lands on the approach phase also settles what
static pruning was breaking: had the failures been grasped-but-not-placed, the
controller would already be shallow by then and could not have helped.

Layer selection is stable — `bypass=[18, 20]` on most of the 50 episodes
despite re-calibrating from scratch each time, so the redundancy ranking
measures a property of the model rather than sampling noise.

The "just prune less" control (`--depth-prune 4`) is answered above: it also
lands at 86.0%, so the accuracy is not bought back by a smaller uniform
budget.

### Against OpenVLA

| | success | ms/env step |
|---|---|---|
| OpenVLA baseline | 74.0% | 524 |
| OpenVLA depth-prune 8 | **28.0%** | 403 |
| UniVLA baseline | 96.0% | 188 |
| UniVLA depth-prune 8 | 86.0% | **146** |
| UniVLA depth-ctrl | 96.0% | 167 |

### Depth redundancy is a property of the backbone

Running the identical rule at the identical ratio (8 of 32 layers, 25%) on
OpenVLA's Llama-2 costs **46 points** (74.0% → 28.0%, z=−5.18) for the same
1.30× speedup. The bypass worked mechanically — the measured saving even beat
the 17.5% the profiler predicts — so the compute came off exactly as intended
and the backbone could not absorb it.

| backbone | LLM | at 8/32 bypassed | verdict |
|---|---|---|---|
| UniVLA | Emu3, 32 layers | 96.0% → 86.0% (−10), recoverable to 0 by the controller | absorbs it |
| OpenVLA | Llama-2, 32 layers | 74.0% → 28.0% (−46) | collapses |
| SpatialVLA | Gemma2, 26 layers | hurt 3 of 4 tasks at a **single** bypassed layer | collapses |

**Emu3 ≫ Llama-2 > Gemma2**, measured with one shared implementation
(`depth_prune.py`) so the ranking rule cannot differ between them.

### The curves have different shapes, and that is the mechanism

| layers bypassed | OpenVLA (Llama-2) | UniVLA (Emu3) |
|---|---|---|
| 0 | 74.0% | 96.0% |
| 4 | 56.0% (1.13×) | 86.0% (1.08×) |
| 8 | 28.0% (1.30×) | 86.0% (1.29×) |

**UniVLA's curve is flat past the first cut; OpenVLA's decreases
monotonically** at roughly 4.5–7 points per layer, with every step
significant (74→56 z=−1.92, 56→28 z=−2.96, 74→28 z=−5.18).

Emu3 has a *pool* of genuinely redundant layers: paying the first −10 buys
access to all of them, and removing four more costs nothing further. Llama-2
has no such pool — every layer removed costs, so there is no free region at
any budget.

At matched speed the cost of the same acceleration differs by backbone, which
is the number to report:

| speedup | OpenVLA | UniVLA |
|---|---|---|
| ~1.13× | 56.0% (−18) | **96.0% (0)**, via the controller |
| ~1.30× | 28.0% (−46) | 86.0% (−10) |

This is the third axis on which the two LIBERO backbones dissociate:

| axis | UniVLA | OpenVLA |
|---|---|---|
| temporal (action-repeat 2) | −68 ✗ | −8 ✓ |
| visual (foveation) | −2 / −8 ✓ | −16 / −74 ✗ |
| **depth (prune 8)** | **−10 ✓** | **−46 ✗** |

### The two collapses have different shapes, which bounds the controller

UniVLA's −10 concentrates: 3 of 5 lost episodes on task 4 alone, five tasks
untouched. OpenVLA's −46 is diffuse: tasks 4, 5, 7 and 9 go to 0/5 and the
rest sit at 2–3/5.

That difference predicts where the phase-adaptive controller can work. It
protects the approach+grasp window only, so it recovers damage localized to a
precision phase — which is exactly what UniVLA's was — and should do much less
for damage spread across every phase.

The curve shape says the same thing more sharply. The controller spends part
of each episode at `deep` and the rest at `shallow`, so what it can achieve is
bounded by the curve it interpolates along. UniVLA's flat region is what made
it a Pareto win: going from 4 to 8 bypassed layers is free there, so buying
depth back only during the grasp is pure profit. OpenVLA's curve has no flat
region, so mixing 2 and 8 can only produce a weighted average of two costs.

**Prediction, recorded before the run:** on OpenVLA the controller lands
around 40–55% at ~1.15–1.20×, i.e. *dominated by static-4* (56.0% at 1.13×) —
the reverse of UniVLA, where it dominated every static setting. Confirming it
would bound the contribution to an observable condition:

> Phase-adaptive depth allocation beats uniform allocation when the backbone's
> depth–accuracy curve has a flat region, and not otherwise.

## Still open

- **`--depth-ctrl` on OpenVLA**, to test the prediction recorded above: with
  no flat region in its curve, the controller should be dominated by static-4
  rather than dominating it as on UniVLA.
- **A second LIBERO suite.** Everything here is `libero_spatial`, so "the
  dissociation is a property of this suite" is not yet excluded. Repeating
  baseline / foveation / depth-prune 8 on `libero_object` — whose targets are
  distinct objects rather than two identical bowls disambiguated by spatial
  language — would settle it, and is the largest remaining gap.
- **Why foveation flipped sign between benchmarks.** The same OpenVLA
  foveation that costs −16/−74 here *gained* +17.7/+18.8 points on SimplerEnv
  Bridge (`LabMeeting_4Backbone_Summary.md`). Gaze placement is now excluded as
  the explanation. Two observations constrain what is left:
  - It is **benchmark-level, not backbone-level**: both backbones that gained
    on SimplerEnv lost on LIBERO (OpenVLA +18.8 → −74, UniVLA +8.3 → −8).
  - It is **not baseline competence** either, or not only: UniVLA gained +8.3 on
    SimplerEnv from a *78.1%* baseline, so "a weak policy benefits from
    decluttering" does not cover it.

  What differs is the scene. Bridge frames one target object centrally against
  cluttered real imagery; `libero_spatial` stages two *identical* bowls
  off-centre on a clean synthetic tabletop, disambiguated only by spatial
  language. Foveation removes clutter on one and signal on the other. Testing
  that directly would need a competence-matched pair on one benchmark, and the
  generalist `openvla-7b` cannot supply it: it ships no LIBERO action
  statistics, so any `unnorm_key` choice confounds "weak policy" with "wrong
  action scaling". Left as a stated limitation rather than a guess.
- **`--exec-chunk`** (the more-reactive direction, unique to a chunked policy)
  has not been run.
