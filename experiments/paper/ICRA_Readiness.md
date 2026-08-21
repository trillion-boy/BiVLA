# Is this enough for ICRA? — an honest assessment

*Asked directly, so answered directly. Written against what the current grid
plus the planned expansion (5 new models, LIBERO) would actually put in front
of a reviewer.*

---

## Short answer

**The measurement is strong enough. The framing and the venue fit are the
risk.** As it stands today: borderline — the kind of paper that gets one
champion and one reviewer who says "simulation only." The expansion to 15
cells adds breadth but does not touch that.

Two changes would move it more than 10 more models would, and both are cheap.
They are §5 below.

---

## What is genuinely strong

These are not things most submissions have.

**1. A quantified, reproducible finding that nobody has reported.**
An unreported eligibility flag moves one cell's result by 45.9 points, with
everything else — backbone, benchmark, method, layer count, compute saved —
held fixed. That is a clean, single-variable result. Reviewers respond to
single-variable results.

**2. Sign reversals where both sides individually survive correction.**
OpenVLA $+15.6$ against SpatialVLA $-17.8$ on the same 135 episodes, same
intervention, $p = 4.3\times10^{-7}$. This is much harder to dismiss than
"results varied."

**3. Statistics well above the norm for the venue.**
Paired McNemar on matched initial states, an explicit determinism check,
Bonferroni over a family derived from row counts rather than counted by hand,
and per-episode records released. Most robotics papers report a single mean
over N episodes with no test at all. A reviewer who reads carefully will notice
this, and it is the strongest single defence against "your N is small."

**4. A negative-mechanism result with a traced cause.**
Foveation peaks at keep = 100%, and we can show why — the log-polar round trip
is lossy asymmetrically. "This gain cannot be explained by compression" is a
substantive claim, not a null result.

**5. Genre precedent, already documented.**
`RelatedWork.md` §2.6 makes the case: *Bag of Tricks for Image Classification*
(CVPR 2019) and the NeurIPS'25 D&B inference-time paper both re-measure methods
their authors did not invent. This is a recognised paper type and the document
already anticipates the "you just re-ran other people's methods" objection.

**6. The competitor table is real.**
Of the five closest papers, none crosses both a backbone axis and a benchmark
axis, and none pairs episodes. Gaze-Reg has both axes and leaves the crossing
cell empty. That is a defensible novelty claim, stated with the right hedge
("we are not aware of").

---

## What could sink it, ranked

### ① "You never ran a published method." — real, but a framing risk, not an experimental gap

**Corrected from an earlier version of this document, which overstated it and
named the wrong target.**

Two things make this weaker than it first looks.

**Each intervention already has a defensible provenance:**

| our intervention | what it actually is |
|---|---|
| depth pruning | ShortGPT's Block Influence criterion, applied to VLA. A published criterion |
| foveation | a **VLM/vision technique tested for transfer to VLA** — which is the question VLA-Cache itself poses about FastV, SparseVLM and ToMe |
| action repeat | frame skip, and the direct ancestor of the action-chunking that OpenVLA-OFT made standard. A training-free lever the field uses |

The foveation framing is the one worth stating explicitly in the paper,
because VLA-Cache already made the argument for us: VLM acceleration methods
*"reduce redundancy within a single image but disregard the temporal and
spatial structure essential for robotic tasks under closed-loop control."*
Asking whether a vision technique survives the move to VLA is a recognised
question in this literature, not a substitute for asking a different one.

**And we already have experimental evidence about published methods — from
their own tables.** The twelve-configuration `pick coke can` / `move near`
split is drawn from EfficientVLA, VLA-Cache and FastV's published numbers. It
is observational rather than a rerun, but it is evidence about the methods
being criticised.

**What the risk actually is.** Not that we must rerun anyone. It is that a
reviewer may read a stronger claim into the paper than we make — "published
methods do not transfer" instead of "effects of this kind are configuration
properties, and current reporting cannot tell you which." The Introduction
already hedges correctly (*"that premise, not any individual method, is what
we test"*), and Related Work should hold the same line. This is a wording
problem, and wording problems are cheap.

**Not MoLe-VLA.** An earlier version of this document listed it as something
we should run. That was wrong: MoLe-VLA requires CogKD self-distillation, and
its own Table 5 shows the router **alone scores below baseline** (56.3%,
$-0.9$) — the gain comes from the distillation. It is not training-free, so it
is out of scope by definition, and running it would be a category error.

**FastV is still worth running, for a better reason than the objection.**
VLA-Cache published a specific number: on OpenVLA, FastV leaves FLOPs
unchanged (1.864 T) and *increases* latency (51.91 → 53.28 ms). That is one
backbone on one benchmark — exactly the shape of claim our grid exists to
test, and this time with a published number to compare against. If FastV's
behaviour is stable across our five cells, that is a result; if it is not, it
is a stronger one, and it lands on a method the field actually cites.

### ② Simulation only, at a robotics conference

ICRA reviewers weight hardware. A VLA efficiency paper with zero real-robot
episodes will draw "SimplerEnv only" from at least one reviewer, and the
expansion to 15 cells does not change that — it is still all simulation.

The plan of substituting breadth for hardware is a reasonable bet, but it is a
bet. Worth knowing that `Overview.md` already frames the hardware experiment
well: whichever way it comes out is a result, because if the simulation
prediction fails on a real WidowX, that *strengthens* the "measured in one
configuration" thesis. Even a small real-robot run — one task, two conditions,
20 paired episodes — changes the paper's category.

### ③ It is a critique paper at a method venue

ICRA rewards new capability. A paper whose contribution is "the existing
evidence does not support what it claims" needs to be framed as **giving the
field something**, not taking something away. Contribution 4 (the measurement
procedure) is that thing, and it is currently last in the list. It should
probably be first, or at least co-equal.

### ④ Two smaller ones

- **Unexplained baseline gap.** OpenVLA on `libero_spatial` at 74.0% against a
  published 84.7%. The "four of five are higher, so it is not a systematic
  error" argument is good and should be in the paper, not only in the
  appendix — a reviewer who finds this number unaided will assume a setup bug.
- **GPU not recorded.** Small, already bounded at 3.1 points, but it is exactly
  the kind of reporting gap the paper criticises others for. Stating it
  ourselves is the only good option, which the draft already does.

---

## Does the expansion fix any of this?

| objection | does 5 new models + LIBERO help? |
|---|---|
| ① a reviewer reads a stronger claim than we make | **no** — but it is a wording fix, not an experiment |
| ② simulation only | **no.** More simulation |
| ③ critique at a method venue | **no** |
| "N is small" | somewhat — 15 cells, and the correction family grows to 105 |
| "only three backbones" | **yes, substantially** |
| "only SimplerEnv" | **yes** — LIBERO is a second benchmark family |
| "is this a scale effect?" | **yes, and this is the expansion's real value** |

The last row is the one to lead with. Going 4–8.5B → 0.2–8.5B lets the paper
ask a question it cannot ask now: **does configuration sensitivity depend on
model size?** If small models are *more* sensitive, that is a finding, and it
is directly useful to anyone deploying a small VLA. If sensitivity is flat
across a 40× parameter range, that is a stronger version of the current claim.

**Frame the expansion as that question, not as coverage.** "We also ran five
more models" invites *more of the same*. "We tested whether the effect is a
property of scale" is a result either way.

---

## What would move the needle most, per unit effort

Ranked. The first two are worth more than the entire model expansion.

**1. Run FastV over the five existing cells. (high impact, low cost)**
`fastv_emu3.py` is 336 lines, unit-tested, blocked only on a `transformers`
install; ~1,100 episodes. The reason is not to deflect objection ① but that
VLA-Cache published a concrete claim about FastV on one backbone — FLOPs
unchanged, latency up — and our grid is built to ask whether such a claim
holds elsewhere. It also adds the token-space half of the visual axis
(`MethodAxes_Survey.md` §3).

**2. A minimal real-robot result. (high impact, real cost)**
One task, baseline vs one intervention, ~20 paired episodes on a WidowX. Not
a hardware campaign — an existence check. Pairing survives without determinism
(place the object in the same spot, alternate conditions), and OpenVLA's own
paper evaluates hardware this way.

**3. Table I: what prior work reports. (medium impact, nearly free)**
Rows: EfficientVLA, VLA-Cache, FastV, ShortGPT — the training-free ones.
Columns: candidate window · selection constraint · keep value · per-task
split · per-episode records. Mostly empty. Each empty cell is a value we show changes the answer.
This is the same move the NeurIPS reference paper makes with its Table 1, and
it converts our first result from an accusation into a documented gap.

**4. Reorder the contributions.** Put the measurement procedure first. The
paper gives more than it takes away; the list should say so.

**5. Then the model expansion**, framed as the scale question.

---

## Venue note

If ICRA rejects, this paper fits a **benchmark/datasets track** better than a
method track — the NeurIPS D&B paper the mentor shared is the closest match in
genre, and CoRL is more tolerant of measurement papers than ICRA. Worth
knowing before writing, because the framing differs: ICRA wants "here is what
you should do differently tomorrow," a D&B track wants "here is a resource and
a protocol."

That is a fallback, not a recommendation. With ①+③ from §5 done, ICRA is a
reasonable target.

---

## The one-line version for the mentor

> The measurement is publication-grade. The two real exposures are that
> everything is in simulation, and that a reviewer may read a stronger claim
> into the paper than we make — the second is a wording fix, the first is not.
> A 20-episode paired real-robot check changes the paper's category; running
> FastV over the existing five cells (~1,100 episodes, code already written)
> adds a published method with a published number to argue against. Both are
> worth more than the five new models, which should be framed as *does
> sensitivity depend on scale* rather than as coverage.
