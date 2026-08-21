# Is this enough for ICRA? — an honest assessment

*Asked directly, so answered directly. Written against what the current grid
plus the planned expansion (5 new models, LIBERO) would actually put in front
of a reviewer.*

---

## Short answer

**The measurement is strong enough. The framing and the venue fit are the
risk.** If I had to put a number on it: as it stands today, borderline — the
kind of paper that gets one champion and one reviewer who says "simulation
only, and you never ran the methods you are criticising." The expansion helps
with breadth but does not touch either of those two objections.

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

### ① "You never ran a published method." — the biggest risk

This is the objection I would raise as a reviewer, and right now the answer is
uncomfortable:

| our intervention | is it a published VLA efficiency method? |
|---|---|
| depth pruning | criterion is ShortGPT's Block Influence — **but ShortGPT is an LLM paper**. MoLe-VLA is the VLA one, and we did not run it |
| foveation | log-polar is from Schwartz / Traver-Bernardino (robot vision). Not a published VLA efficiency method |
| action repeat | frame skip, from DQN. Not a VLA efficiency method at all |

So the sentence "published efficiency claims do not transfer" is currently
supported by **three interventions we implemented ourselves**, none of which
appears in the papers being criticised. A reviewer can say: *you showed your
own three knobs are configuration-sensitive; you did not show that
EfficientVLA or MoLe-VLA are.*

The §2.6 defence ("Bag of Tricks did not invent its methods either") is good
but not complete — Bag of Tricks ran Best-of-N and MCTS **as those methods are
defined**. We ran generic analogues.

**This is fixable cheaply, and that is §5.**

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
| ① never ran a published method | **no.** Same three interventions on more backbones |
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

**1. Run one published VLA method, as published. (highest impact, low cost)**
We already have `fastv_emu3.py` — 336 lines, unit-tested, blocked only on a
`transformers` install. FastV is cited in EfficientVLA's own comparison table.
Running it over the five existing cells is ~1,100 episodes and converts
objection ① from fatal to answered: *we also ran a published method, and it
behaves the same way.*

**2. A minimal real-robot result. (high impact, real cost)**
One task, baseline vs one intervention, ~20 paired episodes on a WidowX. Not
a hardware campaign — an existence check. Pairing survives without determinism
(place the object in the same spot, alternate conditions), and OpenVLA's own
paper evaluates hardware this way.

**3. Table I: what prior work reports. (medium impact, nearly free)**
Rows: EfficientVLA, VLA-Cache, FastV, MoLe-VLA, ShortGPT. Columns: candidate
window · selection constraint · keep value · per-task split · per-episode
records. Mostly empty. Each empty cell is a value we show changes the answer.
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

> The measurement is publication-grade; the exposure is that we test three
> interventions we implemented rather than any published method, and that
> everything is in simulation. Running FastV over the existing five cells
> (~1,100 episodes, code already written) closes the first, and a 20-episode
> real-robot check closes the second — both worth more than the five new
> models, which should be framed as *does sensitivity depend on scale* rather
> than as coverage.
