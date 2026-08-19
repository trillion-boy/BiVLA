# Method notebooks — one per intervention

Notebooks for the four conditions in the comparison grid, split by **method**
rather than by backbone so each can be lifted into a different codebase
(CALVIN, a new simulator, a real robot) without untangling flags — plus a
fifth that runs the campaign with them.

`01`–`04` carry the method's code inline — copied verbatim from
`adaptive_sparse_vla/`, not imported — so each runs on its own.

| notebook | condition | hook | runs standalone? |
|---|---|---|---|
| `01_original_policy.ipynb` | Original policy | — (defines the loop) | yes, on a stub env |
| `02_fixed_foveation.ipynb` | Fixed foveation | **A** — the observation | yes, `cv2` + `numpy` |
| `03_action_repeat.ipynb` | Action repeat | **B** — the actions | yes, `numpy` |
| `04_fixed_depth_pruning.ipynb` | Fixed depth pruning | **C** — the decoder stack | yes, `torch` only |
| `05_run_campaign.ipynb` | — (runs all eight) | — | dry run yes; real runs need a simulator |

`01`–`04` are the method, stated without naming a model or a benchmark.
**`05` is the other half**: it builds the real SimplerEnv and LIBERO
environments, sweeps the eight conditions, and writes the files. It loads
`01`–`04` from the `.ipynb` files in the same directory rather than copying
their code, so it cannot drift from them.

The only thing `05` does not supply is **the policy** — an object exposing
`step(image, instruction)` and `reset()`, plus a `.model` attribute if depth
pruning is wanted. Its `check_policy()` catches the shape mistakes on one
fake frame; the two that fail quietly (`unnorm_key` and the gripper
convention) are caught only by comparing the baseline to the model's own
published number, which is step 3 of `05`'s closing checklist.

**Read `01` first.** It defines the control loop and the three hook points that
`02`–`04` attach to; the other three describe themselves in its terms.

## Method first, our results last

Each notebook states the method and its hook point without reference to any
particular model, then keeps whatever we happened to measure in a clearly
marked appendix at the end. **No code cell names a model or a benchmark** — the
only place benchmark constants appear is inside an `EnvAdapter`, which is the
one object you are meant to replace.

The appendices report backbones anonymously (A/B/C) on purpose. What transfers
is the size of the spread between them and the questions worth asking on a new
setup; the specific numbers do not, and labelling them invites exactly the
assumption the appendix warns against.

## Why the hook points are the same for every backbone

Each method is defined at a point that exists in every VLA, not at one specific
to an architecture:

- **A** is on the *environment's* camera frame, before the policy's own
  preprocessing. Whatever comes next — SigLIP patches, a VQ tokeniser — starts
  from that frame.
- **B** is on the *action array*. Every policy returns one.
- **C** is on a `torch.nn.ModuleList` of decoder layers. Every LLM-based VLA has
  one, though its attribute path differs per wrapper, which `04` handles by
  walking candidates rather than hard-coding.

This matters for the comparison, not just for tidiness: if two backbones are
hooked at different points, a difference in their results says nothing about the
backbones. **When porting, keep the hooks and change only the env/policy
adapters.**

## Each notebook states what it assumes about the policy

The hook *points* are universal; the wiring is not. Every notebook has a
"what this assumes" section listing the places a different architecture or
benchmark breaks the method — because all of these fail **quietly**, lowering
the success rate rather than raising an error, which is indistinguishable from
"the method does not work on this backbone".

The ones most likely to bite when porting:

- **02** — a policy that consumes a *window* of past frames needs every frame in
  the window foveated, not just the newest.
- **03** — action repeat is only meaningful in a **relative/delta** action space.
  In an absolute one it is a no-op, and the run will look like a free win.
  CALVIN supports both modes; check which is active.
- **04** — assumes a flat stack of interchangeable self-attention decoder layers
  whose autoregressive decode dominates the step. Interleaved cross-attention
  blocks, non-LLM action heads, and non-`DynamicCache` cache types each break a
  different part of it.

## Nothing benchmark-specific is in the loop

`01` puts every simulator difference into a single `EnvAdapter`: what `reset()`
and `step()` return (classic gym's 4-tuple vs gymnasium's 5-tuple), where the
camera frame lives, how success is reported, whether there is a settle period.
Porting means writing one adapter — the loop and all three hooks stay untouched.
LIBERO and SimplerEnv adapters are included as examples, neither privileged.

## Verification

Every code cell in all five notebooks executes top to bottom with no simulator
and no checkpoint, and each notebook ends in assertions rather than eyeballing:

- **01** runs the loop against both simulator APIs × both policy shapes
  (chunked and single-action, including a flat `(action_dim,)` return), plus a
  settle period and an unsolvable env.
- **02** checks the hook on a single array, a dict of views and a list of views,
  and that restricting to named views leaves the others bit-identical.
- **03** shows repeat=2 halving the calls while leaving env steps unchanged, for
  both a single-action and a chunk-10 policy.
- **04** ends with nine checks on a synthetic stack — including that a
  bypassed layer is an exact identity, that the KV cache has no gaps (the
  failure that would otherwise surface only as a quietly lower success rate),
  that the two layer-window conventions really do select different layers,
  and that calibration survives `reset_episode()` by default.
- **05** dry-runs the whole driver on stubs: eight conditions produce eight
  correctly named directories, the depth conditions bypass layers and restore
  them, every env the factory built gets closed, and the written files pair on
  `ep_id`. Only the two `build_*_env` functions need a simulator.

## The output files

`01`'s `run_condition` writes `<out_dir>/<condition>/<task>/results_<task>.json`
with one record per episode. The field names are load-bearing: the paired
tooling looks up `ep_id`, `success`, `steps`, `model_ms_per_infer` and
`model_ms_per_env_step` by name, and a file written with other names does not
load at all rather than loading wrong.

`ep_id` is **the environment's** episode index, not a loop counter — it is what
fixes the initial state, and it is the key every condition is paired on. `01`
carries the mapping for both SimplerEnv suites, including the coke-can tasks,
which have no `episode_id` at all and index a 5 × 5 grid of object placements
instead. A bare `range(n)` there does not reach that grid, and the result is two
conditions that look paired and are not — which nothing downstream can detect.

## Known gaps

Stated rather than worked around.

- **One camera view.** `run_episode` passes a single image to
  `policy.step(image, instruction)`. UniVLA also consumed a wrist view and its
  wrapper had to be given one; a model that needs a second view needs it
  plumbed through `EnvAdapter.get_image`, the hook and the policy call.
  Foveation's hook already handles a dict or a list of views, so the loop is
  the only piece to extend.
- **Bridge episodes are seeded in `05`; ours were not.** Our Bridge runs passed
  only `episode_id`; `build_bridge_env` also sets `seed=ep_id`, the way the
  Fractal protocol always did. That matches conditions more exactly, not less,
  but a Bridge column produced by `05` is not bit-identical to ours. It does
  not need to be — every column is compared against its own baseline.

## What is deliberately not here

The **phase-adaptive depth controller** (bypass fewer layers during the grasp,
more afterwards) is a separate condition from `04`'s fixed pruning and is not
included. Same for gaze-tracking variants of foveation: `02` is the fixed
image-centre version only.
