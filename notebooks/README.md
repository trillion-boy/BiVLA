# Method notebooks — one per intervention

Self-contained notebooks for the four conditions in the comparison grid, split
by **method** rather than by backbone so each can be lifted into a different
codebase (CALVIN, a new simulator, a real robot) without untangling flags.

Each notebook carries the method's code inline — copied verbatim from
`adaptive_sparse_vla/`, not imported — so it runs on its own.

| notebook | condition | hook | runs standalone? |
|---|---|---|---|
| `01_original_policy.ipynb` | Original policy | — (defines the loop) | yes, on a stub env |
| `02_fixed_foveation.ipynb` | Fixed foveation | **A** — the observation | yes, `cv2` + `numpy` |
| `03_action_repeat.ipynb` | Action repeat | **B** — the actions | yes, `numpy` |
| `04_fixed_depth_pruning.ipynb` | Fixed depth pruning | **C** — the decoder stack | yes, `torch` only |

**Read `01` first.** It defines the control loop and the three hook points that
`02`–`04` attach to; the other three describe themselves in its terms.

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

## Verification

Every code cell in all four notebooks executes top to bottom with no simulator
and no checkpoint. `04` ends with seven assertions on a small synthetic stack —
including that a bypassed layer is an exact identity and that the KV cache has
no gaps, which is the failure that would otherwise show up only as a quietly
lower success rate.

## What is deliberately not here

The **phase-adaptive depth controller** (bypass fewer layers during the grasp,
more afterwards) is a separate condition from `04`'s fixed pruning and is not
included. Same for gaze-tracking variants of foveation: `02` is the fixed
image-centre version only.
