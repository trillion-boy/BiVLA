# What other training-free axes could we add?

*Answering the question "are there methods beyond depth pruning, foveation and
action repeat that we could cover?" — with a filter, a repo inventory, and a
recommendation.*

---

## 1. The filter — what makes something a *fourth axis*

Our three interventions were not picked because they are popular. They were
picked because each **spends a different resource**:

| axis | resource | our intervention |
|---|---|---|
| time | how **often** the policy runs | action repeat *k* |
| vision | what the policy **sees** (pixels) | foveation |
| depth | how many **layers** each call uses | depth pruning *k* |

So a candidate earns a place only if it passes three tests:

1. **New resource.** It must spend something the three above do not. A method
   that removes layers by a different criterion is a *variant*, not an axis.
2. **Training-free and checkpoint-untouched.** Anything requiring a fine-tune
   is a different paper.
3. **Runs uniformly on all three backbones.** This is the hard one, and it is
   the test that eliminates most candidates. Our whole argument rests on the
   grid being uniform: an axis that only runs on two of three backbones cannot
   answer "does this transfer across backbones," which is the question.

Test 3 is worth stating in the paper regardless of what we add. **Several
well-known methods are structurally impossible on one of our backbones**, and
that is itself evidence for our thesis — the field's methods are not as
architecture-agnostic as the papers imply.

---

## 2. What we already have code for

More than I expected. The result-file schema already carries seven
intervention fields, of which we report three:

```
foveate · tome · temporal_stride · depth_prune · spec_decode · action_repeat · exec_chunk
        ^^^^^^   ^^^^^^^^^^^^^^^                 ^^^^^^^^^^^                  ^^^^^^^^^^
        already in the schema, never run in the grid
```

| implementation | file | lines | state |
|---|---|---:|---|
| FastV (attention-based token pruning) | `adaptive_sparse_vla/fastv_emu3.py` | 336 | written, unit-tested (`test_fastv_logic.py`) |
| Self-speculative decoding | `adaptive_sparse_vla/self_spec_decode.py` + `emu3_self_spec_decode.py` | 399 | written, unit-tested, **logic test passes today** |
| ToMe (token merging) | `SpatialVLA/experiments/tome/tome_siglip.py` | 255 | written for SigLIP |
| Temporal feature caching | `SpatialVLA/.../tome_spatialvla_eval_adaptive.py` | — | SpatialVLA-only wiring |
| Chunk execution | `--exec-chunk` in `eval.py` | — | wired, and deliberately excluded (see §4) |
| Phase-adaptive depth | `DepthController_univla.md` | — | run on LIBERO, not in the grid |

The two failing unit tests fail only on a missing `transformers` install, not
on logic. So the engineering cost of a fourth axis is much lower than starting
from zero.

---

## 3. The candidates, scored against the filter

| candidate | new resource? | all 3 backbones? | in repo? | verdict |
|---|---|---|---|---|
| **Visual token pruning** (FastV, SparseVLM, VLA-Pruner) | **yes** — sequence length | yes, via 3 impls | **yes** | **add first** |
| **Self-speculative decoding** | **yes** — decode steps | yes (all 3 decode autoregressively) | **yes** | **add second** |
| **Input resolution** | **yes** — token count via pixels | yes, trivially | partly (`--image-size`) | **cheap, high value** |
| Token merging (ToMe) | same as token pruning | **no** — UniVLA has no ViT | yes | variant, and blocked |
| Temporal feature caching (VLA-Cache) | yes — recompute frequency | probably, untested | SpatialVLA only | possible, more work |
| Quantization (INT8/FP8/4-bit) | **yes** — bits per weight | yes | no | possible, but breaks determinism |
| Early exit / adaptive depth (CALM, MoLe-VLA) | no — depth again | yes | partly | variant of our axis 3 |
| Chunk execution (`exec-chunk`) | yes — actions per call | **no** — see §4 | yes | **excluded, and we say why** |
| KV-cache compression | yes — memory bandwidth | probably | no | out of scope for latency claims |

### The three worth doing, and why each helps *this* paper

**① Visual token pruning — it completes the visual axis.**

This is the one I would add first, because it fixes a gap the paper already
has to explain. Our foveation reduces **pixels** but leaves the **token count**
unchanged — which is exactly why it saves ≈0% compute, and why we have to
argue that it is "an input transformation, not an efficiency technique."

A token-space intervention on the same visual axis turns that from a caveat
into a designed contrast:

> We tested the visual axis in both pixel space and token space. Only the
> latter saves compute, and the two do not agree on which backbones they help.

It also puts us directly alongside the closest prior work (VLA-Cache reports
that FastV and SparseVLM *degrade more on VLAs than on VLMs* — which is our
"does not transfer" claim, from someone else's data).

**② Self-speculative decoding — it is a lossless control.**

Unique property: the output is **byte-identical to full-model greedy decoding
by construction**, because every draft token is verified before acceptance.
That makes it the only intervention in the grid where compute is saved and the
success change is *guaranteed* to be exactly zero.

That is worth more than another data point. Our second claim is "compute saved
does not predict success change." A lossless axis anchors one end of that
scatter with a point we do not have to measure to know: large saving, zero
change. It also gives an honest counterexample to any reading of our paper as
"efficiency methods don't work" — this one always works, and the reason is
that it does not trade anything.

**③ Input resolution — nearly free, and it is the honest comparison.**

Feeding a smaller image reduces the token count for real. It costs almost no
engineering (`--image-size` already exists), and it is the natural control for
foveation: same visual axis, same direction of "show the policy less," but one
of them actually reduces the budget and the other does not.

---

## 4. What we should *not* add, and why saying so is worth a paragraph

**Chunk execution** is already excluded, for a documented reason: native chunk
length differs across our backbones (OpenVLA 1, SpatialVLA 1, UniVLA 5), so
"execute *k* of the chunk" is not the same operation in each cell. It fails
test 3.

**ToMe** merges continuous ViT tokens by averaging. UniVLA/Emu3 has no ViT —
images become a fixed grid of **discrete VQ token IDs** whose dimensions are
declared in the text prefix. You cannot average two discrete IDs, and dropping
any breaks the declared grid. So ToMe is structurally impossible there.

**These two exclusions are findings, not gaps.** They are concrete instances
of the paper's claim: two widely-cited efficiency methods cannot be applied
uniformly across three ordinary open VLAs, which means any cross-backbone
claim about them is untestable as published. One paragraph in Related Work,
and it does real work.

---

## 5. What it costs

Per axis with two settings, on the **current five cells**:

| | cells | episodes |
|---|---:|---:|
| Bridge (3 cells × 2 settings × 96) | 6 | 576 |
| Fractal (2 cells × 2 settings × 135) | 4 | 540 |
| **one new axis** | **10** | **1,116** |

That is **about 15% of the 7,198 episodes already run** — much cheaper than it
sounds, because the grid is already built and the pairing infrastructure
exists. Three new axes is roughly 3,300 episodes.

On the **expanded grid** (15 cells, after the five new models) one axis becomes
about 3,400 episodes, so the ordering matters: **add axes on the current five
cells first, decide what survives, then carry only the survivors to the new
models.**

**One cost that is not episodes.** Each axis widens the correction family. Our
grid family is currently 38 tests (α ≈ 0.0013); three axes × 2 settings × 5
cells adds 30, giving 68 and α ≈ 0.0007. Two of our eight significant cells sit
at p ≈ 0.0010 and would drop out. That is not a reason to avoid the axes, but
it is a reason to decide the family definition *before* running rather than
after.

---

## 6. Recommendation

**Add two axes, not five.** Visual token pruning and self-speculative decoding.
Resolution as a cheap third if time allows.

The reasoning is that the paper's argument does not get stronger by covering
more methods — it gets stronger by covering **more resources**, because the
claim is about whether an effect is a property of the method or of the
configuration. Five token-pruning variants would be one axis measured five
times. FastV plus self-speculative decoding adds two genuinely different
resources, one of which is lossless by construction and therefore anchors the
compute-vs-success argument at a point we cannot be wrong about.

And whatever we add or don't, **§4 should go into Related Work either way**.
The fact that ToMe cannot run on UniVLA and that chunk execution is not the
same operation across backbones costs nothing to report and is directly on
thesis.

---

### Open question for the mentor

Whether to treat the new axes as part of the same Bonferroni family as the
current grid, or as a second family. Pooling is conservative and costs us two
currently-significant cells; splitting is defensible because the axes ask
about different resources. Worth settling before the runs, not after — this is
the same decision we flagged for LIBERO in `ModelExpansion_Plan.md`.
