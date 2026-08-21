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

So a candidate earns a place only if it passes three tests, **checked in this
order**:

1. **Training-free and checkpoint-untouched.** Checked *first*, because it is
   disqualifying and because a method's name and framing do not tell you.
   Anything with a training objective is a different paper. What counts as
   evidence: a loss function, learnable parameters, a router that is fit, a
   distillation stage. Not the abstract's adjectives.
2. **New resource.** It must spend something the three above do not. A method
   that removes layers by a different criterion is a *variant*, not an axis.
3. **Runs uniformly on all three backbones.** Our whole argument rests on the
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

**Test 1 first.** Two candidates fail it outright and are struck through; no
further column matters for them.

| candidate | **training-free?** | new resource? | all 3 backbones? | in repo? | verdict |
|---|---|---|---|---|---|
| **Visual token pruning** (FastV) | **yes** — our impl. has no parameters/loss; uses the model's own attention | **yes** — sequence length | yes, via 3 impls | **yes** | **add first** |
| **Self-speculative decoding** | **yes** — draft and verifier are the same weights | **yes** — decode steps | yes (all 3 decode autoregressively) | **yes** | **add second** |
| **Input resolution** | **yes** — a preprocessing argument | **yes** — token count via pixels | yes, trivially | partly (`--image-size`) | **cheap, high value** |
| Token merging (ToMe) | yes — the off-the-shelf variant needs no training | same as token pruning | **no** — UniVLA has no ViT | yes | variant, and blocked |
| Temporal feature caching (VLA-Cache) | yes — reuses KV, nothing fit | yes — recompute frequency | probably, untested | SpatialVLA only | possible, more work |
| Post-training quantization | yes — PTQ by definition | **yes** — bits per weight | yes | no | possible, but breaks our determinism claim |
| ~~MoLe-VLA~~ | **NO — verified from source** | (moot) | (moot) | no | **disqualified**, see below |
| ~~Early exit (CALM, DeeR-VLA)~~ | **no** — learned exit criterion *(not read from source; provisional)* | (moot) | (moot) | no | **disqualified, pending a read** |
| Chunk execution (`exec-chunk`) | yes | yes — actions per call | **no** — see §4 | yes | **excluded, and we say why** |
| KV-cache compression | yes, for the eviction-policy variants | yes — memory bandwidth | probably | no | out of scope for latency claims |

**MoLe-VLA, checked against the paper rather than its framing.** An earlier
version of this document listed it as a depth-axis variant without checking.
It is not training-free:

- §3.5 is titled *Optimization Objective* and gives a training loss,
  $\mathcal{L}_{\text{MoLe}} = \mathcal{L}_{\text{task}} + \lambda_2
  \mathcal{L}_{\text{cog}} + \lambda_3 \mathcal{L}_{\text{lb}}$ (Eq. 19),
  with an EMA teacher at $\alpha = 0.999$ (Eq. 18).
- The STAR router has learnable parameters and selects layers through
  Gumbel-Softmax, i.e. it is fit by gradient descent.
- CogKD adds a *learnable cognition token* and a teacher-student stage.
- The string "training-free" appears once in the entire paper — in the title
  of reference [42].

And its own Table 5 shows why the training is not incidental: layer skipping
**alone** scores *below* the baseline it starts from.

| | STAR | cognition token | CogKD | mean |
|---|:--:|:--:|:--:|---:|
| Ex0 — CogAct baseline | ✗ | ✗ | ✗ | 57.2% |
| **Ex1−1 — STAR only** | ✓ | ✗ | ✗ | **56.3%** |
| Ex2−4 — full | ✓ | ✓ | ✓ | 60.8% |

So MoLe-VLA is a **citation**, not a candidate. It belongs in Related Work as
the VLA layer-skipping paper, with the observation that its reported gain
comes from the distillation rather than from the skipping — which is a point
in our favour, not a method for us to run.

### How each row's training-free status was checked

The MoLe-VLA error was that I took the framing instead of reading. So this is
what backs each cell of that column, and where it is weaker than it should be.

| candidate | evidence | strength |
|---|---|---|
| **FastV** | our implementation has **no** `nn.Parameter`, no optimiser, no loss, no `.backward()`; the importance signal is the model's own attention read at layer *k*. Grepped, not assumed | **strong for what we would run**; the ECCV paper itself is unread |
| **Self-speculative decoding** | draft and verify **share every weight** — only which layers are active differs, toggled by a caller-supplied mode. Same grep, clean. And `test_self_spec_decode.py` passes today: output byte-identical to plain greedy over 20 seeds at both 0.4 and 0.9 draft-disagreement rates | **strong**, and the losslessness is demonstrated rather than claimed |
| **ToMe** | merging is a weighted average between frozen SigLIP layers, then unmerged; no parameters. Same grep, clean. The ICLR paper does also offer a *with-training* variant we would not use | **strong for the off-the-shelf variant** |
| **Input resolution** | a preprocessing argument | trivially true |
| **Post-training quantization** | training-free by definition of PTQ | true by definition; the specific toolchain still needs choosing |
| **EfficientVLA** | title is literally *Training-Free…*, and `RelatedWork.md` A.1 records from the PDF that the three reductions are done **without training** | **verified from source** |
| **VLA-Cache** | A.2, from the PDF: reuses KV for tokens unchanged between frames; nothing is fit | **verified from source** |
| **ShortGPT** | A.3, from the PDF: BI metric plus **training-free layer removal** | **verified from source** |
| ~~**MoLe-VLA**~~ | §3.5 *Optimization Objective*, Eq. 18–19, learnable STAR router, CogKD distillation | **verified from source — fails** |
| ~~**Early exit** (CALM, DeeR-VLA)~~ | the exit criterion is a learned predictor | **not verified from source** — treat as provisional |
| **SparseVLM, VLA-Pruner** | named in `RelatedWork_Candidates.md`, never read | **not verified** — do not put in a training-free list until read |

**What is still missing, and why.** arXiv, OpenReview, Semantic Scholar and
HuggingFace are all blocked by this environment's egress policy, so FastV,
ToMe, self-speculative decoding, SparseVLM and the early-exit papers could not
be fetched. For the three we would actually run this matters less than it
looks — the question that decides whether they belong in our grid is whether
*our* implementation trains anything, and that is answered by reading our own
code, which is what the table does. But the sentence "FastV is training-free"
in a paper needs the paper, so those three PDFs should be checked once before
submission. The two rows marked *not verified* should not appear in any
training-free list until then.

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

**MoLe-VLA is excluded for a different reason** — it is not training-free at
all (§3). That one is not a finding about architecture, just a scope boundary,
and it should be stated once in Related Work so no reader wonders why the
obvious VLA layer-skipping paper is absent from our comparison.

**The first two exclusions are findings, not gaps.** They are concrete instances
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
