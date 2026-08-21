# The five expansion models, read from source

*We had no papers for any of them — the names appeared only in our own two
`ModelExpansion_*` files. All five are now read. **Three things about the
expansion plan have to change, and one of them is a nine-fold error in a
parameter count.***

Sources: SmolVLA (arXiv 2506.01844), TurboVLA, CoTinyVLA, FLOWER — all PDFs.
MiniVLA is a **blog post with a `@misc` citation**, not a peer-reviewed paper
(Belkhale & Sadigh, Stanford, Dec 2024).

---

## 1. The per-model table, filled in

| | TurboVLA | CoTinyVLA | MiniVLA | FLOWER | SmolVLA |
|---|---|---|---|---|---|
| **params (as given)** | 0.2B | 0.9B | 1B | 1B | **4B** |
| **params (paper)** | **0.2B** ✅ | **0.9B** ✅ | **~1B** ✅ | **950M** ✅ | **450M** ❌ |
| **backbone** | *no LLM* — direct V&L interaction | Qwen3.5-0.8B | Qwen2.5-0.5B + OpenVLA ViT | pruned VLM + Flow Transformer | SmolVLM-2, **first 16 LLM layers only** |
| **action head** | compact decoder, continuous chunk, single forward | autoregressive + CoT tokens | **VQ discrete** (Residual VQ) | flow matching / diffusion | flow matching expert (~100M), 10 steps |
| **native chunk** | chunk (continuous) | chunk-level | **8** (VQ h8) | **20** pretrain / 50 eval | **50** |
| **benchmarks** | LIBERO, RoboTwin, real | **LIBERO-Plus only** | LIBERO-90, **SimplerEnv Bridge** | CALVIN, LIBERO, **SIMPLER Bridge + Google Robot**, +6 | LIBERO, Meta-World, real (SO-100) |
| **code** | `H-EmbodVis/TurboVLA` | `BrainJellyPie/CoTinyVLA` | `Stanford-ILIAD/openvla-mini` | github + HF | `huggingface/lerobot` |

### The parameter error matters

SmolVLA is **450M**, not 4B — from its own §: *"Our main model contains 450
million parameters, with approximately 100 million dedicated to the action
expert."* A factor of nine.

This changes the framing of the whole expansion. The new models are
**0.2–1B**, not 0.2–4B, and our existing grid is **4–8.5B**. There is no
overlap and a gap between 1B and 4B. So the expansion is not "extending the
scale axis downward" — it is **two clusters with a hole in the middle**, which
is a weaker basis for "does sensitivity depend on scale" than a continuous
sweep would be. Worth saying before the runs, not after.

---

## 2. Three of our four conditions do not apply uniformly

This is the axis filter's test 3 — *does it run the same way on every
backbone* — applied to models instead of methods. It fails, and the reasons
are specific.

### Depth pruning: inapplicable or already applied

| model | why |
|---|---|
| **TurboVLA** | **There is no LLM to prune.** It *"independently encodes visual observations and language instructions… and predicts continuous action chunks with a compact decoder"*, explicitly *"avoiding the computation and memory overhead of processing multimodal inputs through a billion-parameter language model"*, *"without autoregressive action-token generation."* No decoder stack, no Block Influence, no axis |
| **SmolVLA** | already uses **only the first 16 layers** of its LLM, by design. Pruning further prunes an already-truncated stack |
| **FLOWER** | *"we prune between **30% and 50%** of the pretrained VLM's layers"* — layer pruning **is its architecture**, not an intervention on it |
| CoTinyVLA | Qwen3.5-0.8B: a normal stack. Applicable |
| MiniVLA | Qwen2.5-0.5B: a normal stack. Applicable |

So depth pruning runs on **two of five**. And the two that already prune are
not a nuisance — they are a result: *the field has begun building our
intervention into the architecture*, which makes "is the reported effect a
property of the method or of the configuration" more pressing, not less.

### Action repeat: the chunk lengths are not comparable

Our existing backbones were 1 / 1 / 5. The new ones are **8, 20–50, 50**, plus
two chunked-but-unstated.

A model that already emits 50 actions per forward is amortised 50× before
action repeat touches it; repeat = 2 puts it at 100 environment steps per
forward. That is not the same operation as repeat = 2 on OpenVLA's single
action. The row can still be run, but it **cannot be compared down the column
without stating the chunk length**, which is exactly the reporting failure our
first result is about.

SmolVLA even publishes the sweep we would want (their Table 12): chunk 1 →
50.0%, chunk 10 → 84.0%, chunk 50 → 80.3% on LIBERO. A 34-point range from the
chunk length alone.

### Foveation: one model needs it applied 16 times

**CoTinyVLA** takes *"dual-view temporal input of **16 history frames** per
step."* Our notebook 02 already warns about this: a policy consuming a window
of past frames needs **every frame in the window** foveated, not just the
newest. Two views × 16 frames = 32 images per step. Applicable, but it is a
different amount of work and a different intervention strength.

**Foveation is the only one of the three that applies to all five.**

---

## 3. Benchmark coverage: only two of five have SimplerEnv

| model | LIBERO | SimplerEnv Bridge | SimplerEnv Fractal |
|---|:--:|:--:|:--:|
| TurboVLA | ✓ | ✗ | ✗ |
| CoTinyVLA | ✓ (LIBERO-Plus) | ✗ | ✗ |
| MiniVLA | ✓ (90) | **✓ — our exact four tasks** | ✗ |
| **FLOWER** | ✓ | **✓** | **✓** |
| SmolVLA | ✓ | ✗ | ✗ |

**This inverts the priority in `ModelExpansion_Plan.md`.** That document treats
SimplerEnv as the primary grid and LIBERO as the extension. For these five it
is the other way round: **all five have LIBERO, two have Bridge, one has
Fractal.**

Two consequences:

- **Baseline validation** — the check that catches `unnorm_key` and gripper
  errors — is only possible against published numbers on LIBERO for all five.
  On Fractal it is possible for FLOWER alone.
- **FLOWER is the model to run first.** It is the only one with published
  numbers on both of our suites, so it is the only cell where a wrong setup
  would be caught before the intervention rows are spent.

MiniVLA is second, and it comes with a bonus: its SimplerEnv table is on our
exact four Bridge tasks, and it shows the same per-task divergence we build
§2.5 on — against OpenVLA it is **+24 on spoon, +8 on stack, −2 on carrot, and
−52 on eggplant**. A 7× smaller model that matches on three tasks and collapses
on the fourth. That is a fifth author group for the per-task-split argument.

---

## 4. What they give Related Work

These five are not just runnable models — they are the efficient-VLA trend
itself, and three of them implement our axes **by construction**:

| our axis | who builds it in |
|---|---|
| depth pruning | FLOWER (30–50% of VLM layers), SmolVLA (first 16 layers only) |
| removing the LLM entirely | TurboVLA |
| action chunking | all five, at 8 to 50 |

That is the paragraph Related Work needs: *the field is no longer only
applying these interventions at inference — it is baking them into
architectures. Which makes it more urgent, not less, to know whether the
reported effect of such an intervention is a property of the method or of the
configuration it was measured in.*

One caution for the bibliography: **MiniVLA is a blog post**, cited as
`@misc{belkhale2024minivla}`. It is legitimate to cite and widely used, but it
should not be introduced with the same weight as a peer-reviewed result.

---

## 5. What to change

| item | change |
|---|---|
| `ModelExpansion_Plan.md`, `ModelExpansion_Tables.md` | SmolVLA 4B → **450M**; reorder the range as 0.2–1B |
| both tables | mark depth pruning **N/A for TurboVLA**, and *"already pruned"* for FLOWER and SmolVLA |
| both tables | add the native chunk column, filled: 8 / 20–50 / 50 |
| plan §"per-model information" | now filled from source, not blank |
| priority | **LIBERO first, not SimplerEnv** — it is the only suite all five share |
| first cell to run | **FLOWER**, the only model with published Bridge *and* Fractal numbers |
| Related Work | the "interventions are becoming architecture" paragraph |
