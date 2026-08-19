# Model Expansion Plan — Small-Parameter VLAs

**Purpose.** Extend the backbone axis downward in parameter count. The current
grid sits at 7–8.5B; these five models cover 0.2–4B. If the findings hold at
this scale too, the claim stops being "a property of large VLAs."

**Status legend.** `Done` = per-episode records exist under `results/`.
`Not` = not yet run.

---

## 1. The handoff table

Four policies per cell, matching the existing grid's four conditions:
**baseline (no intervention) · action repeat · foveation · depth pruning.**

### Already done (current grid, 8 conditions each — more than the 4 below)

| # | Model | Params | Dataset | Policies | Status |
|---|---|---:|---|:--:|---|
| 1 | OpenVLA | 7B | SIMPLER Bridge | 8 | **Done** |
| 2 | OpenVLA | 7B | SIMPLER Fractal | 8 | **Done** |
| 3 | SpatialVLA | ~4B | SIMPLER Bridge | 8 | **Done** |
| 4 | SpatialVLA | ~4B | SIMPLER Fractal | 8 | **Done** |
| 5 | UniVLA | 8.5B | SIMPLER Bridge | 8 | **Done** |
| — | UniVLA | 8.5B | SIMPLER Fractal | — | **N/A** — public checkpoint is Bridge-only |

### To run (new models)

| # | Model | Params | Dataset | Policies | Status |
|---|---|---:|---|:--:|---|
| 6 | TurboVLA | 0.2B | SIMPLER Bridge | 4 | Not |
| 7 | TurboVLA | 0.2B | SIMPLER Fractal | 4 | Not |
| 8 | CoTinyVLA | 0.9B | SIMPLER Bridge | 4 | Not |
| 9 | CoTinyVLA | 0.9B | SIMPLER Fractal | 4 | Not |
| 10 | FLOWER | 1B | SIMPLER Bridge | 4 | Not |
| 11 | FLOWER | 1B | SIMPLER Fractal | 4 | Not |
| 12 | MiniVLA | 1B | SIMPLER Bridge | 4 | Not |
| 13 | MiniVLA | 1B | SIMPLER Fractal | 4 | Not |
| 14 | SmolVLA | 4B | SIMPLER Bridge | 4 | Not |
| 15 | SmolVLA | 4B | SIMPLER Fractal | 4 | Not |

**10 cells × 4 policies = 40 runs.** Bridge is 96 episodes per run, Fractal is
135, so the full set is ~9,240 episodes.

> **If time is limited, run Bridge first (rows 6, 8, 10, 12, 14).** But see
> §4 — running only one benchmark buys only half of what this expansion is
> for.

---

## 2. What each run must produce — please read this before running

Our entire protocol is **per-episode pairing**: we pair episodes by initial
state and count only the ones whose outcome flipped. **A summary success rate
cannot be paired**, so a run that reports only "34.4%" cannot enter the grid
at all.

Each run needs to write, per episode:

| Field | Why |
|---|---|
| `ep_id` | the pairing key — must match across baseline and interventions |
| `success` | boolean |
| `steps` | used for the determinism check |
| `model_ms_per_infer` / `model_ms_per_env_step` | the compute-saving column |
| terminal `info` / `episode_stats` (if the task provides it) | failure-type analysis (§6) |

The existing harnesses already do this; the format to match is any file under
`results/<campaign>/<condition>/<task>/results_<task>.json`.

**One more request:** please record the **GPU model** in the result file.
Our current grid cannot prove all five cells ran on the same card, and that is
an open limitation we would rather not repeat (Report §7 ⑤). One line fixes it
permanently.

---

## 3. Before running — check per model whether each policy applies

This is not bureaucracy. We already learned this the hard way with chunk
execution: the same flag was a speed-up on SpatialVLA, a slowdown on UniVLA,
and undefined on OpenVLA — so it could not be used as a cross-backbone axis at
all (RelatedWork §2.2 c ③). The same risk applies here.

| Policy | What it needs from the model | Likely risk on these five |
|---|---|---|
| **baseline** | a SimplerEnv-compatible checkpoint | **Which checkpoints exist for Bridge / Fractal?** Several of these were trained on LIBERO or their own data |
| **action repeat k** | nothing special — hold one action for k steps | ⚠️ **If the model natively emits action chunks**, repeat interacts with chunking. Need to know each model's native chunk length |
| **foveation** | applied to the observation *before* the encoder | Should apply to all five — it does not touch the model. Confirm input resolution per model |
| **depth pruning** | a stack of decoder layers, and Block Influence must be computable per layer | ⚠️ **The biggest risk.** FLOWER is flow-matching; SmolVLA uses an action expert. If there is no autoregressive decoder, "depth pruning" means something different from what it means on OpenVLA |

**Per-model fields to fill in before the runs:**

| Model | Checkpoint (HF path) | Decoder layers | Native chunk length | Action head type |
|---|---|---:|---:|---|
| TurboVLA | ? | ? | ? | ? |
| CoTinyVLA | ? | ? | ? | ? |
| FLOWER | ? | ? | ? | flow matching? |
| MiniVLA | ? | ? | ? | ? |
| SmolVLA | ? | ? | ? | action expert? |

> Parameter counts above are as given in the planning note and should be
> confirmed against each paper — some of these models ship in several sizes.

---

## 4. Two things worth deciding before the runs start

**(a) Both benchmarks, or just one?**

Our biggest open limitation is that **the benchmark axis rests on only two
backbones** (OpenVLA and SpatialVLA), because UniVLA is Bridge-only. If these
five models run on **both** Bridge and Fractal, that limitation largely
closes — the benchmark axis would rest on seven backbones instead of two.
If they run on Bridge only, the expansion makes the backbone axis richer but
leaves our weakest point untouched.

**Recommendation: both, if the compute allows.** If not, Bridge first, and
say explicitly in the paper that the benchmark axis did not grow.

**(b) Is this a "scale" axis or just "more backbones"?**

Tempting to say "we cover 0.2B to 8.5B." But these models differ in
architecture, training data, and action representation all at once — not just
size. So a difference between TurboVLA and OpenVLA **cannot be attributed to
parameter count.**

What the expansion legitimately buys:
- ✅ the backbone axis gets many more points, so "the sign is unstable across
  backbones" gets much harder to dismiss as a three-model artifact
- ✅ if the pattern holds at 0.2B as well as 8.5B, the claim clearly is not
  about model size
- ❌ it does **not** give a controlled scale study

Writing it the first way is defensible; writing it as a scale study is not.

---

## 5. Notebook check — what the four notebooks match, and what they do not

The four notebooks (`01_original_policy`, `02_fixed_foveation`,
`03_action_repeat`, `04_fixed_depth_pruning`) were compared function by
function against the code that actually produced `results/`. They are
**standalone reimplementations** — they do not import the eval scripts — so
this had to be checked directly.

**Last touched 2026-08-05. The campaign ran 08-06 through 08-12**, so some
later changes are not in them.

### Matches — safe to run as is

| Piece | Verdict |
|---|---|
| `foveate_image_logpolar` | **identical computation** to `adaptive_sparse_vla/foveation.py`; differences are type hints, variable names, line wrapping |
| `foveate_image_blur` | **identical**, including `frame.copy()` at `keep >= 1.0` and the exact-fovea write-back |
| `apply_action_repeat` | **identical semantics** — `np.repeat` gives `[a,a,b,b]`, matching `eval.py`'s `for a in actions for _ in range(k)` |
| depth-pruning ranking | **identical** to `depth_prune.py` — BI = 1 − cos, ascending, fraction-based window via `floor(min_layer * n)`, same greedy gap rule |
| per-episode records | `run_condition` keeps an `episodes` list — pairing is possible ✅ |

### Differences that need action before running

**① `reset_episode()` has no `ctrl` guard — this is the dangerous one.**

```python
# notebook                         # real code (depth_prune.py)
def reset_episode(self):           def reset_episode(self):
    self._done = False                 self.close_gripper_num = 0
                                       if not self.ctrl:
                                           return        # ← static mode: no re-rank
                                       self._ranking_ready = False
```

Our grid ran with `depth_ctrl: False`, so calibration happened **once per
run** and the layer set stayed fixed (Report §3.0). The notebook's version
re-calibrates unconditionally. It is currently **never called** — but the
name invites wiring it into an episode loop, and if that happens the new
models get **per-episode recalibration** while the existing grid has
per-run. That is a different experiment, and the two columns would not be
comparable.

**Action: either delete `reset_episode` from the notebook, or add the
`if not ctrl: return` guard.**

**② Field names differ — the output will not load into our tooling.**

| Notebook writes | Our records use |
|---|---|
| `trial` | **`ep_id`** ← this is the pairing key |
| `ms_per_call` | `model_ms_per_infer` |
| `ms_per_env_step` | `model_ms_per_env_step` |
| (nothing) | `grasped`, `final_info` |

`build_grid_report.py` pairs on `(task, ep_id)`. With `trial` instead, the
files will not be read at all. **Action: rename on write, or add an
adapter.**

**③ No last-layer protection.** The notebook implements the OpenVLA/UniVLA
convention only. SpatialVLA's implementation always protects the final layer
(Report §3.5.1 ②). For small models with few layers this matters more, not
less — our `window875` condition, which was forced to delete the last four
layers, took OpenVLA/Fractal to 0/25 on all three pick tasks. **Decide per
model whether the last layer is eligible, and record the decision.**

**④ No `episode_stats` / terminal `info`.** `run_episode` returns success,
steps, and timing but not the environment's terminal state. That is the
2026-08-10 change to `tome_spatialvla_eval.py` (Report §6.6), which post-dates
these notebooks. Without it, the §6 failure-type analysis (`wrong_object` vs.
`no_contact`) cannot be done for the new models. Acceptable for a first pass —
but say so rather than discovering it later.

**⑤ No GPU field.** Same gap as §7 ⑤. One line while the harness is already
being touched.

---

## 6. Questions for the meeting

1. Which of these five have **SimplerEnv** checkpoints (Bridge and/or
   Fractal)? Any that were only trained on LIBERO would need a different
   benchmark or would have to be dropped.
2. For **FLOWER** and **SmolVLA** — is there an autoregressive decoder that
   depth pruning can apply to, or does the action head make that intervention
   mean something different? If so, we may have to report depth pruning for a
   subset and say so.
3. What is each model's **native action chunk length**? This decides whether
   `action repeat` is comparable across all of them.
4. Both benchmarks or Bridge only (see §4 a)?
