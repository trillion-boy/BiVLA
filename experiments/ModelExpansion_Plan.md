# Model Expansion — Run Table

Extending the backbone axis downward in parameter count. The current grid sits
at 4–8.5B; the five new models cover 0.2–4B.

**Benchmarks.** Two families, six suites in total:

| family | suites | episodes per run |
|---|---|---:|
| **SimplerEnv** | WidowX-Bridge (4 tasks × 24) | 96 |
| | Google Robot / Fractal (coke can 3 poses × 25 + move_near 60) | 135 |
| **LIBERO** | Spatial, Object, Goal, Long (= LIBERO-10) | 50 each |

**Conditions (8)** = baseline plus the seven interventions. This is the row set
of the existing grid, and it is what has to be run per cell — a single
"foveation" or "depth pruning" run is not enough, because the axis is the
*setting* (k = 2 vs. 4, layers = 1 vs. 2 vs. 4), not the method.

**Status.** `Done` = per-episode records exist under `results/`.

---

## SimplerEnv run table

Rows are conditions, columns are cells (model × benchmark). Values are the
**change in success rate against that column's own baseline, measured on
paired episodes**. Bold = passes the current multiple-comparison threshold
(α ≈ 0.0013). `Not` = to be run.

| | OpenVLA<br>Bridge | OpenVLA<br>Fractal | SpatialVLA<br>Bridge | SpatialVLA<br>Fractal | UniVLA<br>Bridge | UniVLA<br>Fractal | TurboVLA<br>Bridge | TurboVLA<br>Fractal | CoTinyVLA<br>Bridge | CoTinyVLA<br>Fractal | FLOWER<br>Bridge | FLOWER<br>Fractal | MiniVLA<br>Bridge | MiniVLA<br>Fractal | SmolVLA<br>Bridge | SmolVLA<br>Fractal |
|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *params* | *7B* | *7B* | *4B* | *4B* | *8.5B* | *8.5B* | *0.2B* | *0.2B* | *0.9B* | *0.9B* | *1B* | *1B* | *1B* | *1B* | *4B* | *4B* |
| *decoder layers* | *32* | *32* | *26* | *26* | *32* | *32* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* | *?* |
| *baseline success* | *15.6%* | *38.5%* | *30.2%* | *84.4%* | *81.2%* | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 2 | −8.3 | +5.2 | +12.5 | ±0.0 | **−69.8** | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 4 | **−11.5** | −1.5 | −12.5 | **−40.0** | **−81.2** | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation log-polar | +18.8 | **−19.3** | −8.3 | +0.7 | +5.2 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation blur | +17.7 | −8.9 | ±0.0 | −1.5 | −8.3 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 1 | +2.1 | +0.7 | −10.4 | +8.1 | −3.1 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 2 | ±0.0 | ±0.0 | −9.4 | +3.0 | −4.2 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 4 | +1.0 | **+15.6** | **−28.1** | **−17.8** | −2.1 | — | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |

UniVLA/Fractal is `—` throughout: there is no public Fractal checkpoint, so
that column cannot be filled by running anything.

**To run: 10 cells × 8 conditions = 80 runs.**
Per new model: 8 × 96 (Bridge) + 8 × 135 (Fractal) = 1,848 episodes.
Five models = **9,240 episodes.**

> Parameter counts for the five new models are as given; worth confirming
> against each paper, since several ship in more than one size.

### Optional extra rows

Two conditions exist in the current data but not in every cell, so they are
not grid rows. Adding them for the new models is optional, and only worth it
if the base 8 come back looking like the existing cells.

| Condition | Where it was run | Value |
|---|---|---:|
| depth prune 8 | OpenVLA / Bridge | −18.8 |
| depth prune 8 | UniVLA / Bridge | −79.2 |
| depth prune 2 + action repeat 2 | SpatialVLA / Fractal | −1.5 |

These three are why the current test family is **38**, not 35.

---

## LIBERO run table

Four suites, and one naming point first: in LIBERO, **Long and LIBERO-10 are
the same suite** (`libero_10`, the ten long-horizon tasks). So "Spatial,
Object, Long, 10" is four names for three suites unless the fourth is
**Goal** — which is also the reading that matches what is actually
downloadable, since OpenVLA released exactly four LIBERO checkpoints:
spatial, object, goal, 10. The table below assumes Spatial / Object / Goal /
Long(=10); if the intent was three suites, drop the Goal columns.

Rows are the same eight conditions as SimplerEnv, so the two families stay
comparable. 50 episodes per run (10 tasks × 5 initial states) — see the note
on protocol below.

| | TurboVLA<br>Spatial | TurboVLA<br>Object | TurboVLA<br>Goal | TurboVLA<br>Long | CoTinyVLA<br>Spatial | CoTinyVLA<br>Object | CoTinyVLA<br>Goal | CoTinyVLA<br>Long | FLOWER<br>Spatial | FLOWER<br>Object | FLOWER<br>Goal | FLOWER<br>Long | MiniVLA<br>Spatial | MiniVLA<br>Object | MiniVLA<br>Goal | MiniVLA<br>Long | SmolVLA<br>Spatial | SmolVLA<br>Object | SmolVLA<br>Goal | SmolVLA<br>Long |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *params* | *0.2B* | *0.2B* | *0.2B* | *0.2B* | *0.9B* | *0.9B* | *0.9B* | *0.9B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *1B* | *4B* | *4B* | *4B* | *4B* |
| *LIBERO checkpoint?* | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| *baseline success* | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 2 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| action repeat 4 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation log-polar | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| foveation blur | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 1 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 2 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |
| depth prune 4 | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not | Not |

**To run: 20 cells × 8 conditions = 160 runs**, 8,000 episodes at 50 per run.

### The checkpoint row is the gate, and it is per suite

**LIBERO cannot be run zero-shot.** Success collapses to roughly 0 without a
LIBERO fine-tune, and each of OpenVLA's four checkpoints is trained on **one
suite only** — pairing a checkpoint with a different suite measures nothing.
So a model needs up to four separate released checkpoints to fill its four
columns, and that is the first thing to check for each of the five, before
any scheduling.

This is not hypothetical: it is why our own LIBERO work used OpenVLA rather
than SpatialVLA. SpatialVLA reports LIBERO numbers but never released those
weights — the `IPEC-COMMUNITY` org ships only `224-pt`, `mix-224-pt`,
`sft-bridge`, `sft-fractal`. A model in the same position can appear on the
SimplerEnv table and be absent from this one.

Where a model ships fewer than four, run the columns it has and mark the rest
`—` rather than substituting a checkpoint from another suite.

### What we already have on LIBERO

Only `libero_spatial`, only two backbones, and a condition set that does not
line up with the SimplerEnv grid (no repeat 4, no prune 1/2; prune 8 and the
phase-adaptive controller instead). Reference, not a grid row:

| condition | OpenVLA | UniVLA |
|---|---:|---:|
| *baseline* | *74.0%* | *96.0%* |
| action repeat 2 | −8.0 | **−68.0** |
| foveation blur 20% | −16.0 | −2.0 |
| foveation log-polar 20% | **−74.0** | −8.0 |
| depth prune 4 | −18.0 | −10.0 |
| depth prune 8 | −46.0 | −10.0 |
| depth controller 2→8 | −24.0 | ±0.0 |

Worth keeping in view for two reasons. First, the same interventions behave
differently here than on SimplerEnv — log-polar took OpenVLA to 0.0% on
`libero_spatial` while *helping* it on Bridge (+18.8) — so LIBERO is not a
redundant copy of the SimplerEnv columns. Second, our OpenVLA baseline of
74.0% sits 10.7 points under the published 84.7% on the same suite, a
reproducible gap we never explained. Differences measured with that gap held
fixed are still usable; absolute numbers are not comparable to the
literature. The same check is worth running for each new model before its
intervention rows are trusted.

### Protocol: 50 episodes or 500?

Ours is 10 tasks × 5 initial states = **50 per run**. OpenVLA's reference eval
is 10 tasks × 50 trials = **500**, which is what the published numbers come
from. At 50, a 4-point difference on one suite is inside the noise; the
numbers above are only useful because the drops are large.

Both are defensible, and the choice is the mentor's, but it should be the
same for every model and every suite in the table. At 500 the LIBERO half of
this expansion is **80,000 episodes** rather than 8,000 — worth deciding
before rather than after.

### Pairing on LIBERO

Simpler than SimplerEnv, and worth saying because the two are not the same
mechanism. LIBERO exposes an explicit array of initial states per task, and
`env.set_init_state(init_states[i])` fully determines the episode. So the
pairing key is **(`task_id`, `trial`)**, where `trial` indexes that array —
here a plain counter *is* the right thing, unlike the SimplerEnv coke-can
tasks. Both numbers have to be in every record.

Everything else in the next section applies unchanged.

---

## What each run has to write out

Our protocol pairs episodes by **initial state** and counts only the ones whose
outcome flipped, so a summary success rate alone cannot enter the table above.
Each run needs one record per episode:

| Field | Why |
|---|---|
| `ep_id` (SimplerEnv) | the pairing key — the **environment's** episode/init id, identical across all 8 conditions |
| `task_id` + `trial` (LIBERO) | the pairing key there; `trial` indexes the suite's `init_states` array |
| `success` | boolean, from the env's own success flag |
| `steps` | determinism check |
| `model_ms_per_infer`, `model_ms_per_env_step` | the compute-saving column |
| `grasped` | from `episode_stats.is_src_obj_grasped`, for the failure-type analysis (SimplerEnv only) |

Format to match: `results/<campaign>/<condition>/<task>/results_<task>.json`.
Notebook 01's `run_condition(..., out_dir=...)` writes exactly this, so it
does not have to be assembled by hand.

**`ep_id` is not a loop counter.** On Bridge it is passed to
`env.reset(options={"obj_init_options": {"episode_id": ep_id}})` and happens to
run 0–23. On Fractal the two task families differ:

- **move_near** accepts `episode_id` directly (0–59).
- **coke can** has no `episode_id` at all — the initial pose comes from an
  explicit 5 × 5 xy grid over `[−0.35, −0.12] × [−0.02, 0.42]`, indexed by
  `ep_id`, with the seed also set to `ep_id`.

Notebook 01 carries both mappings (`EPISODE_IDS` and `reset_options`). If a
new model's runner re-implements the reset itself, Fractal episodes will not
be paired and the whole column becomes unusable.

The GPU name is now written into the file as well, which closes a limitation
we currently have to disclose (Report §7 ⑤).

---

## Per-model information needed before the runs

| Model | SimplerEnv checkpoint | LIBERO checkpoints (which suites) | Decoder layers | Native chunk length | Action head |
|---|---|---|---:|---:|---|
| TurboVLA | | | | | |
| CoTinyVLA | | | | | |
| FLOWER | | | | | |
| MiniVLA | | | | | |
| SmolVLA | | | | | |

The LIBERO column decides how many of that model's four LIBERO columns can be
filled at all — see the gate above. The two after it decide whether a
condition is even applicable:

- **Native chunk length** — if a model natively emits action chunks, action
  repeat interacts with that. On our existing backbones this varied
  (OpenVLA 1, SpatialVLA 1, UniVLA 5), which is why chunk execution itself
  could not be used as a cross-backbone axis (RelatedWork §2.2 c ③).
- **Action head type** — depth pruning needs a stack of decoder layers with a
  computable Block Influence. If a model uses flow matching or a separate
  action expert, that intervention means something different than it does on
  OpenVLA. This is likely for FLOWER.

Foveation applies to all five — it touches the observation before the encoder,
not the model.

**Decoder layer count matters more than usual here.** `depth prune 4` on a
32-layer stack removes 12.5% of it; on a small model with 12–18 layers it
removes 22–33%. The row is only comparable across columns if the layer counts
go in the table.

---

## Totals

| | cells | runs | episodes |
|---|---:|---:|---:|
| SimplerEnv (10 new cells × 8) | 10 | 80 | 9,240 |
| LIBERO (20 new cells × 8, 50 ep) | 20 | 160 | 8,000 |
| **total** | **30** | **240** | **17,240** |

At OpenVLA's own LIBERO protocol (500 per run) the second row becomes 80,000
and the total 89,240. That single choice is roughly a 5× swing in the whole
campaign, which is why it is worth settling first.

---

## One consequence to expect

The multiple-comparison threshold is derived from how many tests we run, so
adding cells tightens it for the cells we already have.

| family | intervention tests | α |
|---|---:|---:|
| now (5 SimplerEnv cells × 7) | 35 | 0.0013 |
| + 10 SimplerEnv cells | 105 | 0.0005 |
| + 20 LIBERO cells, if pooled into one family | 245 | 0.0002 |

Whether LIBERO joins the same family is a judgement call rather than a fact:
it is a different benchmark family asking the same question, so treating it
as a second family with its own α is defensible. Pooling is the conservative
choice, and it is the one that costs us something.

Two of the eight cells that currently pass sit just above the first line
already:

| Cell | current p | α ≈ 0.0005 | α ≈ 0.0002 |
|---|---:|---|---|
| OpenVLA/Fractal `depth prune 4` (**+15.6**) | 0.0011 | **no** | **no** |
| OpenVLA/Bridge `action repeat 4` (**−11.5**) | 0.0010 | **no** | **no** |

The first is the cell we lean on for "saves compute and success goes up."
It does not become wrong — the estimate is unchanged — but it stops being
callable as corrected-significant once the grid is this wide. Worth saying out
loud before the runs rather than after.

---

## Appendix — do the four notebooks match the code that produced `results/`?

Checked by parsing both sides and comparing function bodies after removing
docstrings, type annotations and variable names, because the notebooks are
standalone reimplementations rather than imports.

**The interventions themselves were already the same computation.**

| Piece | Verdict |
|---|---|
| `foveate_image_logpolar` | same — only `fwd`/`inv` vs. `forward_flags`/`inverse_flags`, `ys, xs` vs. `sample_ys, sample_xs` |
| `foveate_image_blur` | same — `out[dist <= r0] = frame[dist <= r0]` written in one line instead of two |
| `_uniform_sample_grid` | identical after normalization |
| `find_decoder_layers`, `BypassDecoderLayer`, the redundancy hook | identical after normalization |
| `measure_redundancy_with_hooks` | same — one `return` folded into a conditional expression |
| `rank_layers` | same rule — module-level args instead of `self.min_layer` / `self.min_gap` |
| `apply_action_repeat` | `np.repeat` gives `[a,a,b,b]`, which is what the runs did |

**Five things did not match, and all five have now been fixed in the
notebooks.** They are listed because each was a way to produce a run that
looks fine and is not comparable — worth knowing about even now that the
code no longer does it.

| # | What was wrong | What it would have caused | Fix |
|---|---|---|---|
| 1 | records written as `trial` / `ms_per_call` / `ms_per_env_step` | the grid tooling pairs on `ep_id`; those files do not load at all | `run_episode` now emits `ep_id`, `success`, `steps`, `elapsed`, `model_ms_per_infer`, `model_ms_per_env_step` |
| 2 | `trial` was a loop counter with nothing tying it to the env's initial state | Bridge coincides by accident; **coke can does not** — two conditions scored on different scenes while looking paired | `EPISODE_IDS` + `reset_options()` carry both mappings, including the 5 × 5 placement grid |
| 3 | `reset_episode()` cleared the calibration | per-episode recalibration — a different experiment from the grid, invisible in the output file | calibration is once per run; `reset_episode()` is a no-op unless `recalibrate_each_episode=True` |
| 4 | only the OpenVLA/UniVLA layer window (ratio, gap rule, last layer eligible) | a SpatialVLA-shaped model would be cut differently than our SpatialVLA columns were | `rank_layers(..., window="count", min_gap=0, protect_last=True)` gives SpatialVLA's convention; both are documented side by side |
| 5 | no `episode_stats`, no `grasped`, no GPU field | no failure-type analysis for the new models; latency caveat stays open | all three are captured; `grasped` is `None` rather than `False` when the env does not report it |

The notebooks also now **write the files themselves.** `run_condition(...,
out_dir=...)` produces
`<out_dir>/<condition>/<task>/results_<task>.json` — the layout
`build_grid_report.py` walks — so nothing has to be reformatted afterwards.

Notebook 01 ends by writing two conditions to a temporary directory and
pairing them back the way the analysis does (index by `ep_id`, count only the
episodes whose outcome flipped). That check runs with no simulator and no
checkpoint, so the file format can be confirmed before any GPU time is spent.

All four notebooks still execute top to bottom on stubs — no simulator, no
checkpoint — and each ends in assertions rather than eyeballing. 04's now
include that the two layer windows really do select different layers and that
`reset_episode()` leaves the calibration alone by default.
