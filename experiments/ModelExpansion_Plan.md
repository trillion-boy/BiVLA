# Model Expansion — Run Table

Extending the backbone axis downward in parameter count. The current grid sits
at 4–8.5B; the five new models cover 0.2–4B.

**Benchmarks (2)** = both SimplerEnv suites, for every model:
**WidowX-Bridge** (4 tasks × 24 = 96 episodes) and
**Google Robot / Fractal** (coke can 3 poses × 25 + move_near 60 = 135 episodes).

**Conditions (8)** = baseline plus the seven interventions. This is the row set
of the existing grid, and it is what has to be run per cell — a single
"foveation" or "depth pruning" run is not enough, because the axis is the
*setting* (k = 2 vs. 4, layers = 1 vs. 2 vs. 4), not the method.

**Status.** `Done` = per-episode records exist under `results/`.

---

## Run table

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

## What each run has to write out

Our protocol pairs episodes by **initial state** and counts only the ones whose
outcome flipped, so a summary success rate alone cannot enter the table above.
Each run needs one record per episode:

| Field | Why |
|---|---|
| `ep_id` | the pairing key — the **environment's** episode/init id, identical across all 8 conditions |
| `success` | boolean, from the env's own success flag |
| `steps` | determinism check |
| `model_ms_per_infer`, `model_ms_per_env_step` | the compute-saving column |
| `grasped` | from `episode_stats.is_src_obj_grasped`, for the failure-type analysis |

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

| Model | Checkpoint (HF path) | Decoder layers | Native chunk length | Action head |
|---|---|---:|---:|---|
| TurboVLA | | | | |
| CoTinyVLA | | | | |
| FLOWER | | | | |
| MiniVLA | | | | |
| SmolVLA | | | | |

Two of these decide whether a condition is even applicable:

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

## One consequence to expect

The multiple-comparison threshold is derived from how many tests we run. The
main grid goes from 5 × 7 = 35 to 15 × 7 = 105, so α tightens from ≈ 0.0013
to ≈ 0.0005.

Two of the eight cells that currently pass sit just above that line:

| Cell | current p | survives α ≈ 0.0005? |
|---|---:|---|
| OpenVLA/Fractal `depth prune 4` (**+15.6**) | 0.0011 | **no** |
| OpenVLA/Bridge `action repeat 4` (**−11.5**) | 0.0010 | **no** |

The first one is the cell we lean on for "saves compute and success goes up."
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
