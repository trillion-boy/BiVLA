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

Format to match: any file under
`results/<campaign>/<condition>/<task>/results_<task>.json`.

**`ep_id` is not a loop counter.** On Bridge it is passed to
`env.reset(options={"obj_init_options": {"episode_id": ep_id}})` and happens to
run 0–23. On Fractal the two task families differ:

- **move_near** accepts `episode_id` directly (0–59).
- **coke can** has no `episode_id` at all — the initial pose comes from an
  explicit 5 × 5 xy grid over `[−0.35, −0.12] × [−0.02, 0.42]`, indexed by
  `ep_id`, with the seed also set to `ep_id`.

`simpler_fractal_protocol.py` (`prepackaged_reset_options`) already does both.
If a new model's runner re-implements the reset itself, Fractal episodes will
not be paired and the whole column becomes unusable.

If the GPU model can go into the file too, that closes a limitation we
currently have to disclose (Report §7 ⑤).

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
docstrings, type annotations and variable names. The notebooks are standalone
reimplementations, not imports, so this had to be compared directly rather
than assumed.

**The interventions themselves are the same computation.**

| Piece | Verdict |
|---|---|
| `foveate_image_logpolar` | same — only `fwd`/`inv` vs. `forward_flags`/`inverse_flags`, `ys, xs` vs. `sample_ys, sample_xs` |
| `foveate_image_blur` | same — `out[dist <= r0] = frame[dist <= r0]` written in one line instead of two |
| `_uniform_sample_grid` | byte-identical after normalization |
| `find_decoder_layers`, `BypassDecoderLayer`, the hook | byte-identical after normalization |
| `measure_redundancy_with_hooks` | same — one `return` folded into a conditional expression |
| `rank_layers` | same rule — module-level args instead of `self.min_layer` / `self.min_gap` |
| `apply_action_repeat` | no package counterpart to diff against; `np.repeat` gives `[a,a,b,b]`, which is what the runs did |
| per-episode records kept | yes — `run_condition` keeps an `episodes` list |

**Five things differ, and all five change the experiment rather than the
code's appearance.**

1. **Field names.** The notebook writes `task` + `trial`, `ms_per_call`,
   `ms_per_env_step`. Our files carry `ep_id`, `model_ms_per_infer`,
   `model_ms_per_env_step`. `build_grid_report.py` pairs on `ep_id`, so files
   written with the notebook's names do not load at all — not "load wrong",
   do not load.
2. **`trial` is not `ep_id`.** `run_condition` loops `for trial in range(24)`
   and hands `(task, trial)` to a caller-supplied `adapter_factory`. Nothing
   in the notebook connects that number to the environment's initial state.
   On Bridge the two happen to coincide (0–23); on Fractal they do not, per
   the coke-can note above.
3. **`reset_episode()` re-calibrates.** The notebook's `StaticDepthPruner`
   says "calibrate once per episode" and its `reset_episode` clears `_done`.
   The campaign calibrated **once per run** — in static mode
   (`ctrl=False`) `depth_prune.py`'s `reset_episode` returns before touching
   `_calibrated`. Calling `reset_episode()` in an episode loop gives a
   per-episode-recalibration experiment, which is a different thing from
   what the grid contains.
4. **Layer-selection convention is OpenVLA/UniVLA only.** The notebook uses
   `min_layer` as a **ratio** (0.5 → back half) with a gap rule and no
   protected last layer. The SpatialVLA runs used
   `SpatialVLA/experiments/tome/depth_prune_gemma2.py`, where `min_layer` is a
   **count** (2 → skip layers 0–1), the final layer is always protected
   (`protected = set(range(min_layer)) | {n - 1}`), and there is **no gap
   rule**. For a Gemma-family or otherwise SpatialVLA-shaped small model,
   the notebook would pick a different layer set than our SpatialVLA columns
   did.
5. **No `episode_stats` / terminal `info` capture.** That harness change is
   dated 2026-08-10 (§6.6), after the notebooks. Without it there is no
   `grasped` field and the §6 failure-type analysis is unavailable for the
   new models.

Items 1 and 3 are a few lines each. Item 2 is a wiring decision the runner
has to make. Items 4 and 5 are only worth doing if the corresponding analysis
is wanted for the new models.
