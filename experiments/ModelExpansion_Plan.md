# Model Expansion — Run Table

Extending the backbone axis downward in parameter count. The current grid sits
at 4–8.5B; the five new models cover 0.2–4B.

**Policies (4)** = the four conditions of the existing grid:
**baseline (no intervention) · action repeat · foveation · depth pruning.**

**Benchmarks (2)** = both SimplerEnv suites: **WidowX-Bridge** (96 episodes)
and **Google Robot / Fractal** (135 episodes).

**Status.** `Done` = per-episode records exist under `results/`.

---

## Run table

| # | Model | Params | Benchmark | Policies | Status |
|---:|---|---:|---|:--:|---|
| 1 | OpenVLA | 7B | SimplerEnv — Bridge | 4 | **Done** |
| 2 | OpenVLA | 7B | SimplerEnv — Fractal | 4 | **Done** |
| 3 | SpatialVLA | 4B | SimplerEnv — Bridge | 4 | **Done** |
| 4 | SpatialVLA | 4B | SimplerEnv — Fractal | 4 | **Done** |
| 5 | UniVLA | 8.5B | SimplerEnv — Bridge | 4 | **Done** |
| 6 | UniVLA | 8.5B | SimplerEnv — Fractal | 4 | — *(no public Fractal checkpoint)* |
| 7 | TurboVLA | 0.2B | SimplerEnv — Bridge | 4 | Not |
| 8 | TurboVLA | 0.2B | SimplerEnv — Fractal | 4 | Not |
| 9 | CoTinyVLA | 0.9B | SimplerEnv — Bridge | 4 | Not |
| 10 | CoTinyVLA | 0.9B | SimplerEnv — Fractal | 4 | Not |
| 11 | FLOWER | 1B | SimplerEnv — Bridge | 4 | Not |
| 12 | FLOWER | 1B | SimplerEnv — Fractal | 4 | Not |
| 13 | MiniVLA | 1B | SimplerEnv — Bridge | 4 | Not |
| 14 | MiniVLA | 1B | SimplerEnv — Fractal | 4 | Not |
| 15 | SmolVLA | 4B | SimplerEnv — Bridge | 4 | Not |
| 16 | SmolVLA | 4B | SimplerEnv — Fractal | 4 | Not |

**To run: 10 cells × 4 policies = 40 runs** (~9,240 episodes).
**After completion the grid becomes 8 backbones × 2 benchmarks = 15 filled
cells**, up from the current 5.

> Parameter counts for the five new models are as provided; worth confirming
> against each paper, since several ship in more than one size.

---

## What each run should output

Our protocol pairs episodes by initial state and counts only the ones whose
outcome flipped, so **a summary success rate alone cannot enter the grid.**
Each run needs, per episode:

| Field | Why |
|---|---|
| `ep_id` | the pairing key — must match across the 4 policies |
| `success` | boolean |
| `steps` | determinism check |
| `model_ms_per_infer` / `model_ms_per_env_step` | the compute-saving column |

Format to match: any file under
`results/<campaign>/<condition>/<task>/results_<task>.json`.

If the GPU model can be written into the file as well, that closes a
limitation we currently have to disclose (Report §7 ⑤).

---

## Per-model information needed before the runs

| Model | Checkpoint (HF path) | Decoder layers | Native chunk length | Action head |
|---|---|---:|---:|---|
| TurboVLA | | | | |
| CoTinyVLA | | | | |
| FLOWER | | | | |
| MiniVLA | | | | |
| SmolVLA | | | | |

Two of these decide whether a policy is even applicable:

- **Native chunk length** — if a model natively emits action chunks, action
  repeat interacts with that. On our existing backbones this varied
  (OpenVLA 1, SpatialVLA 1, UniVLA 5), which is why chunk execution itself
  could not be used as a cross-backbone axis (RelatedWork §2.2 c ③).
- **Action head type** — depth pruning needs a stack of decoder layers with a
  computable Block Influence. If a model uses flow matching or a separate
  action expert, that intervention may mean something different than it does
  on OpenVLA.

Foveation applies to all of them — it touches the observation before the
encoder, not the model.

---

## Appendix — notebook check (2026-08-05 vs. the campaign code)

The four notebooks were compared function by function against the code that
produced `results/`. They are standalone reimplementations, not imports, so
this had to be checked directly.

**The interventions themselves match:**

| Piece | Verdict |
|---|---|
| `foveate_image_logpolar` | identical computation; differences are type hints and variable names |
| `foveate_image_blur` | identical, including `frame.copy()` at `keep >= 1.0` |
| `apply_action_repeat` | identical semantics — `np.repeat` gives `[a,a,b,b]` |
| depth-pruning ranking | identical — BI = 1 − cos, ascending, `floor(min_layer × n)` window, same gap rule |
| per-episode records | `run_condition` keeps an `episodes` list ✅ |

**Five things to fix before the runs:**

1. **`reset_episode()` has no `ctrl` guard.** The real code returns early in
   static mode, so calibration happens **once per run**. The notebook version
   resets unconditionally. It is currently never called, but wiring it into an
   episode loop would give **per-episode recalibration** — a different
   experiment from the existing grid.
2. **Field names differ.** The notebook writes `trial`, `ms_per_call`,
   `ms_per_env_step`; the grid tooling pairs on **`ep_id`** and reads
   `model_ms_per_infer` / `model_ms_per_env_step`. Files written with the
   notebook's names will not load at all.
3. **No last-layer protection.** The notebook implements the OpenVLA/UniVLA
   convention only; SpatialVLA always protects the final layer (§3.5.1 ②).
   Matters more on small models with few layers.
4. **No `episode_stats` / terminal `info`.** That harness change is from
   2026-08-10 (§6.6), after the notebooks. Without it the §6 failure-type
   analysis is not available for the new models.
5. **No GPU field.** Same as the request above.
