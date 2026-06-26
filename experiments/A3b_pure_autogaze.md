# A-3b — Pure AutoGaze Ablation (UniVLA path)

## What & Why
The UniVLA shared-controller path mixes the AutoGaze saliency with hand-crafted
priors (color / gripper / temporal / bridge) inside `_hybrid_grid_mask`. In the
default code AutoGaze contributes only ~22–28 % of the final patch selection.

This ablation tests the question: **is the learned AutoGaze signal alone as good
as (or better than) the hand-crafted hybrid?** If yes, the method becomes cleaner
and more robust (color priors are sensitive to lighting/appearance shifts); if no,
it shows the priors are genuinely needed in this domain.

## Change
`adaptive_sparse_vla/inference.py`, `AutoGazeSelectorController._hybrid_grid_mask`:
added an env toggle. With `AUTOGAZE_PURE=1`, **all phases** use pure AutoGaze
saliency (`_budget_grid_mask(autogaze_score, ...)`), bypassing the prior mixing.
Default (unset) keeps the original hybrid behavior unchanged.

```python
if os.environ.get("AUTOGAZE_PURE", "0") == "1" or phase == "grasp":
    return self._budget_grid_mask(autogaze_score, visible_ratio), autogaze_score
```

## How to run (Colab)
Only `widowx_carrot_on_plate` actually engages AutoGaze, so compare carrot:

```python
import os
os.environ["AUTOGAZE_PURE"] = "1"
run("widowx_carrot_on_plate", "shared", "/content/results/carrot_A3b", n=24)
os.environ.pop("AUTOGAZE_PURE", None)   # reset to default
```

Baseline for comparison: carrot shared (hybrid) = 16/24 in our setup
(GPU/hardware-dependent; README hybrid = 19/24).

## What to look at
- `success_rate` vs the hybrid carrot run (16/24)
- `avg_selected_patches`, `router_profiles` (should still show `compact_focus_adaptive`)
- pure-AutoGaze ≥ hybrid → AutoGaze alone suffices (cleaner method)
- pure-AutoGaze < hybrid → color/position priors are needed here
