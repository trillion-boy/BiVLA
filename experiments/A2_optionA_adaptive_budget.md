# A-2 Option A — Unclamp the downstream budget (realize adaptive stopping)

## Problem (from the A-2 sweep)
Turning on AutoGaze adaptive stopping (`SELECTOR_TASK_LOSS=ε`) changed *which*
patches were chosen but **not how many** — `avg_selected_patches` stayed fixed at
16.7 for every ε. Reason: even when AutoGaze gazes fewer patches, the downstream
`_budget_grid_mask` re-selects a **fixed** `visible_ratio × grid²` count, so the
adaptive count (and its latency benefit) is neutralized.

## Change
`adaptive_sparse_vla/inference.py`, `AutoGazeSelectorController`:
1. `_score_gazing_mask` now also records `self._last_gaze_ratio` = AutoGaze's
   actual gazed fraction (gazed patches / total patches, from the binary masks).
2. `select()` — with env `AUTOGAZE_ADAPTIVE_BUDGET=1`, the downstream budget
   `dynamic_ratio` is replaced by `_last_gaze_ratio` (clamped to `[1 patch,
   original budget]`), so AutoGaze's adaptive count flows through instead of the
   fixed budget. Default (unset) keeps the original behavior.

## How to run (Colab)
Combine adaptive stopping (ε) **and** the unclamped budget:

```python
import os
os.environ["AUTOGAZE_ADAPTIVE_BUDGET"] = "1"
for eps in ["0.5", "0.7", "0.9"]:
    os.environ["SELECTOR_TASK_LOSS"] = eps
    run("widowx_carrot_on_plate", "shared", f"/content/results/carrot_A2A_eps{eps}", n=24)
os.environ.pop("SELECTOR_TASK_LOSS", None)
os.environ.pop("AUTOGAZE_ADAPTIVE_BUDGET", None)
```

## What to look at
- `avg_selected_patches` — should now **drop / vary with ε** (proof the adaptive
  count flows through; was stuck at 16.7)
- `avg_elapsed` — fewer patches → expect lower latency (the win the fixed budget hid)
- `success_rate` — does accuracy hold while latency drops? (the goal: maintain
  success at lower cost)
