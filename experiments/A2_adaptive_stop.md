# A-2 — AutoGaze Adaptive Stopping (UniVLA path)

## What & Why
AutoGaze can decide *how many* patches to select per frame by stopping once a
reconstruction-loss threshold ε is met (`task_loss_requirement`). The BiVLA
shared path disables this (`--selector-task-loss-requirement -1`) and forces a
fixed budget (`max-highres-patches 60`).

This experiment turns adaptive stopping **on** so the patch count varies per
scene — fewer patches on simple scenes (faster), more on complex ones. Unlike
A-3b, this affects the **grasp-phase** AutoGaze selection that the WidowX tasks
actually exercise, so it can change results.

## How it flows
`eval.py`: `--selector-task-loss-requirement` → `None if < 0 else value`
→ `PatchSelectionConfig.autogaze_task_loss_requirement`
→ `AutoGazeSelectorController.select(... task_loss_requirement=...)`
→ `AutoGaze.forward(gazing_ratio=visible_ratio, task_loss_requirement=ε)`.
With both set, `gazing_ratio` is the **max** budget and ε controls early stop.

No source change needed — only a CLI value. The repo's
`experiments/run_shared_compact_focus.sh` already exposes it via
`SELECTOR_TASK_LOSS_REQUIREMENT` (default -1).

## How to run (Colab, sweep ε)
Only `widowx_carrot_on_plate` engages AutoGaze, so sweep on carrot:

```python
import os
for eps in ["0.5", "0.7", "0.9"]:
    os.environ["SELECTOR_TASK_LOSS"] = eps
    run("widowx_carrot_on_plate", "shared", f"/content/results/carrot_A2_eps{eps}", n=24)
os.environ.pop("SELECTOR_TASK_LOSS", None)
```
(ε small → stricter recon → more patches; ε large → fewer patches.)

## What to look at
- `avg_selected_patches` — should now **vary with ε** (proof A-2 is active);
  hybrid(off) was 16.7 fixed
- `avg_elapsed` — fewer patches → expect lower latency (the latency win the
  method currently doesn't deliver)
- `success_rate` vs hybrid carrot (16/24 in our setup)
