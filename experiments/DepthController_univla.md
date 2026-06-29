# Phase-adaptive depth controller (UniVLA) — non-uniform depth over time

**Idea.** Static layer pruning trades success for latency uniformly. But our sweep
showed the failures from aggressive pruning concentrate at the **grasp moment**
(`ever_grasped_src=false`), while free-space transport tolerates it. So allocate
depth **non-uniformly in time**: keep full depth for the precise approach+grasp,
prune aggressively once the object is in hand. This is the project's non-uniform
thesis applied to the **depth** axis (the foveation principle, in compute).

## Design (training-free, no external module)
- **Calibrate once.** On the first step, one forward pass gives each layer's
  redundancy (`1 - cos(layer_in, layer_out)`); we store the full eligible layer
  ranking (most-redundant first). Nested prefixes → `deep ⊂ shallow`.
- **Two states, one-way.**
  - `deep` (precise): bypass the top `DEPTH_CTRL_DEEP` layers (default 2).
  - `shallow` (fast): bypass the top `DEPTH_CTRL_SHALLOW` layers (default 8).
  - Transition **deep → shallow only**, when the policy's own gripper has been
    commanded closed for `DEPTH_CTRL_CLOSE_STEPS` consecutive steps (default 2) —
    a fast, unambiguous, policy-internal signal (`self.close_gripper_num`). One-way
    means no oscillation; transport/place stay shallow.
- Switching states just re-bypasses top-N from the stored ranking (module swap,
  **no re-calibration**). Resets to `deep` each episode.

## Why the transition is clean
The grasp signal is the gripper close command the policy already emits every step
(no env ground truth, no detector). It is binary and fast (K≈2 steps of
hysteresis), and the switch is monotonic, so the controller can't flip-flop. The
precise window (approach+grasp) — exactly where depth pruning hurt — keeps full
depth; the forgiving window (transport) runs cheap.

## Implementation
`adaptive_sparse_vla/inference.py`:
- `_compute_layer_importance` (one forward, cosine redundancy),
- `_rank_prune_layers` (full nested ranking),
- `_depth_apply_state` (bypass top-N for the current state),
- `_maybe_calibrate_llm_pruning` (depth branch: rank once, apply deep),
- `step()` (one-way grasp-triggered switch to shallow),
- `reset()` (back to deep + re-calibrate next episode).

Gated by env (works on the `baseline` model type):
`DEPTH_CTRL_ENABLE=1`, `DEPTH_CTRL_DEEP`, `DEPTH_CTRL_SHALLOW`, `DEPTH_CTRL_CLOSE_STEPS`.

## How to run (Colab, `bivla` env)
```python
# baseline reference (no pruning) already at eg_base_n8.
# static prune-8 (uniform) for comparison:
import subprocess, os
subprocess.run(["bash","/content/run_univla.sh","widowx_put_eggplant_in_basket",
                "baseline","/content/results/eg_static8"],
               env={**os.environ,"N_EPISODES":"8","LLM_PRUNE_COUNT":"8"}, check=False)

# phase-adaptive depth controller: deep(prune 2) until grasp, shallow(prune 8) after
subprocess.run(["bash","/content/run_univla.sh","widowx_put_eggplant_in_basket",
                "baseline","/content/results/eg_depthctrl"],
               env={**os.environ,"N_EPISODES":"8",
                    "DEPTH_CTRL_ENABLE":"1","DEPTH_CTRL_DEEP":"2",
                    "DEPTH_CTRL_SHALLOW":"8","DEPTH_CTRL_CLOSE_STEPS":"2"}, check=False)
```

### What to look at (the hypothesis)
- **static prune-8**: success drops (~62% on eggplant), ms/infer ~1.33×.
- **depth controller**: success should **recover toward baseline** (full depth at
  grasp) while ms/infer stays well below baseline (shallow during transport) — i.e.
  *better success than static-8 at comparable latency*. That gap is the
  contribution.
- Also run **stack_cube** (precise placement): if the one-way switch hurts the
  place, that motivates a 3-state variant (deep → shallow → deep-for-place); the
  2-state is the first, cleanest version.

### Tuning
`DEPTH_CTRL_DEEP` (0–2), `DEPTH_CTRL_SHALLOW` (6–10), `DEPTH_CTRL_CLOSE_STEPS`
(1–3). More aggressive shallow = faster but riskier if transport needs precision.
