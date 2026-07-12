# Adaptive Chunk Execution for SpatialVLA

## Overview

Adaptive chunk execution varies the replanning frequency (`k`) based on the current manipulation phase, as detected from the gripper state. This combines the speed benefits of sparse replanning (k=4) with the precision benefits of dense replanning (k=1).

**Motivation:** Fixed k=4 chunk execution achieves 1.9× speedup and +13pp average success, but fails on placement-precision tasks (stack task regresses from 31.3% to 25%). Adaptive k keeps sparse replanning (3.6× speed) during transport and switches to dense replanning (baseline latency) when precision is critical.

## Architecture

```
AdaptiveChunkExecutor
  ├─ Monitors gripper_value from each action step
  ├─ Detects phase: gripper closed → precise phase, gripper open → transport
  ├─ Switches k dynamically via remove + reapply chunk_exec
  └─ Handles reset/mid-episode instruction changes
```

## Usage

### Command Line

```bash
python tome_spatialvla_eval_adaptive.py \
  --model-path <spatialvla-checkpoint> \
  --task widowx_put_eggplant_in_basket \
  --n-episodes 24 \
  --adaptive-chunk \
  --adaptive-k-sparse 4 \
  --adaptive-k-dense 1 \
  --adaptive-close-delay 3 \
  --adaptive-open-delay 5 \
  --output-dir ./results/adaptive_k
```

### Parameters

- `--adaptive-chunk`: Enable adaptive chunk execution (incompatible with `--exec-chunk`)
- `--adaptive-k-sparse`: Chunk size during transport/approach phases (default: 4)
- `--adaptive-k-dense`: Chunk size during grasp/place phases (default: 1)
- `--adaptive-close-delay`: Steps to wait after gripper closes before switching to dense (default: 3)
- `--adaptive-open-delay`: Steps to wait after gripper opens before switching back to sparse (default: 5)

### Python API

```python
from adaptive_chunk_exec import AdaptiveChunkExecutor

# Initialize
executor = AdaptiveChunkExecutor(
    policy,
    k_sparse=4,
    k_dense=1,
    close_steps_before_dense=3,
    open_steps_after_release=5,
)

# Each control step
for step in range(max_steps):
    action = policy.step(image, instruction)
    executor.step(action)  # Updates k based on gripper state
    obs, done, info = env.step(action)
    
    # At episode boundaries
    if done or instruction_changed:
        executor.reset()
```

## Phase Detection

Gripper state is extracted from `action["gripper"]`:
- `action["gripper"] ≤ 0.3`: gripper closed (precise phase)
- `action["gripper"] > 0.3`: gripper open (transport phase)

After detecting a state change, the executor waits for a configurable number of steps before switching k:

- **Gripper closes**: after `close_steps_before_dense` steps, switch to k=1 (dense)
- **Gripper opens**: after `open_steps_after_release` steps, switch to k=4 (sparse)

This hysteresis prevents thrashing and allows the phase decision to stabilize.

## Expected Results

Based on k=2 vs k=4 fixed comparison:

| Task | Baseline | k=4 | Adaptive (k=4→1) | Expected |
|---|---|---|---|---|
| Eggplant | 66.7% | 87.5% | 87.5%+ | +20pp with stable speed |
| Carrot | 25.0% | 41.7% | 40%+ | +15pp |
| Spoon | 8.3% | 29.2% | 28%+ | +20pp |
| **Stack** | 31.3% | **25.0%** | **31%+** | recover precision losses |
| **Mean** | **32.8%** | **45.9%** | **46%+** | sustain +13pp, fix stack |

**Latency trade-off:** Most transport time stays at 3.6× speedup (k=4); only precision moments (grasp/place) use full model (k=1). Expected per-episode speedup: **1.5–1.8×** (faster than k=2's 1.9× but better accuracy than k=4's 25% on stack).

## Testing

Run unit tests to verify phase detection:

```bash
python test_adaptive_chunk_exec.py
```

Expected output:
```
ok: adaptive starts with k_sparse=4
ok: gripper close triggers switch to dense after delay
ok: gripper open triggers switch to sparse after delay
ok: reset flushes queue and resets to sparse
ok: multiple grasp/release cycles work correctly

ALL ADAPTIVE CHUNK TESTS PASS
```

## Implementation Details

### Queue Management

Chunk execution maintains a queue of predicted actions. When k is switched, the queue must be flushed:
1. `remove_chunk_execution(policy)` → pop and discard queued actions
2. `apply_chunk_execution(policy, k=new_k)` → start fresh with new k

This ensures no stale actions from the old phase leak into the new one.

### Gripper State Hysteresis

The executor tracks `steps_since_gripper_change`:
- When state changes, reset counter to 0
- Increment counter each step where state is stable
- Switch k when counter ≥ delay threshold

This prevents oscillations if gripper noise causes momentary false closes/opens.

### Episode Reset and Instruction Changes

Both scenarios must call `executor.reset()` to:
- Flush any queued actions
- Reset gripper tracking state
- Reset to k_sparse for the fresh episode

The evaluation script handles this automatically.

## Next Steps

1. **Run 4-task suite** with default adaptive parameters
2. **Sweep hyperparameters** (close_delay, open_delay) to find optimal balance
3. **Compare vs fixed k** variants (k=2, k=4) and **vs baseline** (k=1)
4. **Analyze per-task failures** to refine phase detection if needed
5. **Cross-validate on new environments** (UniVLA paper tasks) to test generalization

## Known Limitations

- **Phase detection is gripper-only**: does not account for task-specific precision moments unrelated to gripper state (e.g., pre-placement alignment)
- **Hysteresis delays are fixed**: per-task tuning may be needed for optimal trade-offs
- **No multi-stage tasks yet**: assumes single grasp-place cycle; extended tasks (multiple grasps) untested
- **Assumes 4-action chunk**: hardcoded for SpatialVLA; adaptation needed for other chunk sizes

## Related Work

- **Chunk execution (fixed k=2, k=4)**: `chunk_exec.py`, `test_chunk_exec.py`, `tome_spatialvla_eval.py`
- **Task-aware control**: `shared_unified_policy.py` (original task-phase controller from BiVLA)
- **Comparable work**: OpenVLA foveation + action reuse (log-polar visual transform + grasp-triggered frame skip)
