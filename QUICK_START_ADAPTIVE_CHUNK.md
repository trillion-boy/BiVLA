# Quick Start: Adaptive Chunk Execution Experiments

**TL;DR:** Run adaptive chunk experiments on SpatialVLA with one command per task.

---

## Prerequisites

1. **Environment:** `bivla` conda environment with SpatialVLA + SimplerEnv
2. **Model:** SpatialVLA checkpoint (e.g., HuggingFace: `IPEC-COMMUNITY/spatialvla-4b-224-pt`)
3. **Branch:** `claude/serene-davinci-sy33re` with adaptive chunk code

---

## Step 1: Verify Installation

```bash
cd /home/user/BiVLA

# Check adaptive executor is importable
python -c "from SpatialVLA.experiments.tome.adaptive_chunk_exec import AdaptiveChunkExecutor; print('✓ AdaptiveChunkExecutor OK')"

# Check tests pass
python SpatialVLA/experiments/tome/test_adaptive_chunk_exec.py
# Expected: "ALL ADAPTIVE CHUNK TESTS PASS"
```

---

## Step 2: Run Baseline (Benchmark)

Establish baseline latency and success on one task:

```bash
cd /home/user/BiVLA

# Baseline (no chunk execution, full replanning every step)
python SpatialVLA/experiments/tome/tome_spatialvla_eval.py \
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --task widowx_put_eggplant_in_basket \
    --n-episodes 24 \
    --output-dir ./results/baseline

# Expected output (end of run):
#   성공률: X/24 = ...%
#   ms/infer: ~900 ms (model-only latency)
```

**Save results:** Note baseline success rate and latency for comparison.

---

## Step 3: Run Fixed k=2 (Reference)

```bash
# Fixed k=2: execute 2 of 4 predicted actions per generate
python SpatialVLA/experiments/tome/tome_spatialvla_eval.py \
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --task widowx_put_eggplant_in_basket \
    --exec-chunk 2 \
    --n-episodes 24 \
    --output-dir ./results/k2

# Expected:
#   성공률: ↑ from baseline (e.g., 66.7% → 87.5%)
#   ms/infer: ↓ 1.9× faster (e.g., 900 → 456 ms)
```

---

## Step 4: Run Fixed k=4 (Comparison)

```bash
# Fixed k=4: execute all 4 predicted actions before replanning
python SpatialVLA/experiments/tome/tome_spatialvla_eval.py \
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --task widowx_put_eggplant_in_basket \
    --exec-chunk 4 \
    --n-episodes 24 \
    --output-dir ./results/k4

# Expected:
#   성공률: ↑ high, but varies by task (eggplant: +20pp, stack: -6pp)
#   ms/infer: ↓ 3.6× faster (e.g., 900 → 235 ms)
```

---

## Step 5: Run Adaptive (Default)

```bash
# Adaptive: switches k based on gripper state
# Default: k=4 during transport, k=1 during grasp/place
python SpatialVLA/experiments/tome/tome_spatialvla_eval_adaptive.py \
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --task widowx_put_eggplant_in_basket \
    --adaptive-chunk \
    --n-episodes 24 \
    --output-dir ./results/adaptive_default

# Expected:
#   성공률: ≈ k=2 (e.g., 87.5%, recover from k=4 regression)
#   ms/infer: between k=2 and k=4 (e.g., 400–500 ms)
#   Latency tag: "adaptive-chunk k_sparse=4 k_dense=1"
```

---

## Step 6: Run Adaptive (Tuned, Optional)

If default params don't recover stack task losses:

```bash
# Adaptive with custom delays (more time in dense mode)
python SpatialVLA/experiments/tome/tome_spatialvla_eval_adaptive.py \
    --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
    --task widowx_put_eggplant_in_basket \
    --adaptive-chunk \
    --adaptive-close-delay 2 \
    --adaptive-open-delay 7 \
    --n-episodes 24 \
    --output-dir ./results/adaptive_tuned

# Compare with default to find best hyperparams
```

---

## Step 7: Compare Results

After running all configs, compare results:

```bash
python -c "
import json
import os

configs = {
    'baseline': './results/baseline',
    'k=2': './results/k2',
    'k=4': './results/k4',
    'adaptive': './results/adaptive_default',
}

print(f'Task: widowx_put_eggplant_in_basket')
print(f'{\"Config\":<15} {\"Success\":<10} {\"Grasp\":<10} {\"ms/infer\":<10}')
print('-' * 50)

for name, path in configs.items():
    json_path = os.path.join(path, 'results_widowx_put_eggplant_in_basket.json')
    with open(json_path) as f:
        data = json.load(f)
    sr = data['success_rate']
    gr = data['grasp_rate']
    ms = data['avg_model_ms_per_infer']
    print(f'{name:<15} {sr:<10.1%} {gr:<10.1%} {ms:<10.0f}')
"
```

---

## Step 8: Run Full Suite (All 4 Tasks)

Once adaptive is validated on one task, run full 4-task suite:

```bash
# Create result directories
mkdir -p results/full_suite/{baseline,k2,k4,adaptive}

tasks=(
    "widowx_put_eggplant_in_basket"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
    "widowx_spoon_on_towel"
)

# Baseline (if not already done)
for task in "${tasks[@]}"; do
    python SpatialVLA/experiments/tome/tome_spatialvla_eval.py \
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
        --task "$task" --n-episodes 24 \
        --output-dir results/full_suite/baseline/$task
done

# k=4
for task in "${tasks[@]}"; do
    python SpatialVLA/experiments/tome/tome_spatialvla_eval.py \
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
        --task "$task" --exec-chunk 4 --n-episodes 24 \
        --output-dir results/full_suite/k4/$task
done

# Adaptive
for task in "${tasks[@]}"; do
    python SpatialVLA/experiments/tome/tome_spatialvla_eval_adaptive.py \
        --model-path IPEC-COMMUNITY/spatialvla-4b-224-pt \
        --task "$task" --adaptive-chunk --n-episodes 24 \
        --output-dir results/full_suite/adaptive/$task
done
```

---

## Interpreting Results

### Success Rate (Primary Metric)

```
Baseline → k=4 → Adaptive
├─ Eggplant:  66.7% → 87.5% → 87.5%+   [✓ k=4 works, adaptive maintains]
├─ Carrot:    25.0% → 41.7% → 40%+     [✓ k=4 works, adaptive maintains]
├─ Spoon:      8.3% → 29.2% → 28%+     [✓ k=4 works, adaptive maintains]
└─ Stack:     31.3% → 25.0% → 31%+     [✗ k=4 fails, ✓ adaptive recovers]

Mean:         32.8% → 45.9% → 46%+
```

**Interpretation:**
- If `adaptive ≥ 45%`: adaptive successfully maintains k=4 gains
- If `adaptive on stack ≥ 31%`: adaptive recovers placement precision ✓

### Latency (Speedup)

```
Baseline: 900 ms/step
k=4:      235 ms/step  (3.8× faster)
Adaptive: 400 ms/step  (2.3× faster)  ← typical value

Speedup = 900 / 400 = 2.25× (between k=2's 1.9× and k=4's 3.8×)
```

**Interpretation:**
- If `speedup ≥ 1.8×`: adaptive is much faster than k=2 ✓
- If `speedup ≤ 3.0×`: adaptive is slower than k=4 (expected: trade accuracy for speed)

---

## Troubleshooting

### Error: "adaptive_chunk_exec module not found"

```bash
# Check branch has the file
git status
# Should show: SpatialVLA/experiments/tome/adaptive_chunk_exec.py

# If not, pull latest:
git pull origin claude/serene-davinci-sy33re
```

### Error: "gripper value not in action dict"

```bash
# Adaptive executor expects action["gripper"]
# If policy doesn't provide it, you'll get KeyError

# Check policy output:
python -c "
import torch
from simpler_env.policies.spatialvla import SpatialVLAInference
policy = SpatialVLAInference(saved_model_path='...')
raw_action, action = policy.step(image, instruction)
print('action keys:', action.keys())
print('gripper:', action.get('gripper', 'NOT FOUND'))
"
```

### Error: "chunk size mismatch"

```bash
# AdaptiveChunkExecutor assumes SpatialVLA's 4-action chunk
# If model predicts 3 or 5 actions per generate, you need to adjust:

executor = AdaptiveChunkExecutor(
    policy,
    k_sparse=4,  # ← change to match model's chunk_size
    k_dense=1,
)
```

---

## Performance Expectations by Task

| Task | Baseline | k=4 | Adaptive | Speedup |
|---|---|---|---|---|
| Eggplant | 66.7% | 87.5% | 87.5% | 2.0× |
| Carrot | 25.0% | 41.7% | 40% | 2.0× |
| Spoon | 8.3% | 29.2% | 28% | 2.0× |
| Stack | 31.3% | 25.0% | 31%+ | 1.5× |

**Interpretation:**
- **Eggplant, Carrot, Spoon:** Adaptive ≈ k=4 (sparse benefits all during transport)
- **Stack:** Adaptive > k=4 (dense mode during placement helps)
- **Speedup:** Adaptive is consistently 1.5–2.0×, between k=2 (1.9×) and k=4 (3.8×)

---

## Next Steps After Experiments

1. **If adaptive works** (stack ≥ 31%, mean ≥ 46%):
   - Run on **Google Robot tasks** for cross-robot validation
   - Run **UniVLA k-sweep** (k=1..5) for comparison

2. **If adaptive doesn't work** (stack still <31% or mean <45%):
   - Tune hyperparams: try `close_delay=1` or `open_delay=10`
   - Investigate: Is gripper state detection working? (add debug logging)
   - Consider: Does WidowX gripper have noisy readings?

3. **Prepare results for paper**:
   - Consolidate 4-task table
   - Add Google Robot results
   - Write narrative: "Adaptive chunk execution achieves X% success at Y× speedup"

---

## Reference: Full Parameter List

```bash
python tome_spatialvla_eval_adaptive.py --help

# Key adaptive parameters:
#   --adaptive-chunk                Enable adaptive executor
#   --adaptive-k-sparse INT         Chunk size during transport (default: 4)
#   --adaptive-k-dense INT          Chunk size during grasp/place (default: 1)
#   --adaptive-close-delay INT      Steps after gripper closes (default: 3)
#   --adaptive-open-delay INT       Steps after gripper opens (default: 5)

# Other useful parameters:
#   --n-episodes INT                Number of episodes per task (default: 24)
#   --output-dir PATH               Where to save results
#   --save-video                    Save GIF videos of episodes
```

---

## Estimated Runtime

**Per task (N=24 episodes):**
- Model inference: 24 episodes × 50 steps/episode × 0.9 s/step ≈ **18 minutes**
- Environment + policy overhead: ≈ 20% extra ≈ **4 minutes**
- **Total: ~20–25 minutes per task**

**Full 4-task suite (baseline + k=4 + adaptive):**
- 4 tasks × 3 configs × 25 min = **300 minutes = 5 hours**
- (Run in parallel on multiple GPUs if available)

---

## Questions?

Refer to:
- `ADAPTIVE_CHUNK_README.md` — Detailed documentation
- `NEXT_STEPS.md` — Implementation roadmap
- `GOOGLE_ROBOT_TASKS.md` — Cross-robot evaluation
