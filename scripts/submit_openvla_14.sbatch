#!/usr/bin/env bash
# Four-worker OpenVLA 14-configuration synthetic throughput sweep.
#SBATCH --partition=cluster02
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=openvla-14
#SBATCH --output=slurm_logs/openvla_14_%j.out
#SBATCH --error=slurm_logs/openvla_14_%j.err

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla_tricks
PYTHON="$(command -v python)"
MODEL="$PWD/models/openvla-7b"
OUTPUT_ROOT="$PWD/artifacts/results/openvla_7b"
RUNNER=scripts/benchmark_openvla_14.py

run_worker() {
  local gpu="$1"
  local worker="$2"
  local configs="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --model "$MODEL" \
    --configs "$configs" \
    --output-root "$OUTPUT_ROOT" \
    --unnorm-key bridge_orig \
    --warmup 1 \
    --repeats 3 \
    --worker "$worker"
}

# One persistent model process per GPU; groups are balanced to overlap model
# loading and keep all four assigned RTX 5090s busy.
run_worker 0 worker0 "original,fixed_foveation_keep20,fixed_foveation_keep50,action_repeat2" &
run_worker 1 worker1 "action_repeat4,depth_pruning1,depth_pruning2" &
run_worker 2 worker2 "depth_pruning4,guarded_reuse_strict,guarded_reuse_moderate,guarded_reuse_aggressive" &
run_worker 3 worker3 "temporal_fusion_motion_entropy,temporal_fusion_task_aware,temporal_fusion_conservative_adaptive" &
wait
