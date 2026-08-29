#!/usr/bin/env bash
# Submit one independent MiniVLA experiment configuration.
# Usage: sbatch --export=EXPERIMENT=scripts/experiments/original.py this_file
#SBATCH --partition=cluster02
#SBATCH --qos=wanglin_2026_08
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=minivla-experiment
#SBATCH --output=slurm_logs/minivla_experiment_%j.out
#SBATCH --error=slurm_logs/minivla_experiment_%j.err

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

PYTHON="${PYTHON:-$(command -v python)}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT to one scripts/experiments/*.py path}"

echo "Starting experiment=${EXPERIMENT} job=${SLURM_JOB_ID}"
echo "Python=${PYTHON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$PYTHON" "$EXPERIMENT"
