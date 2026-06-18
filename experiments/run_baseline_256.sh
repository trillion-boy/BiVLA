#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../configs/paths.sh"

GPU_ID="${GPU_ID:-0}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
N_EPISODES="${N_EPISODES:-24}"
EPISODE_IDS="${EPISODE_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/baseline_256/${TASK}}"
SAVE_VIDEO_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU_ID="$2"; shift 2 ;;
        --task)        TASK="$2"; shift 2 ;;
        --n-episodes)  N_EPISODES="$2"; shift 2 ;;
        --episode-ids) EPISODE_IDS="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --save-video)  SAVE_VIDEO_FLAG="--save-video"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
source "${CONDA_PROFILE}"

echo "======================================================"
echo "  Adaptive Sparse VLA — Baseline 256x256"
echo "  gpu        : ${GPU_ID}"
echo "  task       : ${TASK}"
echo "  n_episodes : ${N_EPISODES}"
echo "  output_dir : ${OUTPUT_DIR}"
echo "======================================================"

RUN_CMD=(
    conda run -n "${BIVLA_CONDA_ENV}" --no-capture-output
    python "${BIVLA_ROOT}/adaptive_sparse_vla/eval.py"
    --emu-hub "${EMU_HUB}"
    --vq-hub "${VQ_HUB}"
    --fast-path "${FAST_PATH}"
    --task "${TASK}"
    --n-episodes "${N_EPISODES}"
    --output-dir "${OUTPUT_DIR}"
    --model-type baseline
    --image-size 256
    --min-pixels 6400
    --device cuda
)
if [[ -n "${EPISODE_IDS}" ]]; then
    RUN_CMD+=(--episode-ids "${EPISODE_IDS}")
fi

if [[ -n "${SAVE_VIDEO_FLAG}" ]]; then
    RUN_CMD+=("${SAVE_VIDEO_FLAG}")
fi

if command -v xvfb-run >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_GL=osmesa \
        VK_ICD_FILENAMES="${BIVLA_VK_ICD}" \
        xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
        "${RUN_CMD[@]}"
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" MUJOCO_GL=osmesa \
        VK_ICD_FILENAMES="${BIVLA_VK_ICD}" \
        "${RUN_CMD[@]}"
fi

echo ""
echo "======================================================"
echo "  Done: ${OUTPUT_DIR}"
echo "======================================================"
