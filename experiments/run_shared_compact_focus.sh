#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../configs/paths.sh"

GPU_ID="${GPU_ID:-0}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
N_EPISODES="${N_EPISODES:-24}"
EPISODE_IDS="${EPISODE_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_DIR}/shared_compact_focus/${TASK}}"
PATCH_GRID_SIZE="${PATCH_GRID_SIZE:-20}"
MAX_HIGHRES_PATCHES="${MAX_HIGHRES_PATCHES:-60}"
LOWRES_FACTOR="${LOWRES_FACTOR:-0.1}"
PATCH_GRASP_STEPS="${PATCH_GRASP_STEPS:-12}"
SELECTOR_HUB="${SELECTOR_HUB:-nvidia/AutoGaze}"
SELECTOR_TARGET_PATCH_SIZE="${SELECTOR_TARGET_PATCH_SIZE:-16}"
SELECTOR_TASK_LOSS_REQUIREMENT="${SELECTOR_TASK_LOSS_REQUIREMENT:--1}"
SELECTOR_DEVICE="${SELECTOR_DEVICE:-cuda}"
SELECTOR_REFRESH_STRIDE="${SELECTOR_REFRESH_STRIDE:-4}"
SPARSE_GATE_MODE="${SPARSE_GATE_MODE:-auto}"
LLM_PRUNE_COUNT="${LLM_PRUNE_COUNT:-2}"
LLM_PRUNE_MIN_LAYER="${LLM_PRUNE_MIN_LAYER:-0.75}"
LLM_PRUNE_MIN_GAP="${LLM_PRUNE_MIN_GAP:-1}"
LLM_PRUNE_LAYERS="${LLM_PRUNE_LAYERS:-}"
UNIFORM_GATE_DX_RATIO="${UNIFORM_GATE_DX_RATIO:-0.20}"
UNIFORM_GATE_DY_RATIO="${UNIFORM_GATE_DY_RATIO:-0.05}"
UNIFORM_GATE_AREA_MIN="${UNIFORM_GATE_AREA_MIN:-0.010}"
UNIFORM_GATE_AREA_MAX="${UNIFORM_GATE_AREA_MAX:-0.0135}"
FOCUS_GATE_MIN_CONFIDENCE="${FOCUS_GATE_MIN_CONFIDENCE:-0.65}"
FOCUS_GATE_MAX_CONFIDENCE="${FOCUS_GATE_MAX_CONFIDENCE:-0.78}"
FOCUS_GATE_MIN_CONTEXT_CONFIDENCE="${FOCUS_GATE_MIN_CONTEXT_CONFIDENCE:-0.55}"
SAVE_VIDEO_FLAG=""
DISABLE_VIDEO_HISTORY_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --n-episodes) N_EPISODES="$2"; shift 2 ;;
        --episode-ids) EPISODE_IDS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --patch-grid-size) PATCH_GRID_SIZE="$2"; shift 2 ;;
        --max-highres-patches) MAX_HIGHRES_PATCHES="$2"; shift 2 ;;
        --lowres-factor) LOWRES_FACTOR="$2"; shift 2 ;;
        --patch-grasp-steps) PATCH_GRASP_STEPS="$2"; shift 2 ;;
        --selector-hub) SELECTOR_HUB="$2"; shift 2 ;;
        --selector-target-patch-size) SELECTOR_TARGET_PATCH_SIZE="$2"; shift 2 ;;
        --selector-task-loss-requirement) SELECTOR_TASK_LOSS_REQUIREMENT="$2"; shift 2 ;;
        --selector-device) SELECTOR_DEVICE="$2"; shift 2 ;;
        --selector-refresh-stride) SELECTOR_REFRESH_STRIDE="$2"; shift 2 ;;
        --sparse-gate-mode) SPARSE_GATE_MODE="$2"; shift 2 ;;
        --llm-prune-count) LLM_PRUNE_COUNT="$2"; shift 2 ;;
        --llm-prune-min-layer) LLM_PRUNE_MIN_LAYER="$2"; shift 2 ;;
        --llm-prune-min-gap) LLM_PRUNE_MIN_GAP="$2"; shift 2 ;;
        --llm-prune-layers) LLM_PRUNE_LAYERS="$2"; shift 2 ;;
        --uniform-gate-dx-ratio) UNIFORM_GATE_DX_RATIO="$2"; shift 2 ;;
        --uniform-gate-dy-ratio) UNIFORM_GATE_DY_RATIO="$2"; shift 2 ;;
        --uniform-gate-area-min) UNIFORM_GATE_AREA_MIN="$2"; shift 2 ;;
        --uniform-gate-area-max) UNIFORM_GATE_AREA_MAX="$2"; shift 2 ;;
        --focus-gate-min-confidence) FOCUS_GATE_MIN_CONFIDENCE="$2"; shift 2 ;;
        --focus-gate-max-confidence) FOCUS_GATE_MAX_CONFIDENCE="$2"; shift 2 ;;
        --focus-gate-min-context-confidence) FOCUS_GATE_MIN_CONTEXT_CONFIDENCE="$2"; shift 2 ;;
        --save-video) SAVE_VIDEO_FLAG="--save-video"; shift ;;
        --disable-video-history) DISABLE_VIDEO_HISTORY_FLAG="--disable-video-history"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
source "${CONDA_PROFILE}"

RUN_CMD=(
    conda run -n "${BIVLA_CONDA_ENV}" --no-capture-output
    python "${BIVLA_ROOT}/adaptive_sparse_vla/eval.py"
    --emu-hub "${EMU_HUB}"
    --vq-hub "${VQ_HUB}"
    --fast-path "${FAST_PATH}"
    --task "${TASK}"
    --n-episodes "${N_EPISODES}"
    --output-dir "${OUTPUT_DIR}"
    --model-type shared_compact_focus
    --image-size 256
    --min-pixels 6400
    --device cuda
    --patch-grid-size "${PATCH_GRID_SIZE}"
    --max-highres-patches "${MAX_HIGHRES_PATCHES}"
    --lowres-factor "${LOWRES_FACTOR}"
    --patch-grasp-steps "${PATCH_GRASP_STEPS}"
    --selector-hub "${SELECTOR_HUB}"
    --selector-target-patch-size "${SELECTOR_TARGET_PATCH_SIZE}"
    --selector-task-loss-requirement "${SELECTOR_TASK_LOSS_REQUIREMENT}"
    --selector-device "${SELECTOR_DEVICE}"
    --selector-refresh-stride "${SELECTOR_REFRESH_STRIDE}"
    --sparse-gate-mode "${SPARSE_GATE_MODE}"
    --llm-prune-count "${LLM_PRUNE_COUNT}"
    --llm-prune-min-layer "${LLM_PRUNE_MIN_LAYER}"
    --llm-prune-min-gap "${LLM_PRUNE_MIN_GAP}"
    --uniform-gate-dx-ratio "${UNIFORM_GATE_DX_RATIO}"
    --uniform-gate-dy-ratio "${UNIFORM_GATE_DY_RATIO}"
    --uniform-gate-area-min "${UNIFORM_GATE_AREA_MIN}"
    --uniform-gate-area-max "${UNIFORM_GATE_AREA_MAX}"
    --focus-gate-min-confidence "${FOCUS_GATE_MIN_CONFIDENCE}"
    --focus-gate-max-confidence "${FOCUS_GATE_MAX_CONFIDENCE}"
    --focus-gate-min-context-confidence "${FOCUS_GATE_MIN_CONTEXT_CONFIDENCE}"
)
if [[ -n "${LLM_PRUNE_LAYERS}" ]]; then
    RUN_CMD+=(--llm-prune-layers "${LLM_PRUNE_LAYERS}")
fi
if [[ -n "${EPISODE_IDS}" ]]; then
    RUN_CMD+=(--episode-ids "${EPISODE_IDS}")
fi
if [[ -n "${SAVE_VIDEO_FLAG}" ]]; then
    RUN_CMD+=("${SAVE_VIDEO_FLAG}")
fi
if [[ -n "${DISABLE_VIDEO_HISTORY_FLAG}" ]]; then
    RUN_CMD+=("${DISABLE_VIDEO_HISTORY_FLAG}")
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
