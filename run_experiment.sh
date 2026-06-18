#!/usr/bin/env bash
set -euo pipefail

BIVLA_ROOT="$(cd "$(dirname "$0")" && pwd)"
source "${BIVLA_ROOT}/configs/paths.sh"

GPU_ID="${GPU_ID:-0}"
EXP="${EXP:-adaptive_sparse}"
TASK=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            PASS_ARGS+=("--gpu" "$2")
            shift 2
            ;;
        --exp)
            EXP="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            PASS_ARGS+=("--task" "$2")
            shift 2
            ;;
        --n-episodes)
            PASS_ARGS+=("--n-episodes" "$2")
            shift 2
            ;;
        --episode-ids)
            PASS_ARGS+=("--episode-ids" "$2")
            shift 2
            ;;
        --output-dir)
            PASS_ARGS+=("--output-dir" "$2")
            shift 2
            ;;
        --patch-grid-size|--max-highres-patches|--lowres-factor|--selector-hub|--selector-target-patch-size|--selector-task-loss-requirement|--selector-device|--selector-refresh-stride|--patch-grasp-steps|--sparse-gate-mode|--llm-prune-count|--llm-prune-min-layer|--llm-prune-min-gap|--llm-prune-layers|--uniform-gate-dx-ratio|--uniform-gate-dy-ratio|--uniform-gate-area-min|--uniform-gate-area-max|--focus-gate-min-confidence|--focus-gate-max-confidence|--focus-gate-min-context-confidence)
            PASS_ARGS+=("$1" "$2")
            shift 2
            ;;
        --save-video)
            PASS_ARGS+=("--save-video")
            shift
            ;;
        --disable-video-history)
            PASS_ARGS+=("$1")
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "======================================================"
echo "  Adaptive Sparse VLA Launcher"
echo "  gpu : ${GPU_ID}"
echo "======================================================"

case "${EXP}" in
    baseline_256)
        bash "${BIVLA_ROOT}/experiments/run_baseline_256.sh" "${PASS_ARGS[@]}"
        ;;
    adaptive_sparse)
        bash "${BIVLA_ROOT}/experiments/run_adaptive_sparse.sh" "${PASS_ARGS[@]}"
        ;;
    shared_compact_focus)
        bash "${BIVLA_ROOT}/experiments/run_shared_compact_focus.sh" "${PASS_ARGS[@]}"
        ;;
    *)
        echo "Unknown experiment: ${EXP}"
        echo "Valid values: baseline_256, adaptive_sparse, shared_compact_focus"
        exit 1
        ;;
esac
