#!/usr/bin/env bash
# =============================================================================
# Adaptive Sparse VLA — Central Path Configuration
# Source this file from any run script: source "$(dirname "$0")/../configs/paths.sh"
# All paths are derived from BIVLA_ROOT (auto-detected if not set).
# =============================================================================

# ── Root directories ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BIVLA_ROOT="${BIVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export UNIVLA_ROOT="${BIVLA_ROOT}/UniVLA"
export SIMPLER_ENV_PATH="${BIVLA_ROOT}/SimplerEnv"
export PRETRAIN_DIR="${BIVLA_ROOT}/pretrain"
export RESULTS_DIR="${BIVLA_ROOT}/results"

# ── Pre-trained model paths ──────────────────────────────────────────────────
export EMU_HUB="${PRETRAIN_DIR}/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"
export VQ_HUB="${PRETRAIN_DIR}/Emu3-VisionTokenizer"
export FAST_PATH="${UNIVLA_ROOT}/pretrain/fast_bridge_t5_s50"
export BIVLA_VK_ICD="${BIVLA_ROOT}/configs/nvidia_icd_egl.json"

if [ ! -f "${EMU_HUB}/config.json" ] && [ -d "${EMU_HUB}/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K" ]; then
    export EMU_HUB="${EMU_HUB}/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K"
fi

# ── Conda environment ─────────────────────────────────────────────────────────
export BIVLA_CONDA_ENV="bivla"
export CONDA_PROFILE="${HOME}/miniconda3/etc/profile.d/conda.sh"

# ── Rendering settings ────────────────────────────────────────────────────────
export MUJOCO_GL="osmesa"
export DISPLAY="${DISPLAY:-:99}"

# ── Validate critical paths (warn, not fail) ─────────────────────────────────
_check_path() {
    if [ ! -d "$2" ]; then
        echo "  [WARN] $1 not found: $2"
        echo "         See README.md for environment setup and checkpoint download instructions."
    fi
}

if [ "${BIVLA_CHECK_PATHS:-1}" = "1" ]; then
    _check_path "UniVLA"             "${UNIVLA_ROOT}"
    _check_path "SimplerEnv"         "${SIMPLER_ENV_PATH}"
    _check_path "UniVLA model"       "${EMU_HUB}"
    _check_path "Emu3 tokenizer"     "${VQ_HUB}"
    _check_path "Fast tokenizer"     "${FAST_PATH}"
fi
