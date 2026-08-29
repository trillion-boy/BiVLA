#!/usr/bin/env bash
# Source this file: source scripts/activate_env.sh

_VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module load Miniconda3/25.5.1-0
conda activate vla_tricks
export LIBERO_CONFIG_PATH="${_VLA_ROOT}/configs/libero"
export MUJOCO_GL=egl
export VLA_MODEL_PATH="${VLA_MODEL_PATH:-${_VLA_ROOT}/models/openvla-7b}"
unset _VLA_ROOT
