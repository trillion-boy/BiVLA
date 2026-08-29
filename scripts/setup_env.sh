#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
ENV_DIR="${VLA_ENV_DIR:-${HOME}/.conda/envs/vla_tricks}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found at ${CONDA_BIN}; set CONDA_BIN explicitly." >&2
  exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" env create -f "${ROOT}/configs/environment.yml"
else
  echo "Using existing environment at ${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --no-deps -e "${ROOT}/third_party/transformers-vla-cache"
"${PYTHON}" -m pip install --no-deps -e "${ROOT}/third_party/vla-cache/src/openvla"
"${PYTHON}" -m pip install --no-deps -e "${ROOT}/third_party/LIBERO" \
  --config-settings editable_mode=compat
"${PYTHON}" -m pip install --no-deps -e "${ROOT}"
"${PYTHON}" -m ipykernel install --user --name vla_tricks \
  --display-name "Python (vla_tricks)"

echo "Environment ready. Run: make quick"
