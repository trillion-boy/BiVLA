#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-openvla/openvla-7b}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/models/openvla-7b}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI is required. Install it with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"
echo "Downloading ${MODEL_ID} to ${MODEL_DIR}"
hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
echo "Model ready at ${MODEL_DIR}"
