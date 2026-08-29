#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party"

clone_at() {
  local url="$1" revision="$2" directory="$3"
  if [[ -d "${DEST}/${directory}/.git" ]]; then
    echo "Using existing ${directory} checkout"
    git -C "${DEST}/${directory}" fetch --tags origin
  else
    git clone "${url}" "${DEST}/${directory}"
  fi
  git -C "${DEST}/${directory}" checkout --detach "${revision}"
}

mkdir -p "${DEST}"
clone_at https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e LIBERO
clone_at https://github.com/siyuhsu/vla-cache.git a490988 vla-cache
clone_at https://github.com/siyuhsu/transformers.git 2302fce transformers-vla-cache

if git -C "${DEST}/LIBERO" diff --quiet -- libero/libero/benchmark/__init__.py libero/lifelong/evaluate.py libero/lifelong/metric.py; then
  git -C "${DEST}/LIBERO" apply "${DEST}/libero-pytorch26.patch"
else
  echo "LIBERO compatibility patch already applied or checkout is modified"
fi
echo "Third-party dependencies ready under ${DEST}"
