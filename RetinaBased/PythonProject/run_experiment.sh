#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
READY_MARKER="${VENV_DIR}/.bivla_shareable_ready"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results_shareable}"
OPENVLA_MODEL_PATH="${OPENVLA_MODEL_PATH:-openvla/openvla-7b}"
OPENVLA_UNNORM_KEY="${OPENVLA_UNNORM_KEY:-bridge_orig}"
DEVICE="${DEVICE:-cuda}"
AUTO_XVFB="${AUTO_XVFB:-1}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:99}"

TASKS=(
  widowx_put_eggplant_in_basket
  widowx_carrot_on_plate
  widowx_stack_cube
  widowx_spoon_on_towel
)

MODELS=(
  openvla
  openvla_foveated
  openvla_retina
)

print_usage() {
  cat <<'EOF'
Usage:
  ./run_shareable.sh setup [--with-apt]
  ./run_shareable.sh smoke
  ./run_shareable.sh full
  ./run_shareable.sh run --model MODEL --task TASK [extra simple_eval.py args...]

Environment variables:
  PYTHON_BIN            Python executable to use. Default: python3
  VENV_DIR              Virtualenv path. Default: ./Shareable/.venv
  RESULTS_ROOT          Output root. Default: ./Shareable/results_shareable
  OPENVLA_MODEL_PATH    Hugging Face model id or local path. Default: openvla/openvla-7b
  OPENVLA_UNNORM_KEY    Action unnormalization key. Default: bridge_orig
  DEVICE                Inference device. Default: cuda
  HF_TOKEN              Optional Hugging Face token for gated model access
  AUTO_XVFB             Start Xvfb automatically when available. Default: 1
  DISPLAY_VALUE         Xvfb display number. Default: :99

Examples:
  ./run_shareable.sh setup --with-apt
  ./run_shareable.sh smoke
  ./run_shareable.sh run --model openvla_foveated --task widowx_spoon_on_towel --n-episodes 4
  ./run_shareable.sh full
EOF
}

run_cmd() {
  echo "+ $*"
  "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

maybe_install_apt() {
  local use_apt="$1"
  if [[ "${use_apt}" != "1" ]]; then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found; skipping system package installation." >&2
    return
  fi

  local -a apt_cmd=()
  if [[ "$(id -u)" -eq 0 ]]; then
    apt_cmd=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    apt_cmd=(sudo apt-get)
  else
    echo "System package installation requested, but neither root nor sudo is available." >&2
    exit 1
  fi

  echo "Installing system dependencies with apt-get..."
  "${apt_cmd[@]}" update
  "${apt_cmd[@]}" install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libjpeg-dev \
    libosmesa6 \
    libosmesa6-dev \
    libpng-dev \
    libsm6 \
    libvulkan1 \
    libxext6 \
    libxrender1 \
    mesa-vulkan-drivers \
    patchelf \
    xvfb
}

create_venv_if_needed() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    need_cmd "${PYTHON_BIN}"
    run_cmd "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
}

activate_venv() {
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
}

install_python_deps() {
  run_cmd python -m pip install --upgrade pip setuptools wheel
  run_cmd python -m pip install --upgrade "numpy<2.0" scipy==1.12.0
  run_cmd python -m pip install --upgrade pillow opencv-python imageio imageio-ffmpeg
  run_cmd python -m pip install --upgrade gymnasium==0.29.1 sapien==2.2.2 mani-skill2==0.5.0
  run_cmd python -m pip install --upgrade GitPython gdown h5py pyyaml tabulate tqdm
  run_cmd python -m pip install --upgrade transforms3d trimesh rtree ruckig
  run_cmd python -m pip install --upgrade accelerate einops huggingface_hub pandas sentencepiece timm transformers
  run_cmd python -m pip install -e "${ROOT}/SimplerEnv/ManiSkill2_real2sim"
  run_cmd python -m pip install -e "${ROOT}/SimplerEnv"
  touch "${READY_MARKER}"
}

maybe_hf_login() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    return
  fi
  python - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face login complete.")
PY
}

export_runtime_env() {
  export SIMPLER_ENV_PATH="${ROOT}/SimplerEnv"
  export RESULTS_DIR="${RESULTS_ROOT}"
  export OPENVLA_MODEL_PATH
  export OPENVLA_UNNORM_KEY
  export BIVLA_VK_ICD="${ROOT}/configs/nvidia_icd_egl.json"
  export MS2_REAL2SIM_ASSET_DIR="${ROOT}/SimplerEnv/ManiSkill2_real2sim/data"
  export MUJOCO_GL="osmesa"
  export PYOPENGL_PLATFORM="osmesa"
  export DISPLAY="${DISPLAY:-${DISPLAY_VALUE}}"
}

XVFB_PID=""

maybe_start_xvfb() {
  if [[ "${AUTO_XVFB}" != "1" ]]; then
    return
  fi
  if ! command -v Xvfb >/dev/null 2>&1; then
    return
  fi
  if pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1; then
    return
  fi
  Xvfb "${DISPLAY}" -screen 0 1400x900x24 >/dev/null 2>&1 &
  XVFB_PID="$!"
  sleep 2
}

cleanup_xvfb() {
  if [[ -n "${XVFB_PID}" ]] && kill -0 "${XVFB_PID}" >/dev/null 2>&1; then
    kill "${XVFB_PID}" >/dev/null 2>&1 || true
  fi
}

ensure_runtime() {
  create_venv_if_needed
  activate_venv
  if [[ ! -f "${READY_MARKER}" ]]; then
    install_python_deps
  fi
  export_runtime_env
  maybe_start_xvfb
  trap cleanup_xvfb EXIT
}

do_setup() {
  local use_apt="0"
  if [[ "${1:-}" == "--with-apt" ]]; then
    use_apt="1"
  fi
  maybe_install_apt "${use_apt}"
  create_venv_if_needed
  activate_venv
  install_python_deps
  maybe_hf_login
  export_runtime_env
  echo ""
  echo "Setup complete."
  echo "Virtualenv: ${VENV_DIR}"
  echo "Results root: ${RESULTS_ROOT}"
  echo "Model path: ${OPENVLA_MODEL_PATH}"
}

run_eval() {
  local model="$1"
  local task="$2"
  shift 2
  mkdir -p "${RESULTS_ROOT}/${model}/${task}"
  run_cmd python "${ROOT}/simple_eval.py" \
    --model "${model}" \
    --task "${task}" \
    --output-dir "${RESULTS_ROOT}/${model}/${task}" \
    --openvla-model-path "${OPENVLA_MODEL_PATH}" \
    --openvla-unnorm-key "${OPENVLA_UNNORM_KEY}" \
    --device "${DEVICE}" \
    "$@"
}

do_smoke() {
  ensure_runtime
  maybe_hf_login
  local task="widowx_spoon_on_towel"
  for model in "${MODELS[@]}"; do
    run_eval "${model}" "${task}" --n-episodes 1
  done
}

do_full() {
  ensure_runtime
  maybe_hf_login
  for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_eval "${model}" "${task}" --n-episodes 24
    done
  done
}

do_run() {
  ensure_runtime
  maybe_hf_login
  run_cmd python "${ROOT}/simple_eval.py" "$@"
}

main() {
  local cmd="${1:-}"
  if [[ -z "${cmd}" ]]; then
    print_usage
    exit 1
  fi
  shift || true

  case "${cmd}" in
    setup)
      do_setup "${1:-}"
      ;;
    smoke)
      do_smoke
      ;;
    full)
      do_full
      ;;
    run)
      do_run "$@"
      ;;
    help|-h|--help)
      print_usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
