#!/usr/bin/env bash
# Run the OpenVLA condition grid on SimplerEnv Google Robot / Fractal.
#
# This is the fifth column of Figure 1 done properly: SpatialVLA on Fractal
# showed that action repeat 2 goes from +12.5 on Bridge to +0.0 on Fractal
# WITHOUT changing the policy, so the benchmark decides too, not only the
# backbone. That claim rests on one backbone crossing one benchmark boundary.
# OpenVLA is the second crossing, and it is the informative one: on Bridge
# OpenVLA loses monotonically with horizon (15.6 -> 7.3 -> 4.2) while SpatialVLA
# peaks at 2. If OpenVLA/Fractal is monotone too, the horizon effect is a
# property of the backbone and the benchmark only scales it. If it is not, it
# is not.
#
# Run baseline FIRST and check it against the published OpenVLA/Fractal numbers
# before running anything else. Fractal support is new here -- the gripper
# convention (relative + 15-step latch) and the action statistics
# (fractal20220817_data) both differ from Bridge, and both fail silently: a
# wrong one still produces a plausible success rate. The baseline is the only
# thing that can catch it, so treat a large disagreement as a setup bug and
# stop, rather than reporting a "surprising" result.
#
# RESUMABLE: a (condition, task) whose JSON is already present is skipped.
# Colab runtimes drop mid-campaign, and the 2026-08-06 SpatialVLA run lost its
# baseline that way.
#
#   ./run_openvla_fractal_grid.sh                        # everything not yet recorded
#   CONDITIONS=baseline ./run_openvla_fractal_grid.sh    # baseline first, then check it
#   FORCE=1 ./run_openvla_fractal_grid.sh                # re-run even if recorded
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
EVAL="${EVAL:-${REPO_ROOT}/RetinaBased/PythonProject/simple_eval.py}"
MODEL_PATH="${MODEL_PATH:-openvla/openvla-7b}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/openvla_fractal}"

TASKS="${TASKS:-google_robot_pick_horizontal_coke_can google_robot_pick_vertical_coke_can google_robot_pick_standing_coke_can google_robot_move_near_v0}"

# The Bridge grid this mirrors is in experiments/OpenVLA_Bridge_Grid.md. Same
# conditions, same keep-percent, same prune count, so the two columns are read
# side by side.
CONDITIONS="${CONDITIONS:-baseline action_repeat2 action_repeat4 foveate_logpolar foveate_blur depth_prune4}"

condition_flags() {
    case "$1" in
        baseline)          echo "--model openvla" ;;
        # openvla_chunk is action repeat, not chunk execution: OpenVLA emits one
        # action per forward, so there is no predicted chunk to truncate. The
        # class name is historical.
        action_repeat2)    echo "--model openvla_chunk --action-repeat 2" ;;
        action_repeat4)    echo "--model openvla_chunk --action-repeat 4" ;;
        foveate_logpolar)  echo "--model openvla_foveated --foveated-keep-percent 20" ;;
        foveate_blur)      echo "--model openvla_foveated_blur --foveated-keep-percent 20" ;;
        depth_prune4)      echo "--model openvla_depth --depth-prune 4" ;;
        *) echo "UNKNOWN" ;;
    esac
}

echo "======================================================================"
echo "  OpenVLA x SimplerEnv Google Robot (Fractal)"
echo "  python     : ${PYTHON}"
echo "  checkpoint : ${MODEL_PATH}"
echo "  out_root   : ${OUT_ROOT}"
echo "  conditions : ${CONDITIONS}"
echo ""
echo "  unnorm_key and the gripper convention are DERIVED from the task name"
echo "  (google_robot_* -> fractal20220817_data + 15-step relative latch)."
echo "  Each run prints what it resolved; check the first one."
echo "======================================================================"

ran=0; skipped=0; failed=0
for cond in ${CONDITIONS}; do
    flags="$(condition_flags "${cond}")"
    if [[ "${flags}" == "UNKNOWN" ]]; then
        echo "!! unknown condition '${cond}' -- add it to condition_flags() rather than"
        echo "   passing raw flags, so the results directory says what was run"
        exit 1
    fi
    for task in ${TASKS}; do
        out="${OUT_ROOT}/${cond}/${task}"
        done_marker="${out}/results_${task}.json"
        if [[ -f "${done_marker}" && -z "${FORCE:-}" ]]; then
            echo "-- skip ${cond}/${task} (already recorded)"
            skipped=$((skipped + 1))
            continue
        fi
        mkdir -p "${out}"
        echo ""
        echo "== ${cond} / ${task} [${flags}]"
        # --n-episodes defaults to 0 = the whole protocol. A smaller count is an
        # ordered prefix; MoveNear's 60 ids are grouped by object triplet, so
        # the first 24 cover only two of five and miss both triplets where the
        # policy has to tell two look-alike cans apart from the instruction.
        if ! "${PYTHON}" "${EVAL}" \
                --task "${task}" \
                --openvla-model-path "${MODEL_PATH}" \
                --output-dir "${out}" \
                --device cuda \
                ${flags}; then
            echo "!! FAILED ${cond}/${task} -- continuing; re-run this script to retry"
            failed=$((failed + 1))
            continue
        fi
        ran=$((ran + 1))
    done
done

echo ""
echo "======================================================================"
echo "  ran ${ran}, skipped ${skipped} (already recorded), failed ${failed}"
echo "  results: ${OUT_ROOT}"
echo ""
echo "  Sanity-check the baseline against the published OpenVLA/Fractal"
echo "  visual-matching numbers BEFORE reporting any Delta."
echo ""
echo "  Paired tests against baseline:"
echo "    python ${REPO_ROOT}/adaptive_sparse_vla/paired_test.py \\"
echo "      ${OUT_ROOT}/baseline ${OUT_ROOT}/<condition>"
echo "======================================================================"
[[ ${failed} -eq 0 ]]
