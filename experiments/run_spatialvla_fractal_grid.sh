#!/usr/bin/env bash
# Run the SpatialVLA condition grid on SimplerEnv Google Robot / Fractal.
#
# The Fractal column of Figure 1 currently has three cells filled (baseline,
# action repeat 2, action repeat 4) and the rest empty. This runs the rest.
#
# RESUMABLE, on purpose. The 2026-08-06 campaign lost its baseline because the
# Colab runtime dropped between conditions and /content/results went with it,
# which left action-repeat 2 compared against a baseline from a different
# session -- an uncertainty no amount of re-analysis can remove. So: results go
# straight to a directory you control, and a (condition, task) whose JSON is
# already there is skipped rather than re-run. Re-invoking after a dropped
# runtime picks up where it stopped.
#
#   ./run_spatialvla_fractal_grid.sh                      # everything not yet recorded
#   CONDITIONS="baseline foveate_blur" ./run_spatialvla_fractal_grid.sh
#   TASKS=google_robot_move_near_v0 ./run_spatialvla_fractal_grid.sh
#   FORCE=1 ./run_spatialvla_fractal_grid.sh              # re-run even if recorded
#
# One condition over the default four tasks is 135 episodes. Budget from your
# own baseline timing rather than from this comment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
EVAL="${EVAL:-${REPO_ROOT}/SpatialVLA/experiments/tome/tome_spatialvla_eval.py}"
MODEL_PATH="${MODEL_PATH:-IPEC-COMMUNITY/spatialvla-4b-224-pt}"
UNNORM_KEY="${UNNORM_KEY:-fractal20220817_data/0.1.0}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/spatialvla_fractal_0806}"

# The three coke-can orientations plus MoveNear v0 -- v0 because that is the
# variant the authors' reference eval runs. Do not swap in v1 without saying so
# next to the number; they are different envs and score differently.
TASKS="${TASKS:-google_robot_pick_horizontal_coke_can google_robot_pick_vertical_coke_can google_robot_pick_standing_coke_can google_robot_move_near_v0}"

# baseline first, always. Every Delta in the table is measured against it, and
# a baseline run in a different session than its conditions is a caveat you
# carry in the writeup forever.
CONDITIONS="${CONDITIONS:-baseline action_repeat2 action_repeat4 foveate_logpolar foveate_blur depth_prune4}"

condition_flags() {
    case "$1" in
        baseline)          echo "" ;;
        action_repeat2)    echo "--action-repeat 2" ;;
        action_repeat4)    echo "--action-repeat 4" ;;
        # keep 20% of visual sample density -- the RetinaBased default, so the
        # Fractal cell is comparable to the Bridge cell above it.
        foveate_logpolar)  echo "--foveate --foveate-mode logpolar --foveate-keep-percent 20" ;;
        foveate_blur)      echo "--foveate --foveate-mode blur --foveate-keep-percent 20" ;;
        depth_prune4)      echo "--depth-prune 4" ;;
        *) echo "UNKNOWN" ;;
    esac
}

echo "======================================================================"
echo "  SpatialVLA x SimplerEnv Google Robot (Fractal)"
echo "  python     : ${PYTHON}"
echo "  checkpoint : ${MODEL_PATH}"
echo "  unnorm_key : ${UNNORM_KEY}"
echo "  out_root   : ${OUT_ROOT}"
echo "  conditions : ${CONDITIONS}"
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
        echo "== ${cond} / ${task} ${flags:+[${flags}]}"
        # --n-episodes is left at its default of 0 = the whole protocol. A
        # smaller count is an ordered prefix, and MoveNear's ids are grouped by
        # object triplet, so a prefix is a biased sample rather than a smaller
        # unbiased one. That is what made move_near read 91.7% at n=24.
        if ! "${PYTHON}" "${EVAL}" \
                --model-path "${MODEL_PATH}" \
                --policy-setup google_robot \
                --unnorm-key "${UNNORM_KEY}" \
                --task "${task}" \
                --output-dir "${out}" \
                ${flags}; then
            # Keep going. One task dying (OOM, a bad asset) should not cost the
            # other eleven runs, and the skip logic above will pick it up on the
            # next invocation.
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
echo "  Paired tests against baseline:"
echo "    python ${REPO_ROOT}/adaptive_sparse_vla/paired_test.py \\"
echo "      ${OUT_ROOT}/baseline ${OUT_ROOT}/<condition>"
echo "======================================================================"
[[ ${failed} -eq 0 ]]
