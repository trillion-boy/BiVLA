#!/usr/bin/env bash
# SpatialVLA x fixed foveation, both variants, on Bridge or Fractal.
#
# The two variants are a controlled pair, not two settings of one knob:
#
#   log-polar  warps the image -- pixels MOVE. SpatialVLA encodes an explicit
#              pixel->3D correspondence (Ego3D), so a warp invalidates it.
#   blur       space-variant blur -- pixels stay put, only detail is removed.
#              Intrinsics and depth stay valid.
#
# So the difference between them isolates "does this backbone break because it
# lost detail, or because it lost geometry?" On Bridge the legacy numbers say
# log-polar -7.3 and blur -2.1, which is the right ordering for that hypothesis
# -- but they are UNPAIRED (that campaign kept no per-episode records), and a
# 5-point gap between two unpaired numbers at n=96 is not evidence of anything.
# Re-running them here produces per-episode records, which turns both cells from
# "legacy" into a paired McNemar test against a baseline that already exists.
#
#   ./run_spatialvla_foveation.sh                       # Bridge (default)
#   BENCH=fractal ./run_spatialvla_foveation.sh         # Fractal
#   FOVEATE_CENTER=motion ./run_spatialvla_foveation.sh # gaze follows motion
#   FORCE=1 ./run_spatialvla_foveation.sh               # re-run even if recorded
#
# Bridge is the cheaper of the two and the one that concludes immediately: its
# baseline is already on disk with per-episode records, so nothing extra has to
# be paid before the comparison is paired. Fractal has no baseline yet -- see
# the check below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
EVAL="${EVAL:-${REPO_ROOT}/SpatialVLA/experiments/tome/tome_spatialvla_eval.py}"
MODEL_PATH="${MODEL_PATH:-IPEC-COMMUNITY/spatialvla-4b-224-pt}"
BENCH="${BENCH:-bridge}"
KEEP_PERCENT="${KEEP_PERCENT:-20}"

# Fovea placement and scheduling. Both default to the settings every other
# foveation cell in Figure 1 was measured under -- changing them produces a
# different condition, so they are folded into the directory name below rather
# than silently overwriting the comparable run.
FOVEATE_CENTER="${FOVEATE_CENTER:-image}"   # image | motion
FOVEATE_PHASE="${FOVEATE_PHASE:-always}"    # always | pregrasp

case "${BENCH}" in
    bridge)
        POLICY_SETUP="widowx_bridge"
        UNNORM_KEY="${UNNORM_KEY:-bridge_orig/1.0.0}"
        OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/spatialvla_bridge_0805}"
        TASKS="${TASKS:-widowx_put_eggplant_in_basket widowx_carrot_on_plate widowx_stack_cube widowx_spoon_on_towel}"
        ;;
    fractal)
        POLICY_SETUP="google_robot"
        UNNORM_KEY="${UNNORM_KEY:-fractal20220817_data/0.1.0}"
        OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/spatialvla_fractal_0806}"
        TASKS="${TASKS:-google_robot_pick_horizontal_coke_can google_robot_pick_vertical_coke_can google_robot_pick_standing_coke_can google_robot_move_near_v0}"
        ;;
    *)
        echo "BENCH must be 'bridge' or 'fractal', got '${BENCH}'"; exit 1 ;;
esac

# A non-default fovea setting is a different condition and gets its own
# directory. Without this the skip-if-recorded logic would look at a
# center=image run and conclude a center=motion run was already done.
suffix=""
[[ "${FOVEATE_CENTER}" != "image"  ]] && suffix="${suffix}_${FOVEATE_CENTER}"
[[ "${FOVEATE_PHASE}"  != "always" ]] && suffix="${suffix}_${FOVEATE_PHASE}"
[[ "${KEEP_PERCENT}"   != "20"     ]] && suffix="${suffix}_keep${KEEP_PERCENT}"

BASELINE_DIR="${OUT_ROOT}/baseline"

echo "======================================================================"
echo "  SpatialVLA x fixed foveation  --  ${BENCH}"
echo "  checkpoint : ${MODEL_PATH}"
echo "  setup      : ${POLICY_SETUP} / ${UNNORM_KEY}"
echo "  fovea      : keep ${KEEP_PERCENT}%, center=${FOVEATE_CENTER}, phase=${FOVEATE_PHASE}"
echo "  out_root   : ${OUT_ROOT}"
echo "======================================================================"

# The baseline is not optional. Foveation is reported as a Delta, and a Delta
# against a baseline measured in another session carries a caveat that no
# re-analysis removes -- the 2026-08-06 Fractal campaign is still carrying one.
have_baseline=0
if [[ -d "${BASELINE_DIR}" ]]; then
    have_baseline=$(find "${BASELINE_DIR}" -name 'results_*.json' | wc -l)
fi
n_tasks=$(wc -w <<< "${TASKS}")
if [[ "${have_baseline}" -lt "${n_tasks}" ]]; then
    echo ""
    echo "!! baseline incomplete: ${have_baseline} of ${n_tasks} task files under"
    echo "   ${BASELINE_DIR}"
    if [[ -n "${RUN_BASELINE:-}" ]]; then
        echo "   RUN_BASELINE set -- running it first."
    else
        echo ""
        echo "   Foveation numbers are Deltas. Run the baseline in THIS session:"
        echo "     RUN_BASELINE=1 BENCH=${BENCH} $0"
        echo "   or point OUT_ROOT at a directory that already has one."
        exit 1
    fi
fi

run_one() {   # condition_dir_name, extra flags...
    local cond="$1"; shift
    for task in ${TASKS}; do
        local out="${OUT_ROOT}/${cond}/${task}"
        if [[ -f "${out}/results_${task}.json" && -z "${FORCE:-}" ]]; then
            echo "-- skip ${cond}/${task} (already recorded)"
            continue
        fi
        mkdir -p "${out}"
        echo ""
        echo "== ${cond} / ${task}"
        # --n-episodes stays at its default of 0 = the whole protocol. A smaller
        # count is an ordered prefix, not a smaller unbiased sample.
        if ! "${PYTHON}" "${EVAL}" \
                --model-path "${MODEL_PATH}" \
                --policy-setup "${POLICY_SETUP}" \
                --unnorm-key "${UNNORM_KEY}" \
                --task "${task}" \
                --output-dir "${out}" \
                "$@"; then
            echo "!! FAILED ${cond}/${task} -- continuing; re-run to retry"
            FAILED=$((FAILED + 1))
        fi
    done
}

FAILED=0
[[ "${have_baseline}" -lt "${n_tasks}" ]] && run_one baseline

# Same keep-percent for both, so the only thing that differs between them is
# whether the transform moves pixels.
run_one "foveate_logpolar${suffix}" \
    --foveate --foveate-mode logpolar \
    --foveate-keep-percent "${KEEP_PERCENT}" \
    --foveate-center "${FOVEATE_CENTER}" --foveate-phase "${FOVEATE_PHASE}"

run_one "foveate_blur${suffix}" \
    --foveate --foveate-mode blur \
    --foveate-keep-percent "${KEEP_PERCENT}" \
    --foveate-center "${FOVEATE_CENTER}" --foveate-phase "${FOVEATE_PHASE}"

echo ""
echo "======================================================================"
echo "  Paired tests vs baseline"
echo "======================================================================"
for cond in "foveate_logpolar${suffix}" "foveate_blur${suffix}"; do
    echo ""
    echo "-- ${cond}"
    "${PYTHON}" "${REPO_ROOT}/adaptive_sparse_vla/paired_test.py" \
        "${BASELINE_DIR}" "${OUT_ROOT}/${cond}" || true
done

echo ""
echo "======================================================================"
echo "  failed runs: ${FAILED}"
echo "  results: ${OUT_ROOT}"
echo ""
echo "  The two conditions are also worth testing against EACH OTHER --"
echo "  log-polar vs blur at equal keep-percent is the geometry question,"
echo "  and both being negative against baseline does not answer it:"
echo "    ${PYTHON} ${REPO_ROOT}/adaptive_sparse_vla/paired_test.py \\"
echo "      ${OUT_ROOT}/foveate_blur${suffix} ${OUT_ROOT}/foveate_logpolar${suffix}"
echo "======================================================================"
[[ ${FAILED} -eq 0 ]]
