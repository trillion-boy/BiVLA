# 4-task WidowX-Bridge pilot (same tasks/params as bridge.bash), parametrized
# over the chunk-execution / foveation experiment configs so each can be run
# separately and land in its own results dir.
#
# Usage:
#   bash scripts/bridge_chunk_foveate.bash <ckpt_path> <config_path> [mode]
#
# mode (default: all):
#   baseline        exec-chunk off, no foveation  (results_v3_baseline)
#   chunk2          exec-chunk k=2                (results_v3_chunk2)
#   foveate         log-polar foveation, keep 20%  (results_v3_foveate)
#   foveate_chunk2  foveation + exec-chunk k=2     (results_v3_foveate_chunk2)
#   all             run all four, one after another
#
# Examples:
#   bash scripts/bridge_chunk_foveate.bash /content/pretrain/xxx.pt configs/xxx.json chunk2
#   bash scripts/bridge_chunk_foveate.bash /content/pretrain/xxx.pt configs/xxx.json all

set -e

policy_model=robovlm
ckpt_path=$1
config_path=$2
mode=${3:-all}
conda activate robovlms

run_pilot() {
  # $1 = EXTRA_ARGS (space-separated, unquoted so it word-splits), $2 = logging_dir
  local EXTRA_ARGS="$1"
  local LOGGING_DIR="$2"

  scene_name=bridge_table_1_v1
  robot=widowx
  rgb_overlay_path=real_inpainting/bridge_real_eval_1.png
  robot_init_x=0.147
  robot_init_y=0.028

  python eval/simpler/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --config_path ${config_path} \
    --robot ${robot} --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir ${LOGGING_DIR} ${EXTRA_ARGS};

  python eval/simpler/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --config_path ${config_path} \
    --robot ${robot} --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir ${LOGGING_DIR} ${EXTRA_ARGS};

  python eval/simpler/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --config_path ${config_path} \
    --robot ${robot} --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
    --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir ${LOGGING_DIR} ${EXTRA_ARGS};

  scene_name=bridge_table_1_v2
  robot=widowx_sink_camera_setup
  rgb_overlay_path=real_inpainting/bridge_sink.png
  robot_init_x=0.127
  robot_init_y=0.06

  python eval/simpler/main_inference.py --policy-model ${policy_model} --ckpt-path ${ckpt_path} --config_path ${config_path} \
    --robot ${robot} --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
    --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
    --rgb-overlay-path ${rgb_overlay_path} \
    --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --logging-dir ${LOGGING_DIR} ${EXTRA_ARGS};
}

run_baseline()       { run_pilot "" "results_v3_baseline"; }
run_chunk2()         { run_pilot "--exec-chunk 2" "results_v3_chunk2"; }
run_foveate()        { run_pilot "--foveate --foveate-mode logpolar --foveate-keep-percent 20" "results_v3_foveate"; }
run_foveate_chunk2() { run_pilot "--foveate --foveate-mode logpolar --foveate-keep-percent 20 --exec-chunk 2" "results_v3_foveate_chunk2"; }

case ${mode} in
  baseline)       run_baseline ;;
  chunk2)         run_chunk2 ;;
  foveate)        run_foveate ;;
  foveate_chunk2) run_foveate_chunk2 ;;
  all)
    run_baseline
    run_chunk2
    run_foveate
    run_foveate_chunk2
    ;;
  *)
    echo "unknown mode: ${mode} (expected baseline|chunk2|foveate|foveate_chunk2|all)" >&2
    exit 1
    ;;
esac
