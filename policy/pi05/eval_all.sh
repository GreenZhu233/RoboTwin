#!/bin/bash
#
# eval_all.sh — 批量评测 envs 目录下所有任务
#
# 用法:
#   nohup bash eval_all.sh <GPU_ID> <MIN_STEPS> <MAX_STEPS> <WINDOW_SIZE> > /tmp/eval_all.log 2>&1 &
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== 任务列表 ========================
TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)
# ==========================================================

# ======================== 参数解析 ========================
if [ $# -lt 4 ]; then
    echo "用法: $0 <GPU_ID> <MIN_STEPS> <MAX_STEPS> <WINDOW_SIZE>"
    echo "示例: $0 1 10 10 5"
    exit 1
fi

GPU_ID="$1"
MIN_STEPS="$2"
MAX_STEPS="$3"
WINDOW_SIZE="$4"

TASK_CONFIG="demo_clean"
TRAIN_CONFIG="pi05_base_finetune_on_robotwin_clean_randomized_joint_training"
MODEL_NAME="pi05"
SEED="0"

LOG_ROOT="$SCRIPT_DIR/eval_logs"
mkdir -p "$LOG_ROOT"

TOTAL=${#TASKS[@]}
CURRENT=0

echo "================================================"
echo "  Batch Evaluation Start"
echo "  Tasks: ${TOTAL}"
echo "  Config: ${TASK_CONFIG} / ${TRAIN_CONFIG}"
echo "  GPU: ${GPU_ID}"
echo "  Logs: ${LOG_ROOT}"
echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

for task_name in "${TASKS[@]}"; do
    CURRENT=$((CURRENT + 1))

    LOG_FILE="${LOG_ROOT}/${task_name}_${MIN_STEPS}_${MAX_STEPS}.log"

    echo "[${CURRENT}/${TOTAL}] ${task_name} (min=${MIN_STEPS}, max=${MAX_STEPS}) — $(date '+%H:%M:%S')"

    source .venv/bin/activate
    bash eval.sh \
        "${task_name}" \
        "${TASK_CONFIG}" \
        "${TRAIN_CONFIG}" \
        "${MODEL_NAME}" \
        "${SEED}" \
        "${GPU_ID}" \
        "${MIN_STEPS}" \
        "${MAX_STEPS}" \
        "${WINDOW_SIZE}"
        > "$LOG_FILE" 2>&1

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "  !! FAILED (exit=${RC}), see ${LOG_FILE}"
    else
        echo "  -> done"
    fi
done

echo ""
echo "================================================"
echo "  Batch Evaluation Complete"
echo "  Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
