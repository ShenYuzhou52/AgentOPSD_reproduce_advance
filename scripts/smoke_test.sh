#!/usr/bin/env bash
# 冒烟测试：2 卡、2 步、验证集 8 个任务，验证整条链路（数据→rollout→信用→更新→评测）
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

log "==> 冒烟测试（2 卡，2 步）"
check_hardware
bash scripts/prepare_data.sh alfworld

SEED="${SEED:-0}" \
TRAIN_STEPS=2 \
VAL_TASKS=8 \
SAVE_FREQ=1 \
TEST_FREQ=1 \
CUDA_VISIBLE_DEVICES="2,3" \
MICRO_BATCH_PER_GPU=4 \
bash scripts/train.sh --gpus "2,3" --steps 2 --experiment smoke_test

log "==> 冒烟测试完成，检查 logs/smoke_test*/train.log 中的 [agentopsd] 行与 val/ 指标"
