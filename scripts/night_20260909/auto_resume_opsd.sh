#!/usr/bin/env bash
# 仅当 OPSD 进程消失且未到 200 步时调用（用户已授权自动续跑）
# 参数与 2026-09-08 正式跑完全一致，RESUME_MODE=auto 从最近 checkpoint 继续
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
CKPT_FILE=$R/opsd/opsd_formal200_s42/checkpoints/latest_checkpointed_iteration.txt
if pgrep -f "experiment_name=opsd_formal200_s42" >/dev/null; then
  echo "OPSD still running; refuse to double-launch" >&2; exit 2
fi
[ -r "$CKPT_FILE" ] || { echo "no latest_checkpointed_iteration.txt; NOT resuming" >&2; exit 3; }
cd "$OVERLAY"
METHOD=opsd_author_code EXPERIMENT=opsd_formal200_s42 RUN_ROOT=$R/opsd \
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 TRAIN_STEPS=200 TRAIN_PROMPTS=16 ROLLOUT_N=8 MAX_TURNS=5 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=18432 ACTOR_MINI_BATCH=8 \
GPU_MEMORY_UTILIZATION=0.40 TEST_FREQ=5 SAVE_FREQ=5 MAX_CKPTS=3 RESUME_MODE=auto \
VAL_BEFORE_TRAIN=true DATALOADER_WORKERS=0 \
nohup bash scripts/run_simpletir_qwen35_4b.sh >> "$NIGHT/opsd_auto_resume.log" 2>&1 &
echo "relaunched OPSD pid $! at $(date -Is)"
