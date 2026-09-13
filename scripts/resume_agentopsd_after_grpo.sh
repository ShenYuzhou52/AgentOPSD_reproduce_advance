#!/usr/bin/env bash
# Start the AgentOPSD formal run only after the preceding GRPO run finishes
# all 200 steps.  This prevents their combined host-memory peak from causing
# Ray to kill rollout workers.
set -euo pipefail

GRPO_PID="${1:?pass the GRPO main_tir PID}"
ROOT="/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907"
GRPO_CKPT="${ROOT}/grpo/grpo_formal200_s42/checkpoints/latest_checkpointed_iteration.txt"
STATUS="${ROOT}/agentopsd/agentopsd_formal200_s42/resume_after_grpo.status"

while kill -0 "${GRPO_PID}" 2>/dev/null; do
  sleep 60
done

if [[ -r "${GRPO_CKPT}" && "$(tr -d '[:space:]' < "${GRPO_CKPT}")" == "200" ]]; then
  printf '%s GRPO completed; launching AgentOPSD from its auto-resume checkpoint.\n' "$(date -Is)" > "${STATUS}"
  cd /data2/ssd/yixinshen/AgentOPSD-tir
  exec env \
    METHOD=agentopsd EXPERIMENT=agentopsd_formal200_s42 \
    RUN_ROOT="${ROOT}/agentopsd" CUDA_VISIBLE_DEVICES=4,5,6,7 N_GPUS=4 \
    TRAIN_STEPS=200 TRAIN_PROMPTS=16 ROLLOUT_N=8 MAX_TURNS=5 \
    MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=18432 \
    ACTOR_MINI_BATCH=8 GPU_MEMORY_UTILIZATION=0.40 TEST_FREQ=5 SAVE_FREQ=5 \
    MAX_CKPTS=3 RESUME_MODE=auto VAL_BEFORE_TRAIN=true DATALOADER_WORKERS=0 \
    bash scripts/run_simpletir_qwen35_4b.sh
fi

printf '%s GRPO ended without global_step_200; AgentOPSD was not started.\n' "$(date -Is)" > "${STATUS}"
exit 1
