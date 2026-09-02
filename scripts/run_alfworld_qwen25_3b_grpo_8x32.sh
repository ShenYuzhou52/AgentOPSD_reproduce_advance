#!/usr/bin/env bash
# Public AgentOPSD Qwen2.5-3B ALFWorld GRPO recipe, with local P0 mask fixes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
EXPERIMENT="${EXPERIMENT:-qwen25_3b_grpo_8x32_public_s0_20260901}"
MODEL_PATH="${MODEL_PATH:-/data2/ssd/yixinshen/models/Qwen2.5-3B-Instruct}"
CUDA_ROOT="${CUDA_ROOT:-/data2/ssd/yixinshen/cuda-12.8-conda}"

# Match the historical launcher: the server shell itself intentionally has no
# Python/CUDA activation, so make the runtime provenance explicit here.
export CUDA_HOME="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ROOT}/.venv/bin:/data2/ssd/yixinshen/home/.local/bin:${CUDA_ROOT}/bin:${PATH}"

export MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-32}"
export PPO_CLIP_HIGH="${PPO_CLIP_HIGH:-0.2}"
# Leave headroom for the FSDP backward peak while the vLLM rollout engine is
# resident.  This changes cache capacity/throughput only, not GRPO sampling or
# optimization semantics.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
# Keep GRPO group normalisation within each trajectory rather than pooling
# unrelated flattened turns across environment steps.
export COMPUTE_MEAN_STD_CROSS_STEPS="${COMPUTE_MEAN_STD_CROSS_STEPS:-false}"
# ALFWorld's custom AgentSystem emits flattened agent turns. This is distinct
# from verl's SGLang native tool protocol (which needs tool_config_path).
export MULTI_TURN_ENABLE="${MULTI_TURN_ENABLE:-false}"
export MAX_ABS_GRPO_ADVANTAGE="${MAX_ABS_GRPO_ADVANTAGE:-100}"
# An 8-way actor+optimizer checkpoint is ~36 GB. Keep several resumable
# recovery points without exhausting the data volume over 150 updates.
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-5}"

exec bash "${ROOT}/scripts/train.sh" \
  --gpus "${GPU_LIST}" \
  --model "${MODEL_PATH}" \
  --steps "${TRAIN_STEPS:-150}" \
  --seed "${SEED:-0}" \
  --experiment "${EXPERIMENT}" \
  --resume auto \
  --disable-credit
