#!/usr/bin/env bash
# 用已有断点做一次完整验证集评测（只跑 _validate()，不更新参数）
# 用法:
#   bash scripts/eval.sh --ckpt checkpoints/xxx/global_step_40
#   RESUME_CKPT=... bash scripts/eval.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --gpus) CUDA_VISIBLE_DEVICES="$2"; N_GPUS="$(printf '%s\n' "$2" | awk -F',' '{print NF}')"; shift 2 ;;
    --help|-h) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "未知参数: $1" ;;
  esac
done

[[ -n "${RESUME_CKPT:-}" ]] || die "需要 --ckpt checkpoints/.../global_step_N"
[[ -d "${RESUME_CKPT}/actor" ]] || die "断点缺少 actor/ 目录: ${RESUME_CKPT}"

check_hardware
require_sdar
mkdir -p "$RAY_TEMP_DIR"

EXPERIMENT="$(experiment_name)_eval"
RUN_DIR="${LOG_ROOT}/${EXPERIMENT}"
mkdir -p "${RUN_DIR}"

# 与 train.sh 相同的环境分支
case "${ENV_NAME}" in
  alfworld)
    TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
    VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
    ENV_ARG="env.env_name=alfworld/AlfredTWEnv"
    SKILLS_DIR="${SDAR_ROOT}/skills/alfworld"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  webshop)
    TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
    VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
    ENV_ARG="env.env_name=Webshop"
    SKILLS_DIR="${SDAR_ROOT}/skills/webshop"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    MAX_STEPS="${MAX_STEPS:-15}"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  search)
    TRAIN_DATA="${DATA_ROOT}/searchR1_processed_direct/train.parquet"
    VAL_DATA="${DATA_ROOT}/searchR1_processed_direct/val.parquet"
    ENV_ARG="env.env_name=search"
    SKILLS_DIR="${SDAR_ROOT}/skills/search"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    MAX_STEPS="${MAX_STEPS:-4}"
    TP_SIZE="${TP_SIZE:-1}"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  *) die "未知 ENV_NAME=${ENV_NAME}" ;;
esac

[[ -f "${TRAIN_DATA}" && -f "${VAL_DATA}" ]] || die "数据不存在，先执行 prepare_data.sh"

if [[ "${MODEL_NAME}" != /* && -d "${MODEL_CACHE}/$(basename "${MODEL_NAME}")" ]]; then
  MODEL_NAME="${MODEL_CACHE}/$(basename "${MODEL_NAME}")"
fi
[[ -f "${MODEL_NAME}/config.json" ]] || die "模型目录缺少 config.json: ${MODEL_NAME}"

VERL_ARGS=(
  "algorithm.adv_estimator=grpo"
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_DATA}"
  "data.train_batch_size=${TRAIN_TASKS}"
  "data.val_batch_size=${VAL_TASKS}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "${EXTRA_THINKING}"
  "actor_rollout_ref.model.path=${MODEL_NAME}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=True"
  "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
  "algorithm.use_kl_in_reward=False"
  "+algorithm.agentopsd.enable=0"
  "+algorithm.agentopsd.skills_dir=${SKILLS_DIR}"
  "${ENV_ARG}"
  "env.seed=${SEED}"
  "env.max_steps=${MAX_STEPS}"
  "env.rollout.n=${GROUP_SIZE}"
  "env.resources_per_worker.num_cpus=${ENV_CPUS}"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.experiment_name=${EXPERIMENT}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.ray_wait_register_center_timeout=600"
  "+ray_init._temp_dir=$RAY_TEMP_DIR"
  "trainer.resume_mode=resume_path"
  "trainer.resume_from_path=${RESUME_CKPT}"
  "trainer.val_only=True"
  "trainer.val_before_train=True"
  "trainer.total_epochs=0"
  "trainer.save_freq=-1"
  "trainer.default_local_dir=${CHECKPOINT_ROOT}/$(experiment_name)"
  "hydra.run.dir=${RUN_DIR}/hydra"
  "+hydra.job.chdir=False"
)

log "==> 评测断点: ${RESUME_CKPT}（${N_GPUS} 卡: ${CUDA_VISIBLE_DEVICES}）"
cd "${SDAR_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${SDAR_ROOT}:${AGENTOPSD_REPO_ROOT}" \
python3 -m agentopsd.trainer.main_agentopsd "${VERL_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
log "==> 评测完成，结果见 ${RUN_DIR}/eval.log（step:... - val/... 行）"

