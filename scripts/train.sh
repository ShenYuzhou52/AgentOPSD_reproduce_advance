#!/usr/bin/env bash
# AgentOPSD 8B 训练入口 —— 适配 2~4 张 A800，默认使用第 2,3,4,5 卡，支持断点续训。
#
# 用法示例:
#   bash scripts/train.sh                                   # 默认配置（150 步，4 卡 2,3,4,5）
#   bash scripts/train.sh --steps 50 --seed 1
#   bash scripts/train.sh --gpus 2,3 --steps 50             # 2 卡
#   bash scripts/train.sh --fresh                          # 强制从头训练
#   bash scripts/train.sh --resume-ckpt checkpoints/xxx/global_step_40
#   bash scripts/train.sh --disable-credit                 # 退化为 vanilla GRPO 对照
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)        TRAIN_STEPS="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --experiment)   EXPERIMENT_NAME="$2"; shift 2 ;;
    --model)        MODEL_NAME="$2"; shift 2 ;;
    --env)          ENV_NAME="$2"; shift 2 ;;
    --gpus)         CUDA_VISIBLE_DEVICES="$2"; N_GPUS="$(printf '%s\n' "$2" | awk -F',' '{print NF}')"; shift 2 ;;
    --resume)       RESUME_MODE="$2"; shift 2 ;;
    --resume-ckpt)  RESUME_CKPT="$2"; RESUME_MODE="resume_path"; shift 2 ;;
    --fresh)        RESUME_MODE="disable"; shift ;;
    --disable-credit) AGENTOPSD_ENABLE=0; shift ;;
    --help|-h)      usage; exit 0 ;;
    *) die "未知参数: $1（--help 查看用法）" ;;
  esac
done

check_hardware
require_sdar
check_env

EXPERIMENT="$(experiment_name)"
RUN_DIR="${LOG_ROOT}/${EXPERIMENT}"
CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT}"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}"
log "实验名: ${EXPERIMENT}"
log "日志:   ${RUN_DIR}"
log "断点:   ${CKPT_DIR}"

case "${RESUME_MODE}" in
  auto)
    CKPT="$(latest_checkpoint "${CKPT_DIR}")"
    if [[ -n "${CKPT}" ]]; then
      log "auto 续训: 发现最近断点 ${CKPT}"
    else
      log "auto 续训: ${CKPT_DIR} 下没有断点，从头训练"
    fi
    ;;
  resume_path)
    [[ -n "${RESUME_CKPT}" ]] || {
      log "可用断点:"
      find "${CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V || true
      die "RESUME_MODE=resume_path 需要 --resume-ckpt checkpoints/.../global_step_N"
    }
    [[ -d "${RESUME_CKPT}/actor" ]] || die "断点目录不存在或缺少 actor/: ${RESUME_CKPT}"
    log "从指定断点续训: ${RESUME_CKPT}"
    ;;
  disable)
    log "disable: 强制从头训练（不加载任何断点）"
    ;;
  *) die "RESUME_MODE 只支持 auto | resume_path | disable" ;;
esac

# ---- 数据就绪检查 ----
case "${ENV_NAME}" in
  alfworld)
    TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
    VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
    [[ -f "${TRAIN_DATA}" && -f "${VAL_DATA}" ]] || {
      log "训练/验证 parquet 不存在，先执行: bash scripts/prepare_data.sh alfworld"
      bash scripts/prepare_data.sh alfworld
    }
    ENV_ARG="env.env_name=alfworld/AlfredTWEnv"
    SKILLS_DIR="${SDAR_ROOT}/skills/alfworld"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  webshop)
    TRAIN_DATA="${DATA_ROOT}/verl-agent/text/train.parquet"
    VAL_DATA="${DATA_ROOT}/verl-agent/text/test.parquet"
    [[ -f "${TRAIN_DATA}" && -f "${VAL_DATA}" ]] || die "先执行: bash scripts/prepare_data.sh webshop"
    ENV_ARG="env.env_name=Webshop"
    SKILLS_DIR="${SDAR_ROOT}/skills/webshop"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    MAX_STEPS="${MAX_STEPS:-15}"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  search)
    TRAIN_DATA="${DATA_ROOT}/searchR1_processed_direct/train.parquet"
    VAL_DATA="${DATA_ROOT}/searchR1_processed_direct/val.parquet"
    [[ -f "${TRAIN_DATA}" && -f "${VAL_DATA}" ]] || die "先执行: bash scripts/prepare_data.sh search（并保持检索服务运行）"
    ENV_ARG="env.env_name=search"
    SKILLS_DIR="${SDAR_ROOT}/skills/search"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
    MAX_STEPS="${MAX_STEPS:-4}"
    TP_SIZE="${TP_SIZE:-1}"
    EXTRA_THINKING="+data.apply_chat_template_kwargs.enable_thinking=False"
    ;;
  *) die "未知 ENV_NAME=${ENV_NAME}，支持: alfworld | webshop | search" ;;
esac

# ---- 模型路径 ----
if [[ "${MODEL_NAME}" != /* && ! -d "${MODEL_NAME}" ]]; then
  if [[ -d "${MODEL_CACHE}/$(basename "${MODEL_NAME}")" ]]; then
    MODEL_NAME="${MODEL_CACHE}/$(basename "${MODEL_NAME}")"
  else
    log "本地未找到模型 ${MODEL_NAME}，先执行: bash scripts/download_model.sh ${MODEL_NAME}"
    bash scripts/download_model.sh "${MODEL_NAME}"
  fi
fi
[[ -f "${MODEL_NAME}/config.json" ]] || die "模型目录缺少 config.json: ${MODEL_NAME}"

LOGGER="$(logger_arg)"
RAY_ARGS=()
if [[ -n "${RAY_NUM_CPUS:-}" ]]; then
  RAY_ARGS+=("ray_init.num_cpus=${RAY_NUM_CPUS}")
fi

RESUME_ARGS=(
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.default_local_dir=${CKPT_DIR}"
)
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
  RESUME_ARGS+=("trainer.resume_from_path=${RESUME_CKPT}")
fi

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
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.actor.clip_ratio_high=${PPO_CLIP_HIGH}"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.multi_turn.enable=True"
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
  "+algorithm.compute_mean_std_cross_steps=False"
  "+algorithm.agentopsd.enabled=${AGENTOPSD_ENABLE}"
  "+algorithm.agentopsd.lam=${AGENTOPSD_LAM}"
  "+algorithm.agentopsd.b=${AGENTOPSD_B}"
  "+algorithm.agentopsd.gamma=${AGENTOPSD_GAMMA}"
  "+algorithm.agentopsd.eps0=${AGENTOPSD_EPS0}"
  "+algorithm.agentopsd.success_threshold=${AGENTOPSD_SUCCESS_THRESHOLD}"
  "+algorithm.agentopsd.skills_dir=${SKILLS_DIR}"
  "+algorithm.agentopsd.skill_all=${SKILL_ALL}"
  "${ENV_ARG}"
  "env.seed=${SEED}"
  "env.max_steps=${MAX_STEPS}"
  "env.rollout.n=${GROUP_SIZE}"
  "env.resources_per_worker.num_cpus=${ENV_CPUS}"
  "trainer.critic_warmup=0"
  "trainer.logger=${LOGGER}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.experiment_name=${EXPERIMENT}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.ray_wait_register_center_timeout=600"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_epochs=${TRAIN_STEPS}"
  "trainer.val_before_train=True"
  "${RESUME_ARGS[@]}"
  "${RAY_ARGS[@]}"
  "hydra.run.dir=${RUN_DIR}/hydra"
  "+hydra.job.chdir=False"
)

log "==> 训练配置:"
printf '    %s\n' "${VERL_ARGS[@]}" | tee "${RUN_DIR}/config.txt"
if [[ "${AGENTOPSD_ENABLE}" == "1" ]]; then
  TRAIN_ENTRY="agentopsd.trainer.main_agentopsd"
else
  # A true GRPO control must not construct the teacher/skill provider at all.
  TRAIN_ENTRY="verl.trainer.main_ppo"
fi
log "==> 启动训练（${N_GPUS} 卡: ${CUDA_VISIBLE_DEVICES}，entry=${TRAIN_ENTRY}，AgentOPSD enable=${AGENTOPSD_ENABLE}, λ=${AGENTOPSD_LAM}, γ=${AGENTOPSD_GAMMA}）"

cd "${SDAR_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHONPATH="${SDAR_ROOT}:${AGENTOPSD_REPO_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT:-}" \
WANDB_API_KEY="${WANDB_API_KEY:-}" \
AGENTOPSD_METRICS_JSONL="${RUN_DIR}/agentopsd_metrics.jsonl" \
python3 -m "${TRAIN_ENTRY}" "${VERL_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"

log "==> 训练结束（或中断，可用同命令自动续训）。日志: ${RUN_DIR}/train.log"

