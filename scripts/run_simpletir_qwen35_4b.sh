#!/usr/bin/env bash
# Isolated three-way Qwen3.5-4B / SimpleTIR launcher.
#
# Usage (after the current ALFWorld job has released the GPUs):
#   METHOD=grpo TRAIN_STEPS=100 bash scripts/run_simpletir_qwen35_4b.sh
#   METHOD=opsd_author_code TRAIN_STEPS=100 bash scripts/run_simpletir_qwen35_4b.sh
#   METHOD=agentopsd TRAIN_STEPS=100 bash scripts/run_simpletir_qwen35_4b.sh
#
# All runtime state, Ray state, logs, checkpoints and caches are under /data2.
# It intentionally leaves trainer.rollout_data_dir and validation_data_dir null:
# generic rollout dumps can preserve reward_model.ground_truth and violate the
# teacher/student information boundary.

set -euo pipefail

METHOD="${METHOD:?set METHOD to grpo, opsd_author_code, or agentopsd}"
case "${METHOD}" in
  grpo|opsd_author_code|agentopsd) ;;
  *) echo "invalid METHOD=${METHOD}" >&2; exit 2 ;;
esac

OVERLAY_DIR="${OVERLAY_DIR:-/data2/ssd/yixinshen/AgentOPSD-tir}"
VERL_DIR="${VERL_DIR:-/data2/ssd/yixinshen/benchmarks/verl-qwen35-base}"
PYTHON_BIN="${PYTHON_BIN:-${VERL_DIR}/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data2/ssd/yixinshen/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets}"
RUN_ROOT="${RUN_ROOT:-/data2/ssd/yixinshen/experiments/qwen35-simpletir}"

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/simplelr_math_35/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/simplelr_math_35/test_fixed100_s42.parquet}"
# Optional, disjoint external-distribution probe. It is validation data only,
# never an optimization input; using it removes its status as a blind test.
PROBE_VAL_FILE="${PROBE_VAL_FILE:-}"

SEED="${SEED:-42}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
# prompt + five 1k model turns + bounded tool observations needs a little over
# 11k tokens in the worst case; keep the default above that hard envelope.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
ACTOR_MINI_BATCH="${ACTOR_MINI_BATCH:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
N_GPUS="${N_GPUS:-8}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
MAX_CKPTS="${MAX_CKPTS:-3}"
RESUME_MODE="${RESUME_MODE:-auto}"
EXPERIMENT="${EXPERIMENT:-qwen35_4b_simpletir_${METHOD}_s${SEED}}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
# Qwen3.5 opens a <think> block by default.  SimpleTIR needs the first
# sampled action to be fenced Python; keep chat-mode generation by default so
# its 1k action budget is not consumed before the code action begins.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# SimpleTIR's Parquet is tiny and rollout dominates wall time.  Avoid eight
# forked DataLoader processes per runner: they add memory pressure without a
# measurable input-pipeline benefit and one was OOM-killed at smoke teardown.
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "missing model: ${MODEL_DIR}" >&2; exit 2; }
"${PYTHON_BIN}" -c 'import math_verify' || {
  echo "missing required dependency math-verify in ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${TRAIN_FILE}" && -f "${VAL_FILE}" ]] || { echo "missing SimpleTIR Parquet" >&2; exit 2; }
if [[ -n "${PROBE_VAL_FILE}" && ! -f "${PROBE_VAL_FILE}" ]]; then
  echo "missing probe validation Parquet: ${PROBE_VAL_FILE}" >&2
  exit 2
fi
[[ "${TRAIN_PROMPTS}" -gt 0 && "${ROLLOUT_N}" -gt 1 && "${MAX_TURNS}" -gt 0 ]] || {
  echo "TRAIN_PROMPTS>0, ROLLOUT_N>1, and MAX_TURNS>0 are required" >&2; exit 2;
}

VAL_FILES_HYDRA="['${VAL_FILE}']"
if [[ -n "${PROBE_VAL_FILE}" ]]; then
  VAL_FILES_HYDRA="['${VAL_FILE}','${PROBE_VAL_FILE}']"
fi

RUN_DIR="${RUN_ROOT}/${EXPERIMENT}"
CKPT_DIR="${RUN_DIR}/checkpoints"
# Keep paths short enough for Ray's AF_UNIX sockets, but isolate simultaneous
# experiments so stale sessions or multiprocessing managers cannot collide.
RUN_TAG="$(printf '%s' "${EXPERIMENT}" | sha256sum | cut -c1-12)"
RAY_DIR="/data2/ssd/yixinshen/r/${RUN_TAG}"
TMP_DIR="/data2/ssd/yixinshen/t/${RUN_TAG}"
CACHE_DIR="${RUN_DIR}/cache"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${RAY_DIR}" "${TMP_DIR}" "${CACHE_DIR}"

export PYTHONPATH="${OVERLAY_DIR}:${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_TMPDIR="${RAY_DIR}"
export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"
export HF_HOME="${CACHE_DIR}/huggingface"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torchinductor"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Pure OPSD needs the same private teacher forward as AgentOPSD, but replaces
# the policy-gradient loss with the public author's gated SDAR loss.  GRPO has
# no teacher request at all.  The main trainer validates that invariant.
ARGS=(
  "model_engine=dp"
  "trainer.use_v1=True"
  "trainer.v1.trainer_mode=sync"
  "transfer_queue.enable=True"
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=['${TRAIN_FILE}']"
  "data.val_files=${VAL_FILES_HYDRA}"
  "data.train_batch_size=${TRAIN_PROMPTS}"
  "data.gen_batch_size=${TRAIN_PROMPTS}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.return_raw_chat=True"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=4"
  "data.truncation=error"
  "data.shuffle=True"
  "data.seed=${SEED}"
  "data.dataloader_num_workers=${DATALOADER_WORKERS}"
  "+data.apply_chat_template_kwargs={enable_thinking:${ENABLE_THINKING}}"
  "actor_rollout_ref.model.path=${MODEL_DIR}"
  "actor_rollout_ref.model.trust_remote_code=False"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  # Qwen3.5's FSDP/FlashAttention path requires packed (unpadded) token
  # batches.  The upstream Qwen3.5-4B FSDP example uses this setting; false
  # reaches the padded FA path and fails on variable-length agent trajectories.
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.strategy=fsdp2"
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=${N_GPUS}"
  "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.fsdp_config.offload_policy=False"
  "actor_rollout_ref.actor.use_dynamic_bsz=False"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_MINI_BATCH}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.01"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.2"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.data_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.calculate_log_probs=True"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_seqs=128"
  "actor_rollout_ref.rollout.enable_prefix_caching=False"
  "actor_rollout_ref.rollout.trace.token2text=False"
  "actor_rollout_ref.rollout.agent.num_workers=4"
  "actor_rollout_ref.rollout.agent.default_agent_loop=simpletir_python"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${OVERLAY_DIR}/integrations/simpletir_qwen35/agent_loop_config.yaml"
  "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integrations.simpletir_qwen35.agent_loop_manager.SimpleTIRAgentLoopManagerTQ"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}"
  "reward.reward_model.enable=False"
  "distillation.enabled=False"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.project_name=qwen35_simpletir"
  "trainer.experiment_name=${EXPERIMENT}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.total_training_steps=${TRAIN_STEPS}"
  "trainer.total_epochs=1000000"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_CKPTS}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
  "trainer.log_val_generations=0"
  "trainer.rollout_data_dir=null"
  "trainer.validation_data_dir=null"
  "ray_kwargs.ray_init.runtime_env.py_executable=${PYTHON_BIN}"
  "+simpletir.method=${METHOD}"
  "+simpletir.metrics_jsonl=${RUN_DIR}/metrics.jsonl"
  "+simpletir.monitor_every=1"
  "+simpletir.max_turns=${MAX_TURNS}"
  "+simpletir.sandbox_timeout_seconds=5"
  "+simpletir.max_observation_chars=512"
  "+simpletir.reward_stdout_chars=16384"
  "+simpletir.sandbox_concurrency=4"
  "+simpletir.opsd_sdar_coef=0.01"
  "+simpletir.opsd_gate_beta=5.0"
  "+simpletir.agentopsd.lam=0.5"
  "+simpletir.agentopsd.b=0.2"
  "+simpletir.agentopsd.gamma=0.95"
  "+simpletir.agentopsd.eps0=1e-4"
  "+simpletir.agentopsd.eps_q=1e-4"
  "+simpletir.agentopsd.success_threshold=0.5"
)

printf '%q\n' "${PYTHON_BIN}" -m integrations.simpletir_qwen35.main_tir "${ARGS[@]}" > "${RUN_DIR}/command.sh"
printf '%s\n' "${ARGS[@]}" > "${RUN_DIR}/config_overrides.txt"

cd "${VERL_DIR}"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  export SIMPLE_TIR_PREFLIGHT_ONLY=1
fi
"${PYTHON_BIN}" -m integrations.simpletir_qwen35.main_tir "${ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
