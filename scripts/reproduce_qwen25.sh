#!/usr/bin/env bash
# Reproduce the paper-oriented Qwen2.5 experiments for AgentOPSD.
#
# Safe by default: without --run this script only prints the commands that
# would be executed. Use --run after confirming that the requested GPUs are
# free. Examples:
#   bash scripts/reproduce_qwen25.sh 3b
#   bash scripts/reproduce_qwen25.sh 7b --run
#   CUDA_VISIBLE_DEVICES=2,3,4,5 bash scripts/reproduce_qwen25.sh both --run
#
# The model weights are expected under MODEL_CACHE (default: $HOME/models),
# and are downloaded separately with scripts/download_model.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="3b"
RUN=0
GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
ENV_NAME="${ENV_NAME:-alfworld}"
MODEL_CACHE="${MODEL_CACHE:-${HOME}/models}"
TRAIN_STEPS="${TRAIN_STEPS:-150}"
SEED="${SEED:-0}"

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Options:
  --run                 Execute training; without this flag only print commands.
  --gpus DEVICES        Comma-separated CUDA device list (default: 2,3,4,5).
  --steps N             Number of training epochs/steps (default: 150).
  --seed N              Environment seed (default: 0).
  --env NAME            Training environment (default: alfworld).
  --model-cache DIR     Directory containing downloaded model folders.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    3b|7b|both) MODE="$1"; shift ;;
    --run) RUN=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --steps) TRAIN_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${GPUS}" ]]; then
  echo "GPU list must not be empty; pass --gpus 2,3,4,5 or set CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi

run_one() {
  local label="$1"
  local model_id model_dir experiment
  case "${label}" in
    3b) model_id="Qwen/Qwen2.5-3B-Instruct" ;;
    7b) model_id="Qwen/Qwen2.5-7B-Instruct" ;;
    *) echo "Unsupported model label: ${label}" >&2; exit 2 ;;
  esac

  model_dir="${MODEL_CACHE}/$(basename "${model_id}")"
  experiment="agentopsd_${label}_${ENV_NAME}_seed${SEED}"
  local -a command=(
    bash scripts/train.sh
    --gpus "${GPUS}"
    --steps "${TRAIN_STEPS}"
    --seed "${SEED}"
    --env "${ENV_NAME}"
    --model "${model_dir}"
    --experiment "${experiment}"
  )

  if [[ "${RUN}" -ne 1 ]]; then
    printf '[dry-run] model=%s\n' "${model_id}"
    printf '[dry-run] '
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi

  [[ -f "${model_dir}/config.json" ]] || {
    echo "Missing model: ${model_dir}/config.json" >&2
    echo "Download it first with: bash scripts/download_model.sh ${model_id} ${MODEL_CACHE}" >&2
    exit 1
  }

  printf '[run] %s (%s) on GPUs %s\n' "${model_id}" "${experiment}" "${GPUS}"
  "${command[@]}"
}

if [[ "${RUN}" -eq 1 ]]; then
  echo "Training execution enabled. Confirm that GPUs ${GPUS} are available."
  if [[ "${ENV_NAME}" == "alfworld" ]]; then
    bash scripts/prepare_data.sh alfworld
  fi
else
  echo "Dry-run only: no data preparation or training will start."
fi

case "${MODE}" in
  3b) run_one 3b ;;
  7b) run_one 7b ;;
  both) run_one 3b; run_one 7b ;;
esac
