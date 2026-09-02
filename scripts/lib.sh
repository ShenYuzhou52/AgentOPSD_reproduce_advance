#!/usr/bin/env bash
# 公共库：加载配置、路径、工具函数。所有 scripts/*.sh 先 source 本文件。
set -euo pipefail

AGENTOPSD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTOPSD_REPO_ROOT

# 加载配置（默认 scripts/configs/agentopsd.env，可用 AGENTOPSD_CONFIG 覆盖）
_CONFIG_FILE="${AGENTOPSD_CONFIG:-${AGENTOPSD_REPO_ROOT}/scripts/configs/agentopsd.env}"
if [[ -f "${_CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_CONFIG_FILE}"
  set +a
fi

# Prefer an exported key, otherwise load the one-line local key file. The
# latter is ignored by git and works for a bootstrap launcher already running.
_WANDB_KEY_FILE="${WANDB_KEY_FILE:-${AGENTOPSD_REPO_ROOT}/.wandb_api_key}"
if [[ -z "${WANDB_API_KEY:-}" && -r "${_WANDB_KEY_FILE}" ]]; then
  WANDB_API_KEY="$(tr -d '\r\n' < "${_WANDB_KEY_FILE}")"
  export WANDB_API_KEY
fi
unset _WANDB_KEY_FILE

# 从 CUDA_VISIBLE_DEVICES 推导卡数
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="2,3,4,5"
fi
export CUDA_VISIBLE_DEVICES

if [[ -z "${N_GPUS:-}" || "${N_GPUS}" -lt 1 ]]; then
  N_GPUS="$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
fi
export N_GPUS

SDAR_ROOT="${SDAR_ROOT:-${AGENTOPSD_REPO_ROOT}/vendor/SDAR}"
export SDAR_ROOT
export AGENTOPSD_SDAR_ROOT="${SDAR_ROOT}"

log()  { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die()  { log "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "命令 '$1' 未找到（$*）。请先安装。"
}

require_sdar() {
  [[ -d "${SDAR_ROOT}/verl/trainer" ]] || die \
    "SDAR 未克隆到 ${SDAR_ROOT}。请先执行: bash scripts/setup_server.sh"
}

gpu_count_visible() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l
}

check_hardware() {
  require_cmd nvidia-smi
  local total visible
  total="$(gpu_count_visible)"
  visible="$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
  log "GPU 总数=${total}，本次使用 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}（${visible} 卡）"
  if [[ "${visible}" -lt 2 || "${visible}" -gt "${total}" ]]; then
    die "本仓库脚本适配 2~4 张 A800 并行，当前 ${visible} 卡。请设置 CUDA_VISIBLE_DEVICES / N_GPUS。"
  fi
  local max_idx
  max_idx="$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print $NF}')"
  if [[ "${max_idx}" -ge "${total}" ]]; then
    die "CUDA_VISIBLE_DEVICES 包含不存在的卡号（${CUDA_VISIBLE_DEVICES}，机器共 ${total} 卡）。"
  fi
}

check_env() {
  require_cmd python3
  python3 - <<'PY'
import importlib, sys
for m in ("torch", "vllm", "flash_attn", "deepspeed", "verl"):
    try:
        mod = importlib.import_module(m)
        print(f"[env] {m} ok: {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"[env] {m} MISSING: {e!r}")
        sys.exit(1)
try:
    import torch
    print(f"[env] cuda available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
except Exception:
    pass
PY
  df -h /dev/shm | tail -n1
  log "环境检查完成（conda env=${CONDA_ENV:-?}，uv 管理依赖）"
}

latest_checkpoint() {
  local root="$1"
  [[ -d "${root}" ]] || { echo ""; return 0; }
  find "${root}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -n1
}

experiment_name() {
  local model_short="${MODEL_NAME##*/}"
  if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
    echo "${EXPERIMENT_NAME}"
  else
    echo "agentopsd_${model_short}_${ENV_NAME}_g${GROUP_SIZE}_lam${AGENTOPSD_LAM}_b${AGENTOPSD_B}_g${AGENTOPSD_GAMMA}_s${SEED}"
  fi
}

logger_arg() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "['console','wandb']"
  else
    echo "['console']"
  fi
}
