#!/usr/bin/env bash
# 用镜像下载模型权重（支持 hf-mirror / ModelScope / GitHub LFS 三种来源）
# 用法:
#   bash scripts/download_model.sh [模型ID] [本地目录]
#   bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct $HOME/models
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

MODEL_ID="${1:-${MODEL_NAME}}"
LOCAL_DIR="${2:-${MODEL_CACHE}/$(basename "${MODEL_ID}")}"

log "==> 下载模型 ${MODEL_ID} -> ${LOCAL_DIR}"

if [[ -f "${LOCAL_DIR}/config.json" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
  log "模型已存在（${LOCAL_DIR}/config.json），跳过下载。如需重下: FORCE_DOWNLOAD=1 bash scripts/download_model.sh"
  exit 0
fi

mkdir -p "${LOCAL_DIR}"

# 方案 A：HuggingFace（默认走 hf-mirror 镜像）
if [[ "${MODELSCOPE_ENABLED:-0}" != "1" ]]; then
  if command -v hf >/dev/null 2>&1; then
    HF_ENDPOINT="${HF_ENDPOINT}" hf download "${MODEL_ID}" --local-dir "${LOCAL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_ENDPOINT="${HF_ENDPOINT}" huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_DIR}"
  else
    uv pip install --python "$(command -v python3)" "huggingface_hub[cli]"
    HF_ENDPOINT="${HF_ENDPOINT}" hf download "${MODEL_ID}" --local-dir "${LOCAL_DIR}"
  fi
fi

# 方案 B：ModelScope（hf 不可达时的备选）
if [[ ! -f "${LOCAL_DIR}/config.json" ]]; then
  log "HF 下载未完成，尝试 ModelScope ..."
  if ! command -v modelscope >/dev/null 2>&1; then
    uv pip install --python "$(command -v python3)" modelscope
  fi
  modelscope download --model "${MODEL_ID}" --local_dir "${LOCAL_DIR}"
fi

# 校验
if [[ ! -f "${LOCAL_DIR}/config.json" ]]; then
  die "下载失败：${LOCAL_DIR}/config.json 不存在"
fi

if [[ -f "${LOCAL_DIR}/model.safetensors.index.json" ]]; then
  log "safetensors index:"
  cat "${LOCAL_DIR}/model.safetensors.index.json"
fi
log "==> 模型就绪: ${LOCAL_DIR}"
log "    训练时使用: MODEL_NAME=${LOCAL_DIR} bash scripts/train.sh"
log "    （GitHub 镜像/LFS 下载说明见 model/download_model.md）"

