#!/usr/bin/env bash
# 把 FSDP 分片断点合并回 HF 权重（用于下游评测/发布）
# 用法: bash scripts/merge_checkpoint.sh checkpoints/proj/exp/global_step_40 [输出目录]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CKPT_DIR="${1:-}"
[[ -n "${CKPT_DIR}" ]] || die "用法: bash scripts/merge_checkpoint.sh <global_step_N> [输出目录]"
TARGET_DIR="${2:-${CKPT_DIR}-hf}"

require_sdar
require_cmd python3

[[ -d "${CKPT_DIR}/actor" ]] || die "缺少 ${CKPT_DIR}/actor"
log "==> 合并 FSDP 断点 ${CKPT_DIR}/actor -> ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
cd "${SDAR_ROOT}"
python3 scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir "${CKPT_DIR}/actor" \
  --target_dir "${TARGET_DIR}"

log "==> 合并完成: ${TARGET_DIR}"
log "    可用 bash scripts/eval.sh --ckpt ${CKPT_DIR} 评测原始断点，或用合并后的 HF 权重做推理。"
