#!/usr/bin/env bash
# 用法: merge_one.sh <global_step_N 目录> <输出HF目录>   （CPU-only，不占GPU）
# 使用夜间目录下的 model_merger_patched.py（transformers 5.9 无 Vision2Seq / 无 megatron 的兼容补丁）
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
CKPT=${1:?checkpoint dir}; OUT=${2:?output hf dir}
[ -d "$CKPT/actor" ] || { echo "missing $CKPT/actor" >&2; exit 2; }
[ -f "$NIGHT/model_merger_patched.py" ] || { echo "missing patched merger" >&2; exit 2; }
mkdir -p "$OUT"
cd "$OVERLAY"
"$PY" "$NIGHT/model_merger_patched.py" --backend fsdp \
  --hf_model_path "$CKPT/actor/huggingface" \
  --local_dir "$CKPT/actor" \
  --target_dir "$OUT"
# merger 只写权重+config；tokenizer/processor/chat_template 从 checkpoint 的 huggingface/ 补齐
cp -n "$CKPT"/actor/huggingface/* "$OUT"/ 2>/dev/null || true
touch "$OUT/MERGE_OK"
echo "MERGE OK: $OUT"
