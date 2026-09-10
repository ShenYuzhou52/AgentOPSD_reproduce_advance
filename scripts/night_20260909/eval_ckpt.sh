#!/usr/bin/env bash
# 用法: eval_ckpt.sh <HF模型目录> <数据parquet> <GPU号> <输出目录> <标签> [gpu_mem]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
MODEL=${1:?model dir}; DATA=${2:?dataset}; GPU=${3:?gpu}; OUT=${4:?outdir}; TAG=${5:?tag}
GPU_MEM=${6:-0.80}
mkdir -p "$OUT"
cd "$OVERLAY"
export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$OVERLAY:${PYTHONPATH:-}"
"$PY" -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('eval_baseline', '$OVERLAY/scripts/eval_baseline.py')
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
ev.MODEL_DIR = '$MODEL'; ev.DATA = '$DATA'
sys.argv = ['eval_baseline.py','--mode','tir','--out','$OUT/records.jsonl','--max-turns','5','--max-tokens','3072','--temperature','0.0','--gpu-mem','$GPU_MEM']
ev.main()" 2>&1 | tee "$OUT/eval.log"
grep -h "SUMMARY" "$OUT/eval.log" | tail -1 > "$OUT/summary.txt"
mkdir -p "$NIGHT/results"
printf '%s\t%s\n' "$TAG" "$(cat "$OUT/summary.txt")" >> "$NIGHT/results/summary.tsv"
echo "EVAL OK: $TAG"
