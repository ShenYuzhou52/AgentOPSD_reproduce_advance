#!/usr/bin/env bash
# One repeat of the step-100 final eval: aime24 -> aime25 -> val100, sequential on one GPU.
set -u
gpu=$1; rep=$2
PY=/data2/ssd/yixinshen/benchmarks/verl-qwen35-base/.venv/bin/python
EV=/data2/ssd/yixinshen/AgentOPSD-tir/scripts/eval_baseline.py
MODEL=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/final_eval_20260912/merged_step100
D=/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath
R=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/final_eval_20260912
mkdir -p "$R/rep$rep"
export CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/data2/ssd/yixinshen/AgentOPSD-tir
export TMPDIR=/data2/ssd/yixinshen/tmp-pilot TEMP=/data2/ssd/yixinshen/tmp-pilot TMP=/data2/ssd/yixinshen/tmp-pilot
for ds in aime24 aime25 val100; do
  data=$D/val_$ds.parquet
  [ "$ds" = val100 ] && data=$D/val_deepmath100_s42.parquet
  if [ -f "$R/rep$rep/${ds}_summary.json" ]; then echo "skip $ds"; continue; fi
  $PY "$EV" --mode tir --model "$MODEL" --data "$data" --out "$R/rep$rep/$ds.jsonl" \
    --max-turns 15 --max-tokens 32768 --max-model-len 49152 --temperature 0 --thinking --gpu-mem 0.8 \
    > "$R/rep$rep/$ds.log" 2>&1
  [ -f "$R/rep$rep/$ds.jsonl" ] && touch "$R/rep$rep/${ds}_summary.json"
done
touch "$R/rep$rep/DONE"
