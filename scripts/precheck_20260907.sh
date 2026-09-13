#!/usr/bin/env bash
set -uo pipefail
BASE=/data2/ssd/yixinshen
ROOT=${SMOKE_ROOT:-$BASE/experiments/qwen35-simpletir/precheck_20260907}
export ROOT
mkdir -p "$ROOT"
PY=$BASE/benchmarks/verl-qwen35-base/.venv/bin/python
"$PY" - <<'PY'
import pandas as pd
import os
base='/data2/ssd/yixinshen'
pd.read_parquet(base+'/benchmarks/SimpleTIR/datasets/simplelr_math_35/test_fixed100_s42.parquet').head(4).to_parquet(os.environ['ROOT']+'/val4.parquet')
PY
export RAY_ADDRESS=local
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} N_GPUS=2
export TRAIN_STEPS=2 TRAIN_PROMPTS=8 ROLLOUT_N=4 ACTOR_MINI_BATCH=4
export VAL_BEFORE_TRAIN=false TEST_FREQ=2 SAVE_FREQ=2 MAX_CKPTS=1 RESUME_MODE=disable
export VAL_FILE=$ROOT/val4.parquet RUN_ROOT=$ROOT GPU_MEMORY_UTILIZATION=0.30
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
cd "$BASE/AgentOPSD-tir"
for method in ${METHOD_LIST:-grpo agentopsd opsd_author_code}; do
  export METHOD=$method EXPERIMENT=${method}_${RUN_LABEL:-smoke2}_s42
  date -Is > "$ROOT/${method}.started"
  bash scripts/run_simpletir_qwen35_4b.sh > "$ROOT/${method}.console.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$ROOT/${method}.exitcode"
done
