#!/usr/bin/env bash
set -euo pipefail
exec > /data2/ssd/yixinshen/AgentOPSD/pair_bootstrap.log 2>&1
ROOT=/data2/ssd/yixinshen/AgentOPSD
UV=/data2/ssd/yixinshen/home/.local/bin/uv
PY="$ROOT/.venv/bin/python"
export PATH="$ROOT/.venv/bin:/data2/ssd/yixinshen/home/.local/bin:$PATH"
export DATA_ROOT=/data2/ssd/yixinshen/data
export MODEL_CACHE=/data2/ssd/yixinshen/models
export SDAR_ROOT="$ROOT/vendor/SDAR"
export AGENTOPSD_SDAR_ROOT="$SDAR_ROOT"
export CUDA_HOME=/data2/ssd/yixinshen/cuda-12.8-conda
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:$PATH"
export UV_CACHE_DIR=/data2/ssd/yixinshen/cache/uv
echo "[$(date '+%F %T')] waiting for FlashAttention build"
while pgrep -f 'uv pip install.*flash-attn' >/dev/null; do sleep 30; done
"$PY" -c 'import flash_attn; print("flash_attn ready")'
echo "[$(date '+%F %T')] installing DeepSpeed + ALFWorld runtime"
"$UV" pip install --python "$PY" deepspeed==0.18.4 gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
echo "[$(date '+%F %T')] preparing ALFWorld data"
bash "$ROOT/scripts/prepare_data.sh" alfworld
echo "[$(date '+%F %T')] verifying runtime"
"$PY" -c 'import torch,vllm,deepspeed,flash_attn,ray,hydra,verl,alfworld; print(torch.__version__, vllm.__version__, torch.cuda.is_available())'
cd "$ROOT"
GRPO_EXP=qwen25_3b_grpo_s0_20260830
OPSD_EXP=qwen25_3b_agentopsd_s0_20260830
nohup bash scripts/train.sh --gpus 0,1,2,3 --model "$MODEL_CACHE/Qwen2.5-3B-Instruct" --steps 150 --seed 0 --experiment "$GRPO_EXP" --disable-credit > "$ROOT/launch_grpo.log" 2>&1 &
GRPO_PID=$!
nohup bash scripts/train.sh --gpus 4,5,6,7 --model "$MODEL_CACHE/Qwen2.5-3B-Instruct" --steps 150 --seed 0 --experiment "$OPSD_EXP" > "$ROOT/launch_opsd.log" 2>&1 &
OPSD_PID=$!
printf 'GRPO_PID=%s\nOPSD_PID=%s\n' "$GRPO_PID" "$OPSD_PID" > "$ROOT/pair_launch_pids.txt"
echo "[$(date '+%F %T')] launched GRPO=$GRPO_PID AgentOPSD=$OPSD_PID"