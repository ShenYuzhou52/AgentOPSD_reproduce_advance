#!/usr/bin/env bash
# Watchdog: wait until GPUs 4-7 are all free, then launch the formal GRPO run
# (4-GPU config) plus the corrected overlong monitor. Idempotent via state files.
set -u

R=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910
OVERLAY=/data2/ssd/yixinshen/AgentOPSD-tir
PY=/data2/ssd/yixinshen/benchmarks/verl-qwen35-base/.venv/bin/python
STATE=$R/watchdog_state.json
LOCK=$R/watchdog.lock
MIN_DISK_GB=500
DEADLINE=$(( $(date +%s) + 7*24*3600 ))   # give up after 7 days

mkdir -p "$R"
exec 9>"$LOCK" || exit 1
flock -n 9 || exit 0   # another watchdog instance is already running

now() { date +%s; }
free_gb() { df -BG /data2 | awk 'NR==2{gsub("G","",$4); print $4}'; }
gpus_free() {
  # free = <1GB used and <5% util on ALL of 4,5,6,7 for two consecutive checks
  local raw
  raw=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null) || return 1
  local idx mem util
  while IFS=',' read -r idx mem util; do
    idx=$(echo "$idx" | tr -d ' '); mem=$(echo "$mem" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
    case "$idx" in
      4|5|6|7) [ "$mem" -lt 1024 ] && [ "$util" -lt 5 ] || return 1 ;;
    esac
  done <<< "$raw"
  return 0
}
write_state() {
  $PY - "$STATE" "$1" <<'PYW'
import sys, json
from datetime import datetime
path, stage = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    data = {}
data.update({"stage": stage, "updated_at": datetime.now().isoformat()})
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(data, indent=1, ensure_ascii=False))
open(path, "w").write(open(tmp).read())
PYW
}

write_state waiting
while [ "$(now)" -lt "$DEADLINE" ]; do
  # if the training we launched is already running, hand over and exit
  FP=$(cat "$R/formal_pid" 2>/dev/null || echo 0)
  if [ "$FP" != "0" ] && [ -d "/proc/$FP" ]; then
    write_state training_running
    exit 0
  fi
  if ! gpus_free; then
    write_state waiting_gpus_busy
    sleep 300
    continue
  fi
  # second consecutive confirmation after 2 minutes
  sleep 120
  if ! gpus_free; then
    write_state waiting_gpus_busy_flap
    sleep 240
    continue
  fi
  DISK=$(free_gb)
  if [ "$DISK" -lt "$MIN_DISK_GB" ]; then
    write_state waiting_disk_low
    sleep 600
    continue
  fi
  # clean possible leftovers from a previous killed run
  for p in $(pgrep -f "main_tir" 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
  pkill -f "monitor_overlong.py" 2>/dev/null
  sleep 5

  write_state launching
  M=$($PY -c "import json;m=json.load(open('/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath/build_manifest.json'));print(m['train_file'],m['val_file'],m['aime_files']['aime24'],m['aime_files']['aime25'],sep=':')")
  TRAIN=$(echo "$M" | cut -d: -f1); VALS=$(echo "$M" | cut -d: -f2-)
  EXPERIMENT=grpo_deepmath_think32k_t15_s42_formal200
  RUN_DIR=$R/$EXPERIMENT
  if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
    mv "$RUN_DIR" "${RUN_DIR}_stale_$(date +%m%d%H%M)"
  fi
  cd "$OVERLAY"
  METHOD=grpo TRAIN_FILE="$TRAIN" VAL_FILE="$VALS" RUN_ROOT="$R" \
  EXPERIMENT="$EXPERIMENT" TRAIN_STEPS=200 TRAIN_PROMPTS=16 ROLLOUT_N=8 \
  MAX_TURNS=15 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=32768 \
  MAX_EPISODE_RESPONSE_TOKENS=32768 MAX_MODEL_LEN=45056 ACTOR_MINI_BATCH=8 \
  GPU_MEMORY_UTILIZATION=0.55 AGENT_WORKERS=16 N_GPUS=4 TEST_FREQ=20 \
  SAVE_FREQ=20 MAX_CKPTS=3 RESUME_MODE=auto VAL_BEFORE_TRAIN=true VAL_ROLLOUT_N=1 \
  ENABLE_THINKING=true DATALOADER_WORKERS=0 CUDA_VISIBLE_DEVICES=4,5,6,7 \
  VLLM_USE_FLASHINFER_SAMPLER=0 USE_WANDB=1 LOG_VAL_GENERATIONS=24 \
  TMPDIR=/data2/ssd/yixinshen/tmp-pilot \
  setsid nohup bash scripts/run_simpletir_qwen35_4b.sh \
    > "$R/formal_launcher.log" 2>&1 < /dev/null &
  echo $! > "$R/formal_pid"
  sleep 10
  FP=$(cat "$R/formal_pid")
  if [ ! -d "/proc/$FP" ]; then
    write_state launch_failed
    sleep 600
    continue
  fi
  TMPDIR=/data2/ssd/yixinshen/tmp-pilot setsid nohup "$PY" "$OVERLAY/scripts/monitor_overlong.py" \
    --run-dir "$RUN_DIR" --pid "$FP" \
    --sustained 0.55 --sustained-steps 3 --single 0.65 --interval 60 \
    > "$R/monitor.log" 2>&1 < /dev/null &
  echo $! > "$R/monitor_pid"
  write_state training_launched
  exit 0
done
write_state deadline_exceeded
exit 1
