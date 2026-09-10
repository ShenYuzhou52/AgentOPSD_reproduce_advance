#!/usr/bin/env bash
# 夜间编排：等 OPSD 结束 → 干净性判定 → merge OPSD ckpt → 全部评测(3 GPU 并行) → 打标记
# 启动方式: nohup bash night_chain.sh >> night_chain.log 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mkdir -p "$NIGHT/markers" "$NIGHT/results" "$NIGHT/merged"
log(){ printf '%s %s\n' "$(date -Is)" "$*"; }

log "waiting for OPSD to exit"
while pgrep -f "experiment_name=opsd_formal200_s42" >/dev/null; do sleep 120; done
log "OPSD process gone"
touch "$NIGHT/markers/OPSD_EXIT"

LAST=$(tail -n1 "$R/opsd/opsd_formal200_s42/metrics.jsonl" | "$PY" -c "import json,sys;print(int(json.load(sys.stdin).get('agentopsd/global_step',0)))" 2>/dev/null || echo 0)
if [ "$LAST" != "200" ]; then
  log "UNCLEAN exit at step $LAST - agent must decide (auto-resume authorized)"
  touch "$NIGHT/markers/OPSD_UNCLEAN"
  exit 4
fi
touch "$NIGHT/markers/OPSD_CLEAN200"
log "clean 200-step exit; merging OPSD ckpts"

for CK in $(ls -d "$R"/opsd/opsd_formal200_s42/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -3); do
  STEP=$(basename "$CK" | sed 's/global_step_//')
  bash "$NIGHT/merge_one.sh" "$CK" "$NIGHT/merged/opsd_$STEP" > "$NIGHT/merged/merge_opsd_$STEP.log" 2>&1 &
done
wait
touch "$NIGHT/markers/MERGE_DONE"
log "all merges done; starting eval matrix"

G=0
for CK in "$NIGHT"/merged/*/; do
  [ -f "$CK/MERGE_OK" ] || { log "skip unmerged $CK"; continue; }
  NAME=$(basename "$CK")
  for PAIR in "aime:$AIME" "aime25:$AIME25"; do
    DSN=${PAIR%%:*}; DSP=${PAIR#*:}
    GPU=$((G % 3)); G=$((G+1))
    log "eval $NAME $DSN on gpu$GPU"
    bash "$NIGHT/eval_ckpt.sh" "$NIGHT/merged/$NAME" "$DSP" "$GPU" "$NIGHT/results/${NAME}_${DSN}" "${NAME}_${DSN}" > "$NIGHT/results/${NAME}_${DSN}.runlog" 2>&1 &
    sleep 45
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 20; done
  done
done
wait
touch "$NIGHT/markers/EVALS_DONE"
log "ALL EVALS DONE"
