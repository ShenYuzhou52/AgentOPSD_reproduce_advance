#!/usr/bin/env bash
# 一次性只读状态快照：夜间定时任务每轮先跑这个，输出供 agent 判读
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
echo "== $(date -Is)"
echo "-- main_tir procs:"
pgrep -af "main_tir" | cut -c1-130 || echo "NO main_tir PROCESS"
for m in grpo/grpo_formal200_s42 agentopsd/agentopsd_formal200_s42 opsd/opsd_formal200_s42 opsa/opsa_formal200_s42; do
  D=$R/$m
  [ -f "$D/metrics.jsonl" ] || continue
  N=$(wc -l < "$D/metrics.jsonl")
  LAST=$(tail -n 1 "$D/metrics.jsonl" | "$PY" -c "import json,sys;d=json.load(sys.stdin);print(int(d.get('agentopsd/global_step',-1)), round(float(d.get('critic/score/mean',-1)),4), int(d.get('agentopsd/collapse/signal',0)))" 2>/dev/null || echo "parse-fail")
  echo "-- $m: lines=$N last(step score collapse)= $LAST"
done
echo "-- progress lines:"
for m in opsd/opsd_formal200_s42 opsa/opsa_formal200_s42; do
  grep -o "Training Progress:  [0-9]*%[^]]*" "$R/$m/train.log" 2>/dev/null | tail -1 | sed "s|^|  $m |"
done
echo "-- gpu:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "-- disk:"; df -h /data2 | tail -1
echo "-- markers:"; ls "$NIGHT/markers/" 2>/dev/null || echo none
echo "-- results:"; cat "$NIGHT/results/summary.tsv" 2>/dev/null || echo none
echo "-- chain log tail:"; tail -n 8 "$NIGHT/night_chain.log" 2>/dev/null || echo "chain not started"
echo "-- chain alive:"; pgrep -af "night_chain.sh" || echo "chain NOT running"
