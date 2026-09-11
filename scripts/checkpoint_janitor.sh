#!/usr/bin/env bash
# Checkpoint janitor: keep a model-only (no optimizer states) copy of every
# actor checkpoint before verl's rolling deletion (max_actor_ckpt_to_keep=3)
# removes the originals.  Gradients are never stored in checkpoints, so
# "model + metadata only" is the minimal complete param snapshot.
#
# Result:
#   <run>/checkpoints/        verl-managed, full ckpt (model+optim), newest 3
#   <run>/checkpoints_keep/   model-only copies of EVERY step, never deleted
set -u

RUN_DIR=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/grpo_deepmath_think32k_t15_s42_formal200
KEEP=$RUN_DIR/checkpoints_keep
STATE=$RUN_DIR/janitor_state.json
LOCK=$RUN_DIR/janitor.lock
MIN_FREE_GB=300
INTERVAL=600

mkdir -p "$KEEP"
exec 9>"$LOCK"
flock -n 9 || exit 0   # single instance

while true; do
  free_gb=$(df -BG /data2 | awk 'NR==2{gsub("G","",$4); print $4}')
  # verl updates this pointer only after a checkpoint is fully written
  latest=$(cat "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  preserved=""
  skipped=""
  for ck in $(ls -d "$RUN_DIR"/checkpoints/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
    name=$(basename "$ck")
    step=${name#global_step_}
    [ "$step" -le "$latest" ] || continue   # not yet complete
    [ -d "$ck/actor" ] || continue
    if [ -d "$KEEP/$name" ] && [ -f "$KEEP/$name/.complete" ]; then
      skipped="$skipped $step"
      continue
    fi
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
      echo "$(date -Is) disk low (${free_gb}G), not preserving $step" >> "$RUN_DIR/janitor.log"
      continue
    fi
    tmp="$KEEP/.tmp_$name"
    rm -rf "$tmp"
    mkdir -p "$tmp/actor"
    # model shards + configs + sampler state; drop optimizer moments only
    find "$ck" -maxdepth 1 -type f ! -name "optim_*" -exec cp {} "$tmp/" \; 2>>"$RUN_DIR/janitor.log"
    cp "$ck"/actor/model_* "$ck"/actor/*.json "$tmp/actor/" 2>/dev/null >>"$RUN_DIR/janitor.log" 2>&1
    touch "$tmp/.complete"
    mv "$tmp" "$KEEP/$name" 2>>"$RUN_DIR/janitor.log"
    preserved="$preserved $step"
  done
  cat > "$STATE" <<EOS
{"updated_at": "$(date -Is)", "free_gb": $free_gb, "kept": "$(ls $KEEP 2>/dev/null | grep global_step | sort -t_ -k3 -n | tr '\n' ' ')", "last_preserved": "$preserved", "last_cycle": "$(date -Is)"}
EOS
  [ -n "$preserved" ] && echo "$(date -Is) preserved:$preserved skipped:$skipped free:${free_gb}G" >> "$RUN_DIR/janitor.log"
  sleep "$INTERVAL"
done
