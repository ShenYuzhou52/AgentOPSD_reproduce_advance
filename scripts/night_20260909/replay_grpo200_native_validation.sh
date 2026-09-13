#!/usr/bin/env bash
# Re-run GRPO step-200 validation through the original Verl/TQ/SimpleTIR path.
# The saved command is shell-escaped one argument per line, so reconstruct its
# original argv and override only checkpoint/resume/output/validation settings.
set -euo pipefail

ROOT=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907
RUN="$ROOT/grpo/grpo_formal200_s42"
CKPT="$RUN/checkpoints/global_step_200"
OUT="$ROOT/night_20260909/native_replay/grpo_200_aime25"
COMMAND="$RUN/command.sh"

mkdir -p "$OUT"
[[ -f "$COMMAND" ]] || { echo "missing command: $COMMAND" >&2; exit 2; }
[[ -d "$CKPT/actor" ]] || { echo "missing actor checkpoint: $CKPT" >&2; exit 2; }
EXTRA_ARGS=("$@")

# command.sh is produced by printf %q, so this restores its original argv.
ARGS_TEXT=$(tr '\n' ' ' < "$COMMAND")
eval "set -- $ARGS_TEXT"

exec "$@" \
  "trainer.resume_mode=resume_path" \
  "trainer.resume_from_path=$CKPT" \
  "trainer.val_only=true" \
  "trainer.val_before_train=true" \
  "trainer.save_freq=-1" \
  "trainer.experiment_name=grpo_200_native_validation_replay" \
  "trainer.default_local_dir=$OUT/checkpoints" \
  "trainer.log_val_generations=1" \
  "trainer.validation_data_dir=$OUT/generations" \
  "+simpletir.metrics_jsonl=$OUT/metrics.jsonl" \
  "${EXTRA_ARGS[@]}"
