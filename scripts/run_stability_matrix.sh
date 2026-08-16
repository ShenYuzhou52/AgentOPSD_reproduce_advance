#!/usr/bin/env bash
# 稳定性实验矩阵：多个 seed × 是否启用 AgentOPSD × λ 敏感性
# 用法:
#   SEEDS="0 1 2" bash scripts/run_stability_matrix.sh
#   SEEDS="0 1"   LAM_SWEEP="0.5 0.25 0.1" bash scripts/run_stability_matrix.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

SEEDS="${SEEDS:-0 1 2}"
LAM_SWEEP="${LAM_SWEEP:-0.5}"          # 论文 Figure 3 扫 λ∈{0.5,0.25,0.1,0.01}
STEPS_PER_RUN="${STEPS_PER_RUN:-${TRAIN_STEPS:-150}}"

for seed in ${SEEDS}; do
  for lam in ${LAM_SWEEP}; do
    log "==> seed=${seed} λ=${lam}"
    SEED="${seed}" AGENTOPSD_LAM="${lam}" TRAIN_STEPS="${STEPS_PER_RUN}" \
      bash scripts/train.sh --seed "${seed}" --steps "${STEPS_PER_RUN}"
  done
done

log "==> 矩阵完成。汇总报告:"
python3 scripts/stability_report.py --root "${LOG_ROOT}"

