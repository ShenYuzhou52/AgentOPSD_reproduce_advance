"""Step-level AgentOPSD collapse monitoring.

The monitor is intentionally diagnostic rather than an automatic stop mechanism.
It records the raw signals needed to distinguish belief saturation, dead credit,
non-finite advantages and ordinary reward/policy changes during late training.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping


_TRAINING_METRICS = (
    "actor/entropy_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/lr",
    "agentopsd/teacher_student_gap_mean",
    "agentopsd/teacher_student_gap_rms",
    "episode/reward/mean",
    "episode/reward/max",
    "episode/reward/min",
    "episode/length/mean",
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
)


def _as_finite_float(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().mean().item()
        elif hasattr(value, "item"):
            value = value.item()
        value = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return value if math.isfinite(value) else None


class AgentOPSDMonitor:
    """Build one complete, JSON-serializable record per optimizer step."""

    def __init__(self, total_steps: int, *, late_fraction: float = 0.8, streak: int = 3):
        self.total_steps = max(int(total_steps), 1)
        self.late_fraction = float(late_fraction)
        self.required_streak = max(int(streak), 1)
        self.collapse_streak = 0
        self.late_collapse_streak = 0

    def record(self, diag: Mapping[str, Any], step: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        nonfinite = 0
        for key, value in diag.items():
            scalar = _as_finite_float(value)
            record[str(key)] = scalar
            nonfinite += int(scalar is None)

        step = int(step)
        progress = min(max(step / self.total_steps, 0.0), 1.0)
        is_late = progress >= self.late_fraction
        record.update({
            "agentopsd/global_step": float(step),
            "agentopsd/total_steps": float(self.total_steps),
            "agentopsd/progress": progress,
            "agentopsd/is_late_training": float(is_late),
        })

        for key in _TRAINING_METRICS:
            if key in metrics:
                scalar = _as_finite_float(metrics[key])
                record[key] = scalar
                nonfinite += int(scalar is None)

        saturation = float(record.get("agentopsd/belief_saturation_ratio") or 0.0)
        revision = float(record.get("agentopsd/belief_revision_abs_mean") or 0.0)
        credit = float(record.get("agentopsd/credit_abs_mean") or 0.0)
        mixed_groups = float(record.get("agentopsd/group_success_mixed_ratio") or 0.0)
        adv_nonfinite = float(record.get("agentopsd/adv_nonfinite_ratio") or 0.0)
        adv_abs_max = float(record.get("agentopsd/adv_abs_max") or 0.0)
        adv_large_ratio = float(record.get("agentopsd/adv_large_ratio") or 0.0)
        adv_large_threshold = float(record.get("agentopsd/adv_large_threshold") or 100.0)

        belief_saturated = saturation >= 0.95
        # All-success/all-failure groups have no within-group GRPO contrast and
        # naturally produce zero credit. Only call this collapse when at least
        # one group in the batch is informative.
        credit_dead = (
            mixed_groups > 0.0
            and belief_saturated
            and revision <= 1e-4
            and credit <= 1e-4
        )
        has_nonfinite = nonfinite > 0 or adv_nonfinite > 0.0
        adv_explosion = adv_abs_max > adv_large_threshold or adv_large_ratio > 0.0
        signal = credit_dead or has_nonfinite
        if signal:
            self.collapse_streak += 1
        else:
            self.collapse_streak = 0
        if is_late and signal:
            self.late_collapse_streak += 1
        elif not is_late:
            self.late_collapse_streak = 0
        else:
            self.late_collapse_streak = 0

        record.update({
            "agentopsd/collapse/belief_saturated": float(belief_saturated),
            "agentopsd/collapse/credit_dead": float(credit_dead),
            "agentopsd/collapse/nonfinite": float(has_nonfinite),
            "agentopsd/numerical/adv_explosion": float(adv_explosion),
            "agentopsd/numerical/alert": float(adv_explosion or has_nonfinite),
            "agentopsd/collapse/signal": float(signal),
            "agentopsd/collapse/streak": float(self.collapse_streak),
            "agentopsd/collapse/alert": float(self.collapse_streak >= self.required_streak),
            "agentopsd/collapse/late_signal": float(is_late and signal),
            "agentopsd/collapse/late_streak": float(self.late_collapse_streak),
            "agentopsd/collapse/late_alert": float(
                self.late_collapse_streak >= self.required_streak
            ),
        })
        return record


def append_jsonl(path: str | None, record: Mapping[str, Any]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
