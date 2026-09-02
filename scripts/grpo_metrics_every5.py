#!/usr/bin/env python3
"""Extract durable every-N-step GRPO snapshots from a running verl train.log.

The trainer itself keeps its original per-step console/W&B logging and its
native validation cadence.  This CPU-only sidecar merely copies the complete
structured line at requested validation steps into JSONL; it never evaluates,
uses a GPU, or controls the training process.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TASK_RUNNER = re.compile(r"\(TaskRunner pid=(?P<pid>\d+)\)")
STEP = re.compile(r"\bstep:(?P<step>\d+)\s+-\s+")
METRIC = re.compile(
    r"(?:^|\s-\s)(?P<key>[A-Za-z_][A-Za-z0-9_./-]*):"
    r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taskrunner-pid", required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--stop-at", type=int, default=150)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def prior_steps(output: Path) -> set[int]:
    if not output.exists():
        return set()
    seen: set[int] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            seen.add(int(json.loads(line)["training/global_step"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return seen


def parse_metrics(line: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for match in METRIC.finditer(line):
        try:
            value = float(match.group("value"))
        except ValueError:
            value = float("nan")
        metrics[match.group("key")] = value if math.isfinite(value) else None
    return metrics


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.stop_at <= 0 or args.poll_seconds <= 0:
        raise SystemExit("interval, stop-at, and poll-seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = prior_steps(args.output)
    offset = 0
    while args.stop_at not in seen:
        try:
            with args.log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                lines = handle.readlines()
                offset = handle.tell()
        except FileNotFoundError:
            lines = []
        for raw_line in lines:
            line = ANSI.sub("", raw_line)
            runner = TASK_RUNNER.search(line)
            step_match = STEP.search(line)
            if not runner or not step_match or runner.group("pid") != args.taskrunner_pid:
                continue
            step = int(step_match.group("step"))
            if step % args.interval != 0 or step in seen:
                continue
            metrics = parse_metrics(line)
            # trainer.test_freq=5 means each retained snapshot must contain the
            # held-out success metric; otherwise wait rather than record a
            # partial, non-validation line.
            if "val/success_rate" not in metrics:
                continue
            record: dict[str, object] = {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_taskrunner_pid": int(args.taskrunner_pid),
                "source": "existing trainer test_freq=5 metrics; CPU-only extractor",
            }
            record.update(metrics)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
            seen.add(step)
            print(
                f"recorded step={step} val_success={record.get('val/success_rate')} "
                f"grad_norm={record.get('actor/grad_norm')}",
                flush=True,
            )
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
