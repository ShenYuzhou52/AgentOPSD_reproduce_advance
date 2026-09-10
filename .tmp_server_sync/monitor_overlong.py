"""Live overlong-rate monitor for the SimpleTIR GRPO formal run.

Tails ``<run-dir>/train.log`` once per interval, extracts the per-step
training metrics that matter for episode-budget health, mirrors them into
``<run-dir>/monitor_state.json`` for quick human inspection, and stops the
training process group when the overlong rate breaches the agreed ceiling.

Stop policy (defaults):
  - clip_ratio (response_length/clip_ratio) or simpletir/episode_overlong_ratio
    >= 0.10 for 3 consecutive steps  -> terminate training;
  - either metric >= 0.15 on a single step             -> terminate training.
Warnings at >= 0.08 are recorded but do not stop anything.

Usage:
  python3 scripts/monitor_overlong.py --run-dir <dir> --pid <launcher_pid> \
      [--sustained 0.10] [--sustained-steps 3] [--single 0.15] [--interval 60]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

METRICS = (
    "critic/score/mean",
    "response_length/clip_ratio",
    "response/aborted_ratio",
    "simpletir/episode_overlong_ratio",
    "simpletir/episode_response_tokens_mean",
    "simpletir/episode_response_tokens_max",
    "response_length/max",
)


def parse_log(path: Path, since_byte: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], since_byte
    with path.open("rb") as handle:
        handle.seek(since_byte)
        chunk = handle.read().decode("utf-8", errors="ignore")
        end = handle.tell()
    rows: dict[int, dict] = {}
    val_rows: list[dict] = []
    for line in chunk.splitlines():
        is_val = "val-" in line
        step_m = None
        for needle in ("step:", "global_step:"):
            idx = line.find(needle)
            if idx >= 0:
                digits = ""
                for ch in line[idx + len(needle):]:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    step_m = int(digits)
                    break
        if step_m is None:
            continue
        row = {"step": step_m}
        for key in METRICS:
            marker = f"{key}:"
            pos = line.find(marker)
            if pos < 0:
                continue
            tail = line[pos + len(marker):].strip()
            num = ""
            for ch in tail:
                if ch in "+-0123456789.eE":
                    num += ch
                else:
                    break
            try:
                row[key] = float(num)
            except ValueError:
                pass
        if is_val:
            row["kind"] = "val"
            val_rows.append(row)
        elif "response_length/clip_ratio" in row:
            row["kind"] = "train"
            rows[step_m] = row
    train_rows = [rows[k] for k in sorted(rows)]
    return (train_rows, val_rows), end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid", type=int, default=0, help="launcher process id (a process group leader)")
    parser.add_argument("--sustained", type=float, default=0.10)
    parser.add_argument("--sustained-steps", type=int, default=3)
    parser.add_argument("--single", type=float, default=0.15)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    state_path = run_dir / "monitor_state.json"
    history: list[dict] = []
    val_history: list[dict] = []
    alerts: list[dict] = []
    since_byte = 0
    overlong_streak = 0

    def write_state(stage: str, **extra) -> None:
        payload = {
            "stage": stage,
            "updated_at": datetime.now().isoformat(),
            "pid": args.pid,
            "history": history[-60:],
            "val_history": val_history[-30:],
            "alerts": alerts[-40:],
            **extra,
        }
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        tmp.replace(state_path)

    while True:
        (train_rows, val_rows), since_byte = parse_log(run_dir / "train.log", since_byte)
        history.extend(train_rows)
        val_history.extend(val_rows)
        proc_alive = args.pid == 0 or Path(f"/proc/{args.pid}").exists()
        if not proc_alive:
            write_state("stopped_training_process_gone")
            break

        breached = None
        for row in train_rows:
            clip = row.get("response_length/clip_ratio")
            ep_overlong = row.get("simpletir/episode_overlong_ratio")
            worst = max(x for x in (clip, ep_overlong) if x is not None) if (clip is not None or ep_overlong is not None) else 0.0
            if worst >= args.single:
                breached = (row, f"single-step overlong {worst:.3f} >= {args.single}")
                break
            if worst >= args.sustained:
                overlong_streak += 1
                if overlong_streak >= args.sustained_steps:
                    breached = (row, f"overlong >= {args.sustained} for {overlong_streak} consecutive steps")
                    break
            else:
                overlong_streak = 0
            if 0.08 <= worst < args.sustained:
                alerts.append({"step": row["step"], "level": "warn", "overlong": worst,
                               "at": datetime.now().isoformat()})

        if breached is not None:
            row, reason = breached
            write_state("stopping_overlong", reason=reason, offending=row)
            if args.pid:
                try:
                    os.killpg(args.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(args.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.time() + 120
            while time.time() < deadline and Path(f"/proc/{args.pid}").exists():
                time.sleep(5)
            if Path(f"/proc/{args.pid}").exists() and args.pid:
                try:
                    os.killpg(args.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            write_state("stopped_overlong", reason=reason, offending=row)
            break

        write_state(
            "monitoring",
            consecutive_overlong_steps=overlong_streak,
            last_train_step=(history[-1]["step"] if history else None),
            val_dumps=sorted(
                p.name for p in (run_dir / "val_generations").glob("*.jsonl")
            )[-6:] if (run_dir / "val_generations").is_dir() else [],
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
