"""Schema-only guard for SimpleTIR data passed to the privileged teacher.

The raw Parquet keeps ``reward_model.ground_truth`` for the training-side
scorer.  This module verifies that it stays a sibling of the student prompt,
not a field inside it, before any rollout worker is launched.  Its summaries
contain schema metadata only and never stringify a question or answer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _validate_prompt(prompt: Any, *, source: str) -> None:
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{source}: prompt must be a non-empty chat-message list")
    for position, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise ValueError(f"{source}: prompt message {position} must be a mapping")
        if set(message).intersection({"ground_truth", "answer", "solution", "target"}):
            raise ValueError(f"{source}: student prompt message {position} contains a forbidden answer field")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"{source}: prompt message {position} has an unsupported role")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{source}: prompt message {position} content must be text")


def validate_simpletir_files(paths: Iterable[str | Path], *, batch_size: int = 1024) -> list[dict[str, Any]]:
    """Validate every source row without retaining or printing its content.

    The function deliberately returns only row counts and column names.  It
    streams Arrow batches, so this preflight neither duplicates the datasets
    nor creates a gold-bearing derived file.  Checking every prompt is
    intentional: a single malformed late row could otherwise expose privileged
    answer fields only after a lengthy training run has begun.
    """
    import pyarrow.parquet as pq

    summaries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        file = pq.ParquetFile(path)
        expected = {"prompt", "reward_model", "data_source"}
        columns = set(file.schema_arrow.names)
        missing = expected - columns
        if missing:
            raise ValueError(f"{path}: required SimpleTIR columns are missing: {sorted(missing)}")
        checked = 0
        for batch in file.iter_batches(batch_size=max(int(batch_size), 1)):
            for row in batch.to_pylist():
                _validate_prompt(row["prompt"], source=str(path))
                reward_model = row["reward_model"]
                if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
                    raise ValueError(f"{path}: reward_model.ground_truth is required for private scoring")
                if not isinstance(row["data_source"], str):
                    raise ValueError(f"{path}: data_source must be a string")
                checked += 1
        if checked == 0:
            raise ValueError(f"{path}: empty SimpleTIR Parquet is not a valid training/evaluation source")
        summaries.append(
            {
                "path": str(path),
                "rows": int(file.metadata.num_rows),
                "checked_rows": checked,
                "columns": sorted(columns),
            }
        )
    return summaries
