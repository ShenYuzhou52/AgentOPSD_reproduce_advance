"""Verl V1 trainer hooks for the three-way SimpleTIR comparison.

The hook is intentionally strict: malformed sessions, absent teacher values,
or any synthetic row that reaches AgentOPSD credit abort the step.  A malformed
AgentOPSD step must never silently fall back to GRPO.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

from agentopsd.credit import AgentOPSDConfig, reshape_advantages
from agentopsd.trainer.monitor import AgentOPSDMonitor, append_jsonl
from integrations.simpletir_qwen35.opsd_loss import sdar_only_loss
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.padding import response_to_nested


_METHODS = {"grpo", "opsd_author_code", "agentopsd"}


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _session_key(row_key: str) -> tuple[str, str, int]:
    fields = str(row_key).rsplit("_", 2)
    if len(fields) != 3:
        raise RuntimeError(f"SimpleTIR expected TransferQueue key uid_session_turn, got {row_key!r}")
    uid, session, turn_text = fields
    try:
        turn = int(turn_text)
    except ValueError as exc:
        raise RuntimeError(f"SimpleTIR turn suffix is not an integer: {row_key!r}") from exc
    if turn < 0:
        raise RuntimeError(f"SimpleTIR turn index must be non-negative: {row_key!r}")
    return uid, session, turn


class SimpleTIRTrainer(PPOTrainerSync):
    """Synchronous V1 trainer with author-code OPSD or AgentOPSD hooks."""

    def __init__(self, config):
        super().__init__(config)
        self.method = str(config.get("simpletir", {}).get("method", "grpo"))
        if self.method not in _METHODS:
            raise ValueError(f"simpletir.method must be one of {sorted(_METHODS)}, got {self.method!r}")
        simpletir = config.get("simpletir", {})
        self.monitor_path = str(simpletir.get("metrics_jsonl", ""))
        if self.monitor_path and not self.monitor_path.startswith("/data2/"):
            raise ValueError("simpletir.metrics_jsonl must reside on /data2, never the root filesystem")
        self.monitor_every = max(int(simpletir.get("monitor_every", 1)), 1)
        self._monitor: AgentOPSDMonitor | None = None
        self._last_credit_diag: dict[str, float] = {}
        self._agentopsd_cfg = AgentOPSDConfig.from_dict(
            OmegaConf.to_container(simpletir.get("agentopsd", {}), resolve=True)
        )
        self._agentopsd_cfg.enabled = self.method == "agentopsd"

    def on_init_end(self):
        super().on_init_end()
        self._monitor = AgentOPSDMonitor(total_steps=self.total_training_steps)
        if self.method == "opsd_author_code":
            actor_cfg = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
            actor_cfg.model_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
            simpletir = self.config.simpletir
            self.actor_rollout_wg.set_loss_fn(
                partial(
                    sdar_only_loss,
                    config=actor_cfg,
                    sdar_coef=float(simpletir.get("opsd_sdar_coef", 0.01)),
                    gate_beta=float(simpletir.get("opsd_gate_beta", 5.0)),
                )
            )

    def _get_extra_fields(self, batch) -> list[dict[str, Any]]:
        values = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])[
            "extra_fields"
        ]
        entries = _as_list(values)
        if len(entries) != len(batch.keys):
            raise RuntimeError("TransferQueue extra_fields count does not match batch keys")
        return [entry if isinstance(entry, dict) else {} for entry in entries]

    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Accept only Verl's zero-loss synthetic padding, never copied turns.

        This is the explicit guard for the historical P0 failure mode.  The
        upstream V1 helper may add rows only to meet the actor's divisibility
        constraint; real generated turn keys must remain a permutation of the
        input and every added row must carry an all-zero response mask.
        """
        real_keys_before = list(batch.keys)
        if len(real_keys_before) != len(set(real_keys_before)):
            raise RuntimeError("SimpleTIR received duplicate real turn keys before batch balancing")
        balanced = super()._balance_batch(batch, metrics, logging_prefix, keep_minibatch)
        real_keys_after = [key for key, tag in zip(balanced.keys, balanced.tags, strict=True) if not tag.get("is_padding")]
        if len(real_keys_after) != len(set(real_keys_after)) or set(real_keys_after) != set(real_keys_before):
            raise RuntimeError("batch balancing duplicated, dropped, or rewrote a real SimpleTIR turn")
        padding_keys = [key for key, tag in zip(balanced.keys, balanced.tags, strict=True) if tag.get("is_padding")]
        if padding_keys:
            padding = tq.kv_batch_get(
                keys=padding_keys,
                partition_id=balanced.partition_id,
                select_fields=["response_mask", "rm_scores"],
            ).to_padded_tensor()
            if padding["response_mask"].any() or padding["rm_scores"].any():
                raise RuntimeError("synthetic SimpleTIR padding is not zero-loss; refusing to train")
        metrics.update(
            {
                "simpletir/real_turns_preserved": float(len(real_keys_after)),
                "simpletir/synthetic_padding_turns": float(len(padding_keys)),
                "simpletir/p0_no_real_turn_duplication": 1.0,
            }
        )
        return balanced

    def _write_teacher_response_logprobs(self, batch, metrics: dict) -> tuple[TensorDict, list[dict[str, Any]], np.ndarray]:
        """Validate and write response-aligned teacher scores for the actor update."""
        tensor_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask", "old_log_probs", "advantages", "returns"],
        )
        response_mask_nested = tensor_data["response_mask"]
        dense = tensor_data.to_padded_tensor()
        response_mask = dense["response_mask"].bool()
        old_log_probs = dense["old_log_probs"]
        extras = self._get_extra_fields(batch)
        is_padding = np.asarray([bool(tag.get("is_padding", False)) for tag in batch.tags], dtype=bool)
        if response_mask.size(0) != len(batch.keys) or len(extras) != len(batch.keys):
            raise RuntimeError("TransferQueue batch dimensions are inconsistent")

        teacher = torch.zeros_like(old_log_probs)
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        accuracy_by_session: dict[tuple[str, str], list[float]] = {}
        for row, row_key in enumerate(batch.keys):
            if is_padding[row]:
                continue
            uid, session, turn = _session_key(row_key)
            entry = extras[row]
            if int(entry.get("turn_step", -1)) != turn:
                raise RuntimeError(
                    f"SimpleTIR turn metadata/key mismatch for {row_key!r}: "
                    f"metadata={entry.get('turn_step')!r}, key={turn}"
                )
            raw_teacher = entry.get("teacher_response_log_probs")
            if raw_teacher is None:
                raise RuntimeError(f"{self.method} requires teacher_response_log_probs; refusing a silent GRPO fallback")
            teacher_values = _as_list(raw_teacher)
            real_len = int(response_mask[row].sum().item())
            if real_len <= 0 or len(teacher_values) != real_len:
                raise RuntimeError(
                    f"teacher/token mask length mismatch for {row_key!r}: teacher={len(teacher_values)}, mask={real_len}"
                )
            teacher[row, :real_len] = torch.as_tensor(teacher_values, dtype=teacher.dtype, device=teacher.device)
            if not torch.isfinite(teacher[row, :real_len]).all():
                raise RuntimeError(f"non-finite teacher log-probability for {row_key!r}")
            reward_info = entry.get("reward_extra_info")
            if not isinstance(reward_info, dict) or "answer_accuracy" not in reward_info:
                raise RuntimeError(f"SimpleTIR missing binary answer_accuracy for {row_key!r}")
            answer_accuracy = float(reward_info["answer_accuracy"])
            if answer_accuracy not in (0.0, 1.0):
                raise RuntimeError(f"SimpleTIR answer_accuracy must be binary, got {answer_accuracy!r}")
            key = (uid, session)
            groups.setdefault(key, []).append((turn, row))
            accuracy_by_session.setdefault(key, []).append(answer_accuracy)

        if not groups:
            raise RuntimeError("SimpleTIR batch contains no active trajectories")
        for key, turns in groups.items():
            ordered = sorted(turns)
            expected = list(range(len(ordered)))
            actual = [turn for turn, _ in ordered]
            if actual != expected:
                raise RuntimeError(
                    f"SimpleTIR session {key!r} has missing/duplicated turns {actual}; no row duplication is permitted"
                )
            accuracies = accuracy_by_session[key]
            if len(set(accuracies)) != 1:
                raise RuntimeError(f"SimpleTIR session {key!r} has inconsistent terminal success labels")

        delta = (teacher - old_log_probs)[response_mask & torch.as_tensor(~is_padding, device=response_mask.device).unsqueeze(1)]
        metrics.update(
            {
                "simpletir/teacher_forward_applied": 1.0,
                "simpletir/teacher_student_gap_mean": float(delta.mean().item()) if delta.numel() else 0.0,
                "simpletir/teacher_student_gap_rms": float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0,
                "simpletir/active_turn_count": float((~is_padding).sum()),
                "simpletir/padding_turn_count": float(is_padding.sum()),
            }
        )
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"teacher_response_log_probs": response_to_nested(teacher, response_mask_nested)},
                batch_size=len(batch),
            ),
        )
        return dense, extras, is_padding

    def _reshape_agentopsd(self, batch, dense: TensorDict, extras: list[dict[str, Any]], is_padding: np.ndarray, metrics: dict):
        active = ~is_padding
        active_tensor = torch.as_tensor(active, dtype=torch.bool, device=dense["advantages"].device)
        teacher_values = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["teacher_response_log_probs"]
        )["teacher_response_log_probs"].to_padded_tensor()
        uid, traj_uid, turn_step, successes = [], [], [], []
        session_advantages: dict[str, list[float]] = {}
        for row, row_key in enumerate(batch.keys):
            if not active[row]:
                continue
            group, session, turn = _session_key(row_key)
            mask = dense["response_mask"][row].bool()
            first = int(mask.float().argmax().item())
            row_advantage = float(dense["advantages"][row, first].item())
            session_advantages.setdefault(f"{group}:{session}", []).append(row_advantage)
            reward_info = extras[row]["reward_extra_info"]
            uid.append(group)
            traj_uid.append(f"{group}:{session}")
            turn_step.append(turn)
            # B0 is defined over binary task success, not the 0.5 no-tool-use reward.
            successes.append(float(reward_info["answer_accuracy"]))
        for session, values in session_advantages.items():
            if max(values) - min(values) > 1e-5:
                raise RuntimeError(f"GRPO advantage was not broadcast consistently across real session {session!r}")

        reshaped, diag = reshape_advantages(
            advantages=dense["advantages"][active_tensor],
            teacher_log_probs=teacher_values[active_tensor],
            student_log_probs=dense["old_log_probs"][active_tensor],
            response_mask=dense["response_mask"][active_tensor],
            uid=np.asarray(uid, dtype=object),
            traj_uid=np.asarray(traj_uid, dtype=object),
            turn_step=np.asarray(turn_step, dtype=np.int64),
            episode_rewards=np.asarray(successes, dtype=np.float64),
            cfg=self._agentopsd_cfg,
        )
        dense["advantages"][active_tensor] = reshaped
        dense["returns"][active_tensor] = reshaped
        diag.update(
            {
                "agentopsd/reshape_applied": 1.0,
                "agentopsd/trajectory_metadata_validated": 1.0,
                "agentopsd/padding_turn_count": float(is_padding.sum()),
                "agentopsd/active_turn_count": float(active.sum()),
                "agentopsd/teacher_student_gap_mean": metrics["simpletir/teacher_student_gap_mean"],
                "agentopsd/teacher_student_gap_rms": metrics["simpletir/teacher_student_gap_rms"],
            }
        )
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    "advantages": response_to_nested(dense["advantages"], dense["response_mask"]),
                    "returns": response_to_nested(dense["returns"], dense["response_mask"]),
                },
                batch_size=len(batch),
            ),
        )
        metrics.update(diag)
        self._last_credit_diag = diag

    def _compute_advantage(self, batch, metrics):
        batch = super()._compute_advantage(batch, metrics)
        self._last_credit_diag = {"agentopsd/reshape_applied": 0.0}
        if self.method == "grpo":
            return batch
        dense, extras, is_padding = self._write_teacher_response_logprobs(batch, metrics)
        if self.method == "agentopsd":
            self._reshape_agentopsd(batch, dense, extras, is_padding, metrics)
        return batch

    def _rollout_monitor_metrics(self, batch) -> dict[str, float]:
        extras = self._get_extra_fields(batch)
        active = [not bool(tag.get("is_padding", False)) for tag in batch.tags]
        selected = [entry for entry, keep in zip(extras, active, strict=True) if keep]
        if not selected:
            return {"simpletir/active_turn_count": 0.0}
        code = np.asarray([float(entry.get("code_present", 0.0)) for entry in selected])
        sandbox_ok = np.asarray([float(entry.get("sandbox_ok", 0.0)) for entry in selected])
        void = np.asarray([float(entry.get("is_void_turn", 0.0)) for entry in selected])
        answers = []
        for entry in selected:
            info = entry.get("reward_extra_info", {})
            if isinstance(info, dict) and "answer_accuracy" in info:
                answers.append(float(info["answer_accuracy"]))
        return {
            "simpletir/code_present_ratio": float(code.mean()),
            "simpletir/sandbox_ok_ratio": float(sandbox_ok[code > 0].mean()) if np.any(code > 0) else 0.0,
            "simpletir/void_turn_ratio": float(void.mean()),
            "simpletir/answer_accuracy": float(np.mean(answers)) if answers else 0.0,
        }

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        metrics.update(self._rollout_monitor_metrics(batch))
        if self._monitor is None or global_steps % self.monitor_every:
            return
        record = self._monitor.record(self._last_credit_diag, int(global_steps), metrics)
        record["simpletir/method"] = self.method
        record["simpletir/monitor_schema"] = 1
        for key, value in metrics.items():
            if key.startswith("simpletir/") or key.startswith("opsd/") or key.startswith("val"):
                if isinstance(value, (int, float, np.number)):
                    record[key] = float(value)
        append_jsonl(self.monitor_path, record)
