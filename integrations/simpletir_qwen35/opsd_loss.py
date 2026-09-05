"""Public-author-code OPSD loss, ported without its SDAR trainer dependency."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


def _metric_aggregation(config: ActorConfig, data: TensorDict) -> AggregationType:
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        return AggregationType.SUM
    return AggregationType.MEAN


def sdar_only_loss(
    config: ActorConfig,
    model_output: dict[str, Any],
    data: TensorDict,
    dp_group=None,
    *,
    sdar_coef: float = 0.01,
    gate_beta: float = 5.0,
):
    """Run the public AgentOPSD repository's ``opsd`` ablation in modern Verl.

    It intentionally contains no task-reward policy-gradient term.  The loss is
    the author's confidence-gated single-sample teacher/student gap, plus the
    same optional entropy and reference-KL terms configured for the other arms.
    ``teacher_response_log_probs`` is produced by the same-policy private
    answer-conditioned rollout forward, and has no gradient.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    fields = ["response_mask", "teacher_response_log_probs"]
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    dense = data.select(*fields).to_padded_tensor()
    response_mask = dense["response_mask"].to(bool)
    teacher_log_probs = dense["teacher_response_log_probs"].to(student_log_probs.dtype).detach()
    if teacher_log_probs.shape != student_log_probs.shape:
        raise RuntimeError(
            "OPSD teacher/student response shapes differ: "
            f"teacher={tuple(teacher_log_probs.shape)} student={tuple(student_log_probs.shape)}"
        )

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    aggregate = _metric_aggregation(config, data)

    # Exact SDAR implementation: gate is detached, so the only gradient is
    # through ``-student_log_probs``.  This is the public-code OPSD baseline.
    delta = teacher_log_probs - student_log_probs.detach()
    gate = torch.sigmoid(float(gate_beta) * delta).detach()
    gated_gap = gate * (teacher_log_probs - student_log_probs)
    sdar_loss = agg_loss(
        loss_mat=gated_gap,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    policy_loss = sdar_loss * float(sdar_coef)

    valid_count = response_mask.sum().clamp(min=1)
    metrics: dict[str, Any] = {
        "actor/pg_loss": Metric(value=torch.zeros_like(sdar_loss), aggregation=aggregate),
        "opsd/sdar_loss": Metric(value=sdar_loss, aggregation=aggregate),
        "opsd/sdar_coef": float(sdar_coef),
        "opsd/gate_beta": float(gate_beta),
        "opsd/gate_mean": Metric(value=(gate * response_mask).sum() / valid_count, aggregation=aggregate),
        "opsd/gate_active_ratio": Metric(
            value=((gate > 0.5).to(student_log_probs.dtype) * response_mask).sum() / valid_count,
            aggregation=aggregate,
        ),
        "opsd/teacher_gap_mean": Metric(value=(delta * response_mask).sum() / valid_count, aggregation=aggregate),
    }

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss - config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=aggregate)

    if config.use_kl_loss:
        ref_log_probs = dense["ref_log_prob"].to(student_log_probs.dtype)
        kl_loss = agg_loss(
            loss_mat=kl_penalty(
                logprob=student_log_probs,
                ref_logprob=ref_log_probs,
                kl_penalty=config.kl_loss_type,
            ),
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss + config.kl_loss_coef * kl_loss
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=aggregate)
        metrics["kl_coef"] = float(config.kl_loss_coef)

    return policy_loss, metrics
