"""Public-author-code OPSD loss, ported without its SDAR trainer dependency.

这是 `opsd_author_code` 消融臂的核心：把 AgentOPSD 作者公开仓库里的 OPSD
脚本移植到现代 Verl，不引入其整个 SDAR 训练器。关键事实（README 里也有
声明）：论文附录 D 写的是分布匹配，而公开代码实际配置是"策略梯度系数为 0 +
置信度门控的自蒸馏损失"——我们复现的是后者，因此臂名叫 opsd_author_code
而不是"论文精确版 OPSD"。

损失结构：
    L = sdar_coef · Σ gate · (logπ_teacher − logπ_student)  [+ 熵正则] [+ 参考KL]

teacher log-prob 来自 agent loop 里带金答案的私有前向（无梯度、已与 student
token 逐位对齐）。与 AgentOPSD 臂的区别：OPSD 用 teacher 直接当蒸馏目标且
完全砍掉任务奖励梯度；AgentOPSD 保留 GRPO 梯度，只用 teacher 差值做
优势重塑。
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


# 根据数据并行与 token 统计选择指标聚合方式。
def _metric_aggregation(config: ActorConfig, data: TensorDict) -> AggregationType:
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        return AggregationType.SUM
    return AggregationType.MEAN


# 实现 AgentOPSD 消融中的 OPSD/SDAR 专用损失。
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
    # 将 actor 输出还原为带 padding 的序列，便于逐 token 对齐。
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # 只取计算 OPSD 与可选 KL 所需的最小字段集合。
    fields = ["response_mask", "teacher_response_log_probs"]
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    dense = data.select(*fields).to_padded_tensor()
    response_mask = dense["response_mask"].to(bool)
    # teacher log-prob 作为无梯度监督信号，不能反传到 rollout 服务。
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
    # 门控的直观含义：delta>0 表示 teacher（看过答案）比 student 更确信这个
    # 动作——它是"朝向成功轨迹的方向"，sigmoid(beta*delta) 给它接近 1 的
    # 权重；delta<0 的 token 被压低权重，避免把学生往错误方向硬拉。
    delta = teacher_log_probs - student_log_probs.detach()
    # 门控本身 detach，梯度只通过 student 的负对数概率传递。
    gate = torch.sigmoid(float(gate_beta) * delta).detach()
    gated_gap = gate * (teacher_log_probs - student_log_probs)
    # 在响应 token mask 内聚合单样本的门控差异。
    sdar_loss = agg_loss(
        loss_mat=gated_gap,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )
    # 用 sdar 系数缩放得到该消融臂的主策略损失。
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

    # 沿用 Verl 的熵正则项，保持各实验臂的优化设置一致。
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss = policy_loss - config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=aggregate)

    # 若启用参考策略 KL，则把同一正则项加入 OPSD 总损失。
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
