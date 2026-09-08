"""OPSA（On-Policy Self-Adaptation）损失 —— arXiv:2608.31046 的 Verl 移植。

OPSA 是论文 "Does On-Policy Distillation Really Distill?" 提出的去教师化方法：
作者证明 on-policy 蒸馏（OPD/OPSD）的收益基本来自"压低低对数概率 token"，
而这个信号不需要 teacher——对每条 rollout 中采样 log-prob 最低的 20% token
施加**熵自适应负优势**即可复现该机制（论文式 4/5，OPSA 配置）：

    S         = 每条 rollout 内按采样 log-prob 升序取最低的 20% token 位置
    A_i^dyn   = -3/4 - (1/4)·δ·r_i,    r_i = 2·(H_i - H_min)/(H_max - H_min) - 1
    L         = -(1/|S|) · Σ_{i∈S} ratio_i · A_i^dyn

- A ∈ [-1, -0.5]：位置熵越高（策略越犹豫）负优势越强；δ=1 是论文 OPSA 配置，
  δ=0 退化为固定负优势 -3/4（论文对照组）。
- 无 teacher、无任务奖励、无参考 KL：训练信号完全来自学生自身的熵结构。
  奖励仍照常计算与落盘，但只用于验证/监控，绝不进入损失。
- ratio 沿用 Verl 的 PPO 单边剪裁（与其它三臂一致的优化器行为）；训练只有
  1 个 ppo epoch，首个 minibatch ratio≡1，严格退化为论文的原始形式。

官方实现基于 slime/Megatron（github.com/DripNowhy/On-Policy-Self-Adaptation），
与本仓库的 Verl 管线不兼容，因此这里按论文公式重新实现，而不是引用其代码。
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding

from integrations.simpletir_qwen35.opsd_loss import _metric_aggregation


# 论文默认超参：最低 20% 采样 token、固定优势 -3/4、熵自适应系数 δ=1。
OPSA_LOWEST_FRAC = 0.2
OPSA_ADV_FIX = -0.75
OPSA_DELTA = 1.0


def opsa_token_advantage(
    old_log_probs: torch.Tensor,
    entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    lowest_frac: float = OPSA_LOWEST_FRAC,
    adv_fix: float = OPSA_ADV_FIX,
    delta: float = OPSA_DELTA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按论文式 (4) 计算 OPSA 的逐 token 负优势。

    纯函数，便于 CPU 单测。输入均为 (batch, response_len) 的 padded 张量：
    - ``old_log_probs``：rollout 采样策略的 token log-prob（仅用于选点，detach）；
    - ``entropy``：位置熵（detach 后只进入优势常数，不产生梯度路径）；
    - ``response_mask``：真实响应 token 的 bool mask。

    返回 ``(selected_mask, advantage)``：selected_mask 标出参与损失的 token
    （每行最多 ``lowest_frac`` 比例、至少 1 个），advantage 在选点上为
    ``adv_fix - 0.25·δ·r_i``（r_i ∈ [-1, 1]），其余位置为 0。
    """
    if old_log_probs.shape != entropy.shape or old_log_probs.shape != response_mask.shape:
        raise RuntimeError(
            "OPSA inputs must share (batch, response_len) shape: "
            f"log_probs={tuple(old_log_probs.shape)} entropy={tuple(entropy.shape)} mask={tuple(response_mask.shape)}"
        )
    if not 0.0 < lowest_frac <= 1.0:
        raise RuntimeError(f"OPSA lowest_frac must be in (0, 1], got {lowest_frac!r}")
    if entropy.dtype.is_floating_point and not torch.isfinite(entropy[response_mask]).all():
        raise RuntimeError("OPSA received non-finite entropy inside response_mask; refusing to build advantages")

    batch, _length = old_log_probs.shape
    device = old_log_probs.device
    selected = torch.zeros_like(response_mask)
    advantage = torch.zeros_like(old_log_probs)

    # 论文按单条 rollout 统计选点与熵归一化，逐行处理；行 = 一条轨迹/一个 turn。
    for row in range(batch):
        valid = response_mask[row]
        n = int(valid.sum().item())
        if n == 0:
            continue
        k = max(1, int(lowest_frac * n))
        # 只在真实 token 中选最低 log-prob 的 k 个；padding 位置 log-prob 无意义。
        row_logp = old_log_probs[row][valid]
        threshold_idx = torch.argsort(row_logp)[:k]
        positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)[threshold_idx]
        selected[row, positions] = True

        h = entropy[row][positions]
        h_min, h_max = float(h.min().item()), float(h.max().item())
        if h_max > h_min:
            r = 2.0 * (h - h_min) / (h_max - h_min) - 1.0
        else:
            # 该行选点熵全相等：熵自适应项退化为 0，只剩固定负优势。
            r = torch.zeros_like(h)
        advantage[row, positions] = adv_fix - 0.25 * float(delta) * r

    return selected, advantage.detach()


def opsa_loss(
    config: ActorConfig,
    model_output: dict[str, Any],
    data: TensorDict,
    dp_group=None,
    *,
    lowest_frac: float = OPSA_LOWEST_FRAC,
    adv_fix: float = OPSA_ADV_FIX,
    delta: float = OPSA_DELTA,
):
    """Verl actor 损失入口：完全替换策略梯度项，签名与 ``sdar_only_loss`` 一致。

    与其它臂的关键差异：不读取 teacher 字段、不读取 RM 分数/GRPO 优势、
    不加参考 KL（论文 OPSA 无参考模型前向）。熵由 actor forward 提供
    （``actor_rollout_ref.actor.calculate_entropy=True``），缺失即报错终止，
    拒绝静默退化成普通 GRPO。
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    raw_entropy = model_output.get("entropy")
    if raw_entropy is None:
        raise RuntimeError(
            "OPSA requires token entropy from the actor forward; "
            "set actor_rollout_ref.actor.calculate_entropy=True for the opsa arm"
        )
    entropy = no_padding_2_padding(raw_entropy, data)

    # OPSA 不需要 teacher / ref / RM 字段：只取 mask 与采样 log-prob。
    dense = data.select("response_mask", "old_log_probs").to_padded_tensor()
    response_mask = dense["response_mask"].to(bool)
    old_log_probs = dense["old_log_probs"].detach()

    selected, advantage = opsa_token_advantage(
        old_log_probs,
        entropy.detach(),
        response_mask,
        lowest_frac=lowest_frac,
        adv_fix=adv_fix,
        delta=delta,
    )
    if not bool(selected.any()):
        raise RuntimeError("OPSA selected no tokens; batch has no real response tokens")

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    aggregate = _metric_aggregation(config, data)

    # PPO 单边剪裁的重要性比（与其它三臂相同的 clip 配置）；单 epoch 下首个
    # minibatch 恒为 1，不影响论文语义，仅约束后续 minibatch 的偏离。
    clip_low = float(getattr(config, "clip_ratio_low", 0.2))
    clip_high = float(getattr(config, "clip_ratio_high", 0.2))
    ratio = (student_log_probs - old_log_probs).exp().clamp(1.0 - clip_low, 1.0 + clip_high)
    loss_mat = -ratio * advantage
    # token-mean 聚合 = 论文的 1/|S| 归一化（只在选点内求平均）。
    policy_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=selected,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )

    selected_count = selected.sum().clamp(min=1)
    real_count = response_mask.sum().clamp(min=1)
    metrics: dict[str, Any] = {
        "actor/pg_loss": Metric(value=policy_loss, aggregation=aggregate),
        "opsa/pg_loss": Metric(value=policy_loss, aggregation=aggregate),
        "opsa/lowest_frac": float(lowest_frac),
        "opsa/delta": float(delta),
        "opsa/adv_fix": float(adv_fix),
        "opsa/selected_ratio": Metric(
            value=selected.sum().to(student_log_probs.dtype) / real_count, aggregation=aggregate
        ),
        "opsa/adv_mean": Metric(
            value=(advantage * selected).sum() / selected_count, aggregation=aggregate
        ),
        "opsa/entropy_selected_mean": Metric(
            value=(entropy.detach() * selected).sum() / selected_count, aggregation=aggregate
        ),
        "opsa/ratio_max": Metric(
            value=(ratio * selected).max(), aggregation=AggregationType.MAX
        ),
    }
    return policy_loss, metrics
