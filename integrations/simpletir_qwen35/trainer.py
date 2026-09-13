"""Verl V1 trainer hooks for the three-way SimpleTIR comparison.

本文件是四组消融（grpo / opsd_author_code / agentopsd / opsa）在训练侧的唯一分岔点。
继承 Verl 的同步 PPO trainer，在四个位置介入：

- ``_balance_batch``      守卫上游的 batch 平衡，禁止复制真实轨迹行（历史 P0 事故）；
- ``_compute_advantage``  GRPO 优势计算后，按 method 决定是否叠加 teacher / credit 分支；
- ``_rollout_monitor_metrics`` 把 agent loop 记录的逐 turn 诊断聚合成训练指标；
- ``_compute_metrics``    把 simpletir/agentopsd/val 指标落盘到 JSONL，供事后画曲线。

设计原则是“宁可失败也不静默降级”：teacher 缺失、turn 元数据错位、padding 行
带非零损失等任何破坏消融可比性的情况都直接抛错终止该 step，而不是悄悄退回
GRPO——否则实验跑完了才发现某一臂其实没走自己的方法。
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
from integrations.simpletir_qwen35.opsa_loss import opsa_loss
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.padding import response_to_nested


# 消融臂白名单。启动脚本以 +simpletir.method 注入；不在名单内的名字立即报错，
# 防止拼写错误（例如 "opsd"）悄悄落入某个默认损失分支，跑出无法归因的结果。
_METHODS = {"grpo", "opsd_author_code", "agentopsd", "opsa"}


def _as_list(value: Any) -> list[Any]:
    """TransferQueue 的取值可能是 numpy 数组 / tensor / list，统一成 python list。

    extra_fields 里的 teacher log-prob 在序列化往返后会变成 ndarray，而后续
    校验（长度比较、isfinite）需要普通的数值容器。
    """
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _session_key(row_key: str) -> tuple[str, str, int]:
    """把 TransferQueue 的行键 ``uid_session_turn`` 拆成 (题目组, 轨迹, turn 序号)。

    AgentOPSD 的信用分配完全依赖这三个字段：uid 是 GRPO 的同题采样组，
    session 是一次多轮 episode，turn 是该 episode 内的轮次。键格式一旦错位，
    credit 就会记到别的轨迹头上，因此这里用严格解析（三段、turn 非负整数），
    任何不合法键直接终止训练而不是猜测修复。
    """
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
    """三条消融路径共用的同步 V1 trainer：

    - grpo            纯 Verl GRPO，无 teacher 请求，优势即组内标准化奖励；
    - opsd_author_code 用作者公开代码的门控自蒸馏损失替换策略梯度（见 opsd_loss.py）；
    - agentopsd       GRPO 之上叠加"答案条件 teacher 前向 + 回合级优势重塑"；
    - opsa            无 teacher/奖励/KL 的熵自适应负优势（见 opsa_loss.py，arXiv:2608.31046）。
    """

    def __init__(self, config):
        super().__init__(config)
        self.method = str(config.get("simpletir", {}).get("method", "grpo"))
        if self.method not in _METHODS:
            raise ValueError(f"simpletir.method must be one of {sorted(_METHODS)}, got {self.method!r}")
        simpletir = config.get("simpletir", {})
        # 逐 step 诊断 JSONL 的输出路径；空串表示只在 console 打印。
        self.monitor_path = str(simpletir.get("metrics_jsonl", ""))
        # 指标必须落在数据盘：训练机上根分区很小，写满会连带杀死正在跑的任务。
        if self.monitor_path and not self.monitor_path.startswith("/data2/"):
            raise ValueError("simpletir.metrics_jsonl must reside on /data2, never the root filesystem")
        # monitor_every 控制 JSONL 落盘频率（默认每步），长跑时可调大省 IO。
        self.monitor_every = max(int(simpletir.get("monitor_every", 1)), 1)
        self._monitor: AgentOPSDMonitor | None = None
        # 最近一次 credit 重塑的诊断字典；metrics 钩子读取它写入 JSONL。
        self._last_credit_diag: dict[str, float] = {}
        # 把 Hydra 里的 agentopsd.* 超参（lam/b/gamma/eps 等）转成强类型配置，
        # 键名拼错在 from_dict 里就会报错，而不是默默用默认值跑完实验。
        self._agentopsd_cfg = AgentOPSDConfig.from_dict(
            OmegaConf.to_container(simpletir.get("agentopsd", {}), resolve=True)
        )
        # credit 重塑只属于 agentopsd 臂；enabled 由 method 而非配置控制，
        # 防止实验臂和超参互相配错。
        self._agentopsd_cfg.enabled = self.method == "agentopsd"

    def on_init_end(self):
        """worker 全部就绪后注册方法专属组件。

        actor worker 在此之前还不存在，所以 OPSD 的损失替换必须放在这里而不是
        __init__：把 verl 默认的策略梯度损失换成 sdar_only_loss（作者公开实现），
        sdar_coef / gate_beta 与作者脚本一致。grpo / agentopsd 保持默认损失。
        """
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
        if self.method == "opsa":
            # OPSA 同样整体替换策略梯度损失，但不需要 teacher（超参见 opsa_loss.py）。
            # calculate_entropy 必须开启：OPSA 的优势由 token 熵驱动，缺失即报错。
            actor_cfg = omega_conf_to_dataclass(self.config.actor_rollout_ref.actor)
            actor_cfg.model_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
            simpletir = self.config.simpletir
            self.actor_rollout_wg.set_loss_fn(
                partial(
                    opsa_loss,
                    config=actor_cfg,
                    lowest_frac=float(simpletir.get("opsa_lowest_frac", 0.2)),
                    adv_fix=float(simpletir.get("opsa_adv_fix", -0.75)),
                    delta=float(simpletir.get("opsa_delta", 1.0)),
                )
            )

    def _get_extra_fields(self, batch) -> list[dict[str, Any]]:
        """按行键取回 agent loop 存进 TransferQueue 的非张量元数据。

        里面是每个 turn 的 turn_step / code_present / sandbox_ok /
        teacher_response_log_probs / reward_extra_info 等字段——张量字段走
        kv_batch_get 的 select_fields，而这些 python 对象统一挂在 extra_fields。
        """
        values = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])[
            "extra_fields"
        ]
        entries = _as_list(values)
        if len(entries) != len(batch.keys):
            raise RuntimeError("TransferQueue extra_fields count does not match batch keys")
        # 中间件可能把 dict 序列化成别的容器；无法识别就当空 dict，让后面的
        # 必填字段校验去报具体缺什么。
        return [entry if isinstance(entry, dict) else {} for entry in entries]

    def _balance_batch(self, batch, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Accept only Verl's zero-loss synthetic padding, never copied turns.

        背景（历史 P0 事故）：Verl V1 的 _balance_batch 为满足 actor 的整除约束
        可能向 batch 里补行。早期版本曾把真实 turn 复制一份充当 padding——同一条
        生成被计两次损失，梯度直接错。此守卫强制两件事：

        1. 真实行键在平衡前后必须是同一集合（只多不少、不重写、不复制）；
        2. 新增的 padding 行 response_mask 与 rm_scores 必须全零（零损失）。
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
                # 常态下 real_turns_preserved == 每步真实生成 turn 数，出现偏差
                # 说明上游行为变化，守卫会在抛错前先在这里留下现场。
                "simpletir/real_turns_preserved": float(len(real_keys_after)),
                "simpletir/synthetic_padding_turns": float(len(padding_keys)),
                "simpletir/p0_no_real_turn_duplication": 1.0,
            }
        )
        return balanced

    def _write_teacher_response_logprobs(self, batch, metrics: dict):
        """校验并写回与 student token 逐位对齐的 teacher log-prob。

        OPSD 和 AgentOPSD 的 teacher 信号来自 agent loop 里的私有前向（带金答案
        的 prompt 过一遍同步后的 rollout 策略），结果存于 extra_fields。这里做
        四件事：

        1. 按 response_mask 对齐：teacher 值个数必须等于该行真实 token 数；
        2. 有限性校验：NaN/Inf 一旦进入 credit 会污染整批优势；
        3. 轨迹完整性：同一 session 的 turn 编号必须连续（0..n-1），防止丢轮
           之后 credit 记错位置；
        4. 终局标签一致：同一 episode 各轮的 answer_accuracy 必须相同（奖励只
           在最后一轮结算，广播到每一轮）。

        校验通过后把 teacher 张量以 nested 形式写回 TransferQueue——actor worker
        只认 nested 布局（这也是一次真实修过的 bug：曾把稠密 mask 传给要求
        nested 的转换函数导致训练崩溃）。
        """
        tensor_data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask", "old_log_probs", "advantages", "returns"],
        )
        # nested 原始结构留一份：写回新字段时必须复用它，不能用 padded 版本。
        response_mask_nested = tensor_data["response_mask"]
        dense = tensor_data.to_padded_tensor()
        response_mask = dense["response_mask"].bool()
        old_log_probs = dense["old_log_probs"]
        extras = self._get_extra_fields(batch)
        # padding 行不参与 teacher/credit 的任何计算或校验。
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
            # 行键里的 turn 与 agent loop 写入的 turn_step 必须互相印证；
            # 不一致说明键被重排或元数据丢失，直接终止而不是猜对应关系。
            if int(entry.get("turn_step", -1)) != turn:
                raise RuntimeError(
                    f"SimpleTIR turn metadata/key mismatch for {row_key!r}: "
                    f"metadata={entry.get('turn_step')!r}, key={turn}"
                )
            raw_teacher = entry.get("teacher_response_log_probs")
            if raw_teacher is None:
                # 这是"拒绝静默降级"的核心检查：teacher 缺失时绝不能装作
                # GRPO 继续训练，否则该臂的方法语义已经变了。
                raise RuntimeError(f"{self.method} requires teacher_response_log_probs; refusing a silent GRPO fallback")
            teacher_values = _as_list(raw_teacher)
            real_len = int(response_mask[row].sum().item())
            if real_len <= 0 or len(teacher_values) != real_len:
                raise RuntimeError(
                    f"teacher/token mask length mismatch for {row_key!r}: teacher={len(teacher_values)}, mask={real_len}"
                )
            # Verl 的 padded 布局里真实 token 总在行首前缀，因此写到 [:real_len]。
            teacher[row, :real_len] = torch.as_tensor(teacher_values, dtype=teacher.dtype, device=teacher.device)
            if not torch.isfinite(teacher[row, :real_len]).all():
                raise RuntimeError(f"non-finite teacher log-probability for {row_key!r}")
            reward_info = entry.get("reward_extra_info")
            if not isinstance(reward_info, dict) or "answer_accuracy" not in reward_info:
                raise RuntimeError(f"SimpleTIR missing binary answer_accuracy for {row_key!r}")
            answer_accuracy = float(reward_info["answer_accuracy"])
            # B0（以及 OPSD 的门控）都要求二值成功标签；0.5 半分只影响奖励
            # 标量，不允许出现在这里。
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

        # teacher-student gap 是论文里的核心信号：teacher 看过答案，其 log-prob
        # 与 student 之差衡量"知道答案后这一步会怎么做"。gap_mean 应接近 0，
        # gap_rms 显著大于 0；若 gap 消失说明 teacher 前向退化成了 student 复读。
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
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {"teacher_response_log_probs": response_to_nested(teacher, response_mask_nested)},
                batch_size=len(batch),
            ),
        )
        return batch, dense, extras, is_padding

    def _reshape_agentopsd(self, batch, dense: TensorDict, extras: list[dict[str, Any]], is_padding: np.ndarray, metrics: dict):
        """AgentOPSD 臂专属：把轨迹级 GRPO 优势重塑为回合级优势。

        输入布局（由 _write_teacher_response_logprobs 准备）：
        - 每行 = 一个 (trajectory, turn)，行内是 token 级张量；
        - GRPO 优势本来整条轨迹恒定（广播到每轮每个 token）。

        reshape_advantages（agentopsd/credit.py，论文 Algorithm 1）用
        teacher-student token gap 计算每轮信用，再以有界乘子放大/缩小各轮优势。
        这里负责喂对元数据并把结果写回队列。"""
        active = ~is_padding
        active_tensor = torch.as_tensor(active, dtype=torch.bool, device=dense["advantages"].device)
        # 重新以 nested 形式取 teacher（上一步刚写入的），保证写回 advantages
        # 时使用同一份 response_mask 结构。
        teacher_data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["teacher_response_log_probs", "response_mask"]
        )
        teacher_values = teacher_data.to_padded_tensor()["teacher_response_log_probs"]
        uid, traj_uid, turn_step, successes = [], [], [], []
        session_advantages: dict[str, list[float]] = {}
        for row, row_key in enumerate(batch.keys):
            if not active[row]:
                continue
            group, session, turn = _session_key(row_key)
            mask = dense["response_mask"][row].bool()
            # 取该行第一个真实 token 的优势：GRPO 的优势是轨迹级常数，
            # 任何位置都一样；取第一个是为了避开 padded 尾部。
            first = int(mask.float().argmax().item())
            row_advantage = float(dense["advantages"][row, first].item())
            session_advantages.setdefault(f"{group}:{session}", []).append(row_advantage)
            reward_info = extras[row]["reward_extra_info"]
            uid.append(group)
            traj_uid.append(f"{group}:{session}")
            turn_step.append(turn)
            # B0 定义在二值任务成功上，不用 0.5 的无工具半分——半分会把
            # "答对但没认真用工具"和"真失败"混进同一档，破坏信用分配。
            successes.append(float(reward_info["answer_accuracy"]))
        # 同一轨迹各轮的优势必须相同（上游 GRPO 按轨迹广播）。若不同说明
        # 优势计算被别的钩子改过，此时重塑的前提已不成立。
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
        # 只有真实行被覆盖；padding 行保持零，维持"零损失 padding"不变量。
        dense["advantages"][active_tensor] = reshaped
        dense["returns"][active_tensor] = reshaped
        diag.update(
            {
                "agentopsd/reshape_applied": 1.0,
                "agentopsd/trajectory_metadata_validated": 1.0,
                "agentopsd/padding_turn_count": float(is_padding.sum()),
                "agentopsd/active_turn_count": float(active.sum()),
                # 重复报一份带 agentopsd/ 前缀的 gap，方便单臂曲线直接对齐。
                "agentopsd/teacher_student_gap_mean": metrics["simpletir/teacher_student_gap_mean"],
                "agentopsd/teacher_student_gap_rms": metrics["simpletir/teacher_student_gap_rms"],
            }
        )
        batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(
                {
                    "advantages": response_to_nested(dense["advantages"], teacher_data["response_mask"]),
                    "returns": response_to_nested(dense["returns"], teacher_data["response_mask"]),
                },
                batch_size=len(batch),
            ),
        )
        metrics.update(diag)
        self._last_credit_diag = diag
        return batch

    def _compute_advantage(self, batch, metrics):
        """优势计算的总分岔口，三臂在此分流。

        顺序很重要：先跑 Verl 原生 GRPO（组内标准化），AgentOPSD 是在它的
        结果之上做重塑，而不是另起炉灶——这样消融差异被严格限制在
        "是否重塑 credit"这一点上。
        """
        batch = super()._compute_advantage(batch, metrics)
        # 每步先复位诊断标志；GRPO 臂保持 0，监控里就能直接看出该臂没走重塑。
        self._last_credit_diag = {"agentopsd/reshape_applied": 0.0}
        # OPSA 的负优势在损失内部构造（opsa_loss.py），与 GRPO 优势完全无关；
        # 这里照 grpo 路径原样返回，既不算 teacher 也不改写优势。
        if self.method in ("grpo", "opsa"):
            return batch
        # OPSD 与 AgentOPSD 都需要 teacher 前向：OPSD 用它驱动蒸馏损失，
        # AgentOPSD 用它做信用分配，因此校验/写回逻辑共用。
        batch, dense, extras, is_padding = self._write_teacher_response_logprobs(batch, metrics)
        if self.method == "agentopsd":
            batch = self._reshape_agentopsd(batch, dense, extras, is_padding, metrics)
        return batch

    def _rollout_monitor_metrics(self, batch) -> dict[str, float]:
        """聚合 agent loop 记录的逐 turn 诊断，用于训练中快速定位故障。

        这四个比率是预跑排障时定位三层零奖励 bug 的直接线索：
        - code_present_ratio 低  → 模型很少产出可执行代码（提示词/截断问题）；
        - sandbox_ok_ratio 低    → 代码有了但沙箱执行失败（曾定位到缺科学库）；
        - void_turn_ratio 高     → 大量"无代码无答案"轮，典型原因是截断级联；
        - answer_accuracy        → 终局正确率，奖励信号是否存在的最终判据。
        """
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
            # 沙箱成功率只在"有代码"的轮上统计——把没写代码的轮算进分母
            # 会让沙箱故障被稀释到看不见。
            "simpletir/sandbox_ok_ratio": float(sandbox_ok[code > 0].mean()) if np.any(code > 0) else 0.0,
            "simpletir/void_turn_ratio": float(void.mean()),
            "simpletir/answer_accuracy": float(np.mean(answers)) if answers else 0.0,
        }

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        """在 Verl 原生指标之上追加 SimpleTIR 诊断并按步落盘 JSONL。

        JSONL 只收标量（simpletir/ opsd/ val 前缀），不含任何文本字段——
        奖励文本、学生代码、金答案都不允许进入指标文件（信息泄漏边界）。
        """
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        metrics.update(self._rollout_monitor_metrics(batch))
        if self._monitor is None or global_steps % self.monitor_every:
            return
        record = self._monitor.record(self._last_credit_diag, int(global_steps), metrics)
        # method 与 schema 版本写进每条记录：多臂 JSONL 合并分析时靠它们区分
        # 来源、检测旧格式。
        record["simpletir/method"] = self.method
        record["simpletir/monitor_schema"] = 1
        for key, value in metrics.items():
            if key.startswith("simpletir/") or key.startswith("opsd/") or key.startswith("val"):
                if isinstance(value, (int, float, np.number)):
                    record[key] = float(value)
        append_jsonl(self.monitor_path, record)
