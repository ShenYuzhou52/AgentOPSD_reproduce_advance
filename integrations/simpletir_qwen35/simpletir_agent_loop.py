"""A P0-safe SimpleTIR Python loop for Verl's TransferQueue trainer.

本文件是 rollout 侧的核心：把每道数学题变成"多轮 围栏Python ⇄ 沙箱执行"的
agent 轨迹。一个 episode 的生命周期：

1. 在原始题目消息前拼接上游 SimpleTIR 的工具使用契约（prompting.py）；
2. 循环最多 max_turns 轮：生成助手动作 → 解析围栏代码（trajectory.py）→
   bwrap 沙箱执行（simpletir_sandbox.py）→ 观察回填为 user 消息；
3. 终止条件：文本或观察中出现 \\boxed{}、出现"无代码无答案"的空轮、或轮数耗尽；
4. episode 结束才读 ground_truth 计算奖励（全文提取 + math_verify）；
5. AgentOPSD/OPSD 臂在每轮额外做一次带金答案的私有 teacher 前向。

关键不变量（P0 安全性的来源）：每个 ``AgentLoopOutput`` 对应真实采样的一轮
assistant 动作，绝不复制旧轮凑 batch 形状；终局奖励只挂在最后一行，由
``AgentLoopWorkerTQ`` 广播到该 episode 的既有各行。金答案只进入 teacher
的私有请求与终局打分两处，永不进入学生可见文本、observation 或 extra_fields。
"""

from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.replica import TokenOutput

from integrations.simpletir_qwen35.simpletir_sandbox import run_python
from integrations.simpletir_qwen35.prompting import with_simpletir_prompt
from integrations.simpletir_qwen35.trajectory import (
    format_observation,
    is_only_final_answer,
    parse_turn,
    requires_sandbox,
    score_simpletir_math,
    with_final_answer_helper,
)


# 按 Ray worker 的事件循环缓存信号量，限制同时运行的本地沙箱数。
_sandbox_limiters: dict[tuple[int, int], asyncio.Semaphore] = {}


def _sandbox_limiter(limit: int) -> asyncio.Semaphore:
    """Return one bounded executor limiter per Ray worker event loop."""
    # 用当前事件循环身份隔离不同 worker 的并发配额。
    loop = asyncio.get_running_loop()
    key = (id(loop), limit)
    limiter = _sandbox_limiters.get(key)
    if limiter is None:
        limiter = asyncio.Semaphore(limit)
        _sandbox_limiters[key] = limiter
    return limiter


def _safe_rollout_extra_fields(token_output: TokenOutput) -> dict[str, Any]:
    """Keep scheduler provenance, never prompt/code/answer text.

    rollout 后端会在 extra_fields 里夹带调度元数据（min/max_global_steps，
    由 async rollout 的 staleness 控制使用）。这里只保留这两个白名单键：
    后端同字段还可能含有完整 prompt 文本，直接透传会把题目/代码泄进训练
    指标与日志。
    """
    source = getattr(token_output, "extra_fields", {}) or {}
    return {
        key: source[key]
        for key in ("min_global_steps", "max_global_steps")
        if key in source
    }


def _debug_dump_episode(record: dict[str, Any]) -> None:
    """Opt-in dump of student-visible trajectory data for failure analysis.

    Enabled only by setting ``SIMPLETIR_DEBUG_DUMP`` to a directory.  The record
    holds exclusively information the student model itself saw (its question,
    its own turn texts, bounded observations) plus scalar reward diagnostics;
    ``ground_truth`` is deliberately never serialised here.  Dump failures must
    not abort a training rollout, so every error is swallowed.
    """
    # 调试轨迹仅在显式设置目录时写盘，默认不产生样本内容。
    directory = os.environ.get("SIMPLETIR_DEBUG_DUMP", "")
    if not directory:
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"episode-{os.getpid()}-{uuid4().hex}.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _as_token_id(value: Any) -> int:
    """Unwrap Verl's singleton top-k axis without relaxing token alignment.

    Verl 的 vLLM 适配器把 prompt_logprobs 模式下的每个 token id 包成单元素
    列表 ``[id]``（top-k 轴）。这里只展开"恰好一个元素"的情况——多于一个
    说明返回结构变了，必须报错重查而不是猜测语义。这是预跑阶段真实修过的
    对齐 bug 的守卫（旧代码拿 [[id],...] 直接和 [id,...] 比较导致首批
    rollout 全部失败）。
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError("teacher prompt_ids must contain exactly one token per position")
        value = value[0]
    if value is None or isinstance(value, (tuple, list)):
        raise RuntimeError("teacher returned an invalid prompt token id")
    return int(value)


def _as_logprob(value: Any) -> float:
    """Normalize a scalar prompt-logprob returned by the rollout backend.

    同 _as_token_id：teacher log-prob 可能带张量包装或单元素列表包装，
    归一成 float；任何"多值"（top-k 分布）都视为配置错误直接终止——
    本项目没有实现 top-k 蒸馏，静默取第一个值会悄悄改变方法语义。
    """
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError("teacher prompt_logprobs must be scalar (top-k distillation is disabled)")
        value = value[0]
    if value is None:
        raise RuntimeError("teacher returned a missing prompt log-probability for a response token")
    return float(value)


@register("simpletir_python")
class SimpleTIRPythonAgentLoop(AgentLoopBase):
    """Fenced-Python multi-turn agent loop with a local sandbox."""

    def __init__(self, *args, **kwargs):
        """读取 simpletir.* 配置并做前置校验。

        - method：消融臂名，决定训练时是否附带 teacher 前向（grpo 不带）；
        - max_turns / sandbox_timeout / observation_limit：多轮预算与每轮
          观察上限（512 字符，控制上下文膨胀）；
        - reward_stdout_limit：打分用 stdout 上限，必须 ≥ 观察上限——奖励
          需要看到比学生可见窗口更多的输出（例如长枚举后的最终 print）；
        - sandbox_concurrency：每个 Ray worker 同时存在的 systemd/bwrap
          单元上限，防止沙箱风暴拖垮宿主。
        """
        super().__init__(*args, **kwargs)
        simpletir_cfg = self.config.get("simpletir", {})
        self.method = str(simpletir_cfg.get("method", "grpo"))
        self.max_turns = int(simpletir_cfg.get("max_turns", 5))
        self.sandbox_timeout = float(simpletir_cfg.get("sandbox_timeout_seconds", 5.0))
        self.observation_limit = int(simpletir_cfg.get("max_observation_chars", 512))
        self.reward_stdout_limit = int(simpletir_cfg.get("reward_stdout_chars", 16 * 1024))
        self.sandbox_concurrency = int(simpletir_cfg.get("sandbox_concurrency", 4))
        if self.max_turns < 1:
            raise ValueError("simpletir.max_turns must be positive")
        if self.reward_stdout_limit < self.observation_limit:
            raise ValueError("simpletir.reward_stdout_chars must be at least max_observation_chars")
        if self.sandbox_concurrency < 1:
            raise ValueError("simpletir.sandbox_concurrency must be positive")

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs=None,
    ):
        """Use text RoPE positions; SimpleTIR never permits multimodal prompts.

        Qwen3.5's processor exposes ``get_rope_index`` even for a text-only
        request.  The generic agent-loop implementation therefore constructs
        a four-channel vision MRoPE tensor, whereas Qwen3.5's language model
        expects ordinary one-dimensional positions without image/video tokens.
        That mismatch first appears during actor old-logprob computation.  A
        hard failure protects the contract if this text-only benchmark is ever
        accidentally supplied image/video data.
        """
        if multi_modal_inputs:
            raise RuntimeError("SimpleTIR is text-only and rejects multimodal position inputs")
        # 文本任务只计算一维 RoPE 位置，拒绝进入 Qwen 的多模态位置路径。
        return compute_position_id_with_mask(attention_mask)

    async def _same_policy_teacher_logprobs(
        self,
        *,
        student_messages: list[dict[str, Any]],
        response_ids: list[int],
        ground_truth: Any,
        priority: int,
    ) -> list[float]:
        """Score this exact action under the current rollout policy plus gold answer.

        This is the AgentOPSD teacher branch: it uses the synchronized policy
        rollout server, not a static external model.  The answer is present only
        in this request's private teacher prompt and is never copied to the
        returned fields.
        """
        # 教师提示中临时注入标准答案；该消息绝不会写入学生轨迹。
        teacher_system = {
            "role": "system",
            "content": (
                "Training-only privileged information: the correct final answer is "
                f"{ground_truth}.  Use it only to assess the likelihood of the next assistant action."
            ),
        }
        # 将私有教师消息和当前学生上下文编码成 rollout 服务器的输入 token。
        teacher_prompt_ids = await self.ct_build_initial_tokens([teacher_system, *deepcopy(student_messages)])
        sequence_ids = teacher_prompt_ids + response_ids
        # 同一同步策略对已采样动作做前向评分，生成无梯度的 teacher 对数概率。
        teacher = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params={
                "max_tokens": 1,
                "temperature": 1.0,
                "prompt_logprobs": 0,
                "detokenize": False,
            },
            priority=priority,
        )
        # 读取后端返回的 token 对齐信息，用于验证教师评分没有错位。
        raw_ids = [_as_token_id(value) for value in teacher.extra_fields.get("prompt_ids", [])]
        raw_logprobs = list(teacher.extra_fields.get("prompt_logprobs", []))
        if len(raw_ids) != len(sequence_ids) or len(raw_logprobs) != len(sequence_ids):
            raise RuntimeError(
                "teacher prompt-logprob alignment failed: "
                f"ids={len(raw_ids)} logprobs={len(raw_logprobs)} expected={len(sequence_ids)}"
            )
        # Verl's vLLM adapter reports the likelihood of token i in slot i-1,
        # followed by a dummy final slot.  Slice the generated student action
        # by its first target position rather than from the tail, which would
        # include that dummy and silently shift every score by one token.
        # vLLM 的 prompt-logprob 下标相对目标 token 左移一位，因此从这里切片。
        first_target = len(teacher_prompt_ids) - 1
        last_target = first_target + len(response_ids)
        if first_target < 0 or raw_ids[first_target:last_target] != response_ids:
            raise RuntimeError("teacher response token ids do not match the sampled student action")
        return [_as_logprob(value) for value in raw_logprobs[first_target:last_target]]

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        is_validation: bool = False,
        **kwargs,
    ) -> list[AgentLoopOutput]:
        """Run up to five student turns; the gold answer is not read until scoring.

        is_validation 由 agent_loop_manager 注入（上游 V1 worker 会把它从
        prompt 里删掉）：验证 rollout 绝不做 teacher 前向，保证评测不接触
        金答案之外的任何特权信息。
        """
        priority = int(priority)
        raw_prompt = kwargs.get("raw_prompt")
        if not isinstance(raw_prompt, list):
            raise TypeError("SimpleTIR requires data.raw_prompt to be a chat-message list")
        # Match SimpleTIR's RLCustomPromptDataset, which prepends the tool-use
        # contract before chat templating.  The generic Verl dataset used by
        # this reproduction does not have that ``data.prompt`` hook.
        # 在原始聊天题目之前加入工具使用协议，再交给 tokenizer 套模板。
        messages = with_simpletir_prompt(raw_prompt)
        self._assert_mm_supported(False)
        # 初始化第一轮 rollout 的完整提示 token。
        runtime_prompt_ids = await self.ct_build_initial_tokens(messages)

        outputs: list[AgentLoopOutput] = []
        # SimpleTIR's math reward follows upstream hf_math_verify: the answer
        # is extracted from the full episode text (every assistant turn plus
        # its bounded observation, in order), so a correct final answer stated
        # after a successful tool run scores exactly like final_answer() output.
        # The earlier stdout-only reading accidentally applied the LeetCode
        # reward path to math and zeroed most correct episodes.
        # 累计每轮助手文本和可见 observation，用于末尾按 SimpleTIR 规则评分。
        episode_text: list[str] = []
        substantive_tool_use = False
        terminal = False
        debug_turns: list[dict[str, Any]] = []
        # 每个样本最多执行配置指定的工具交互轮数。
        for turn_step in range(self.max_turns):
            metrics: dict[str, Any] = {}
            # 确定性调试时使用可复现请求编号；普通训练使用随机 UUID。
            request_id = (
                f"simpletir-{priority}-{turn_step}"
                if getattr(self.rollout_config, "full_determinism", False)
                else uuid4().hex
            )
            # 记录 rollout 生成耗时，供训练日志汇总。
            with simple_timer("generate_sequences", metrics):
                # 向当前策略的 rollout 服务请求本轮助手续写。
                generated: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_prompt_ids,
                    sampling_params=sampling_params,
                    priority=priority,
                )
            metrics["num_preempted"] = generated.num_preempted if generated.num_preempted is not None else -1

            # 把新生成动作拼入 token 序列，并保留响应区间的 mask 与 log-prob。
            merged, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                runtime_prompt_ids,
                generated.token_ids,
                [],
                [] if generated.log_probs else None,
                assistant_logprobs=generated.log_probs if generated.log_probs else None,
            )
            if not response_mask:
                raise RuntimeError("SimpleTIR rollout produced an empty assistant turn")
            # merged.token_ids = prompt_ids + response_ids；按 mask 长度切分出
            # 本轮的 prompt 部分与响应部分，作为一行训练数据的两个区间。
            response_ids = merged.token_ids[-len(response_mask) :]
            prompt_ids = merged.token_ids[: len(merged.token_ids) - len(response_mask)]
            text = self.tokenizer.decode(generated.token_ids, skip_special_tokens=True)
            # 解析 fenced Python 和 boxed answer，决定是否调用沙箱或结束轨迹。
            parsed = parse_turn(text)

            extra_fields = _safe_rollout_extra_fields(generated)
            extra_fields.update(
                {
                    "turn_step": int(turn_step),
                    "code_present": float(parsed.code is not None),
                    "is_void_turn": float(parsed.is_void),
                    "sandbox_ok": 0.0,
                    "sandbox_timeout": 0.0,
                    "observation_chars": 0.0,
                }
            )
            # 为实际生成的这一轮建立一条训练行，不复制或填充旧轮次。
            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                routed_experts=(
                    generated.routed_experts[: len(prompt_ids) + len(response_ids)]
                    if generated.routed_experts is not None
                    else None
                ),
                num_turns=turn_step + 2,
                metrics=metrics,
                extra_fields=extra_fields,
            )
            # num_turns 从 2 起算（Verl 把初始提示构建也计一轮；日志里
            # training/num_turns/min=2 与此一致），仅供指标统计，不参与训练。
            # 仅在非 GRPO 的训练 rollout 中计算私有 teacher，验证阶段不读取答案。
            if self.method != "grpo" and not is_validation:
                reward_model = kwargs.get("reward_model")
                if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
                    raise RuntimeError("SimpleTIR requires reward_model.ground_truth for its private teacher forward")
                # This is one detached teacher forward per actual turn.  Only
                # numeric log-probs leave this scope; no teacher tokens or gold
                # text enter output fields.
                output.extra_fields["teacher_response_log_probs"] = await self._same_policy_teacher_logprobs(
                    student_messages=messages,
                    response_ids=response_ids,
                    ground_truth=reward_model["ground_truth"],
                    priority=priority,
                )
            # 保存本轮输出，末轮奖励将在终止后回填给此前各轮。
            outputs.append(output)
            episode_text.append(text)
            debug_turn = {
                "turn_step": int(turn_step),
                "text": text,
                "code_present": parsed.code is not None,
                "is_void": parsed.is_void,
            }
            debug_turns.append(debug_turn)
            runtime_prompt_ids = merged.token_ids

            # A direct boxed answer without code ends immediately.  If code is
            # also present, upstream SimpleTIR executes it first and only then
            # closes the episode; skipping it would change the reward.
            # 没有待执行代码时，本轮已给出答案或空输出，轨迹在此终止。
            if not requires_sandbox(parsed):
                # A response containing neither a code action nor an answer is a
                # terminal void turn.  It is an actual sampled row, not padding.
                terminal = True
                break

            # ``subprocess.run`` is blocking.  Offload it while bounding the
            # number of local systemd/bwrap units within each Ray worker.
            # 将阻塞的 systemd/bwrap 执行移出事件循环，并受并发阈值限制。
            async with _sandbox_limiter(self.sandbox_concurrency):
                sandbox = await asyncio.to_thread(
                    run_python,
                    with_final_answer_helper(parsed.code),
                    timeout_seconds=self.sandbox_timeout,
                    output_limit=self.reward_stdout_limit,
                )
            # 将 stdout、stderr 和超时状态压缩成模型下一轮可见的 observation。
            observation = format_observation(
                stdout=sandbox.stdout,
                stderr=sandbox.stderr,
                timed_out=sandbox.timed_out,
                limit=self.observation_limit,
            )
            output.extra_fields.update(
                {
                    "sandbox_ok": float(sandbox.ok),
                    "sandbox_timeout": float(sandbox.timed_out),
                    "observation_chars": float(len(observation)),
                }
            )
            debug_turn.update(
                {
                    "sandbox_ok": bool(sandbox.ok),
                    "sandbox_timed_out": bool(sandbox.timed_out),
                    "sandbox_returncode": sandbox.returncode,
                    "observation": observation,
                }
            )
            episode_text.append(observation)
            # 仅把真正执行计算的代码计为工具使用；纯 final_answer() 不计入。
            substantive_tool_use = substantive_tool_use or (
                sandbox.ok and not is_only_final_answer(parsed.code)
            )
            if parsed.has_boxed_answer or parse_turn(observation).has_boxed_answer:
                terminal = True
                break

            # The assistant text is already represented by ``runtime_prompt_ids``.
            # Only merge the newly appended non-assistant observation next.
            # 把助手文本追加到对话历史，再加入沙箱 observation 供下一轮使用。
            messages.append({"role": "assistant", "content": text})
            previous_messages = list(messages)
            messages.append({"role": "user", "content": observation})
            merged_observation, _, _ = await self.ct_merge_non_assistant_msg(
                previous_messages,
                messages,
                runtime_prompt_ids,
                [],
                None,
            )
            runtime_prompt_ids = merged_observation.token_ids

        if not outputs:
            raise RuntimeError("SimpleTIR generated no turn rows")
        if not terminal and len(outputs) != self.max_turns:
            raise RuntimeError("SimpleTIR ended without a terminal condition")

        # Gold never enters student-visible messages, observations, output
        # fields, or logs.  AgentOPSD may use it privately in the teacher
        # forward above; terminal scoring is the only other permitted read.
        reward_model = kwargs.get("reward_model")
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise RuntimeError("SimpleTIR requires reward_model.ground_truth for terminal scoring")
        # 终止后用完整 episode 文本与标准答案计算任务奖励。
        reward = score_simpletir_math(
            "\n".join(episode_text),
            reward_model["ground_truth"],
            substantive_tool_use=substantive_tool_use,
        )
        final_output = outputs[-1]
        # 只在末轮存主奖励；worker 随后将其广播到该 episode 的前序行。
        # reward.pop 把 score 从诊断 dict 中移除，避免同一数值以两种
        # 形态进入 extra_fields。
        final_output.reward_score = reward.pop("score")
        # reward_extra_info（answer_accuracy/is_boxed_ratio/substantive_tool_use）
        # 会一路传到 trainer：answer_accuracy 是 AgentOPSD B0 与一致性校验的
        # 数据源，也是 rollout 监控指标的输入。
        final_output.extra_fields["reward_extra_info"] = reward
        _debug_dump_episode(
            {
                "is_validation": bool(is_validation),
                "question": next(
                    (str(m.get("content")) for m in reversed(raw_prompt) if isinstance(m, dict) and m.get("role") == "user"),
                    "",
                ),
                "turns": debug_turns,
                "reward_extra_info": reward,
            }
        )
        return outputs
