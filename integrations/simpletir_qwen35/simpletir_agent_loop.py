"""A P0-safe SimpleTIR Python loop for Verl's TransferQueue trainer.

Each returned ``AgentLoopOutput`` is one generated assistant turn.  The final
outcome is attached only after the trajectory terminates and is then broadcast
by ``AgentLoopWorkerTQ`` to those already-existing turn rows.  No row is made
by copying a prior turn to satisfy a batch shape.
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


_sandbox_limiters: dict[tuple[int, int], asyncio.Semaphore] = {}


def _sandbox_limiter(limit: int) -> asyncio.Semaphore:
    """Return one bounded executor limiter per Ray worker event loop."""
    loop = asyncio.get_running_loop()
    key = (id(loop), limit)
    limiter = _sandbox_limiters.get(key)
    if limiter is None:
        limiter = asyncio.Semaphore(limit)
        _sandbox_limiters[key] = limiter
    return limiter


def _safe_rollout_extra_fields(token_output: TokenOutput) -> dict[str, Any]:
    """Keep scheduler provenance, never prompt/code/answer text."""
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
    """Unwrap Verl's singleton top-k axis without relaxing token alignment."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError("teacher prompt_ids must contain exactly one token per position")
        value = value[0]
    if value is None or isinstance(value, (tuple, list)):
        raise RuntimeError("teacher returned an invalid prompt token id")
    return int(value)


def _as_logprob(value: Any) -> float:
    """Normalize a scalar prompt-logprob returned by the rollout backend."""
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
        teacher_system = {
            "role": "system",
            "content": (
                "Training-only privileged information: the correct final answer is "
                f"{ground_truth}.  Use it only to assess the likelihood of the next assistant action."
            ),
        }
        teacher_prompt_ids = await self.ct_build_initial_tokens([teacher_system, *deepcopy(student_messages)])
        sequence_ids = teacher_prompt_ids + response_ids
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
        """Run up to five student turns; the gold answer is not read until scoring."""
        priority = int(priority)
        raw_prompt = kwargs.get("raw_prompt")
        if not isinstance(raw_prompt, list):
            raise TypeError("SimpleTIR requires data.raw_prompt to be a chat-message list")
        # Match SimpleTIR's RLCustomPromptDataset, which prepends the tool-use
        # contract before chat templating.  The generic Verl dataset used by
        # this reproduction does not have that ``data.prompt`` hook.
        messages = with_simpletir_prompt(raw_prompt)
        self._assert_mm_supported(False)
        runtime_prompt_ids = await self.ct_build_initial_tokens(messages)

        outputs: list[AgentLoopOutput] = []
        # SimpleTIR's math reward is computed from actual code stdout, not a
        # \boxed{} string merely written inside a model's code fence.  Keeping
        # that distinction prevents an execution failure from being rewarded.
        execution_stdout: list[str] = []
        substantive_tool_use = False
        terminal = False
        debug_turns: list[dict[str, Any]] = []
        for turn_step in range(self.max_turns):
            metrics: dict[str, Any] = {}
            request_id = (
                f"simpletir-{priority}-{turn_step}"
                if getattr(self.rollout_config, "full_determinism", False)
                else uuid4().hex
            )
            with simple_timer("generate_sequences", metrics):
                generated: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_prompt_ids,
                    sampling_params=sampling_params,
                    priority=priority,
                )
            metrics["num_preempted"] = generated.num_preempted if generated.num_preempted is not None else -1

            merged, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                runtime_prompt_ids,
                generated.token_ids,
                [],
                [] if generated.log_probs else None,
                assistant_logprobs=generated.log_probs if generated.log_probs else None,
            )
            if not response_mask:
                raise RuntimeError("SimpleTIR rollout produced an empty assistant turn")
            response_ids = merged.token_ids[-len(response_mask) :]
            prompt_ids = merged.token_ids[: len(merged.token_ids) - len(response_mask)]
            text = self.tokenizer.decode(generated.token_ids, skip_special_tokens=True)
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
            outputs.append(output)
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
            if not requires_sandbox(parsed):
                # A response containing neither a code action nor an answer is a
                # terminal void turn.  It is an actual sampled row, not padding.
                terminal = True
                break

            # ``subprocess.run`` is blocking.  Offload it while bounding the
            # number of local systemd/bwrap units within each Ray worker.
            async with _sandbox_limiter(self.sandbox_concurrency):
                sandbox = await asyncio.to_thread(
                    run_python,
                    with_final_answer_helper(parsed.code),
                    timeout_seconds=self.sandbox_timeout,
                    output_limit=self.reward_stdout_limit,
                )
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
            if sandbox.stdout:
                execution_stdout.append(sandbox.stdout)
            substantive_tool_use = substantive_tool_use or (
                sandbox.ok and not is_only_final_answer(parsed.code)
            )
            if parsed.has_boxed_answer or parse_turn(observation).has_boxed_answer:
                terminal = True
                break

            # The assistant text is already represented by ``runtime_prompt_ids``.
            # Only merge the newly appended non-assistant observation next.
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
        reward = score_simpletir_math(
            "\n".join(execution_stdout),
            reward_model["ground_truth"],
            substantive_tool_use=substantive_tool_use,
        )
        final_output = outputs[-1]
        final_output.reward_score = reward.pop("score")
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
