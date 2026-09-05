"""TransferQueue worker that explicitly keeps teachers out of validation.

Upstream Verl's V1 worker removes ``validate`` from each prompt before it calls
``AgentLoopBase.run``.  That is fine for the built-in loops, but a
gold-conditioned training-only teacher needs this bit of provenance.  This
small adapter copies the V1 TransferQueue publication contract and injects one
boolean into our custom loop.  It never changes the row count or fabricates a
turn to make batches divisible.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import hydra
import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import AgentLoopManager, AgentLoopOutput, AgentLoopWorker, get_trajectory_info
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap, _agent_loop_registry
from integrations.simpletir_qwen35.simpletir_agent_loop import SimpleTIRPythonAgentLoop
from verl.trainer.ppo.v1.agent_loop_tq import apply_greedy_sampling_params, _settle_session_tasks
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_attr
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def text_only_position_ids(
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, Any] | None,
) -> torch.Tensor:
    """Return standard text positions and reject accidental media payloads."""
    if multi_modal_inputs is not None and (not isinstance(multi_modal_inputs, dict) or multi_modal_inputs):
        raise RuntimeError("SimpleTIR is text-only and rejects multimodal position inputs")
    return compute_position_id_with_mask(attention_mask)


def text_only_multi_modal_inputs(multi_modal_data: dict[str, Any] | None) -> dict[str, torch.Tensor]:
    """Reject real media while omitting processor-only auxiliary fields."""
    if multi_modal_data:
        raise RuntimeError("SimpleTIR received unexpected multimodal trajectory data")
    return {}


@ray.remote
class SimpleTIRAgentLoopWorkerTQ(AgentLoopWorker):
    """V1 worker with exact TQ fields plus an explicit validation argument."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks: set[asyncio.Task[Any]] = set()

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Publish one-dimensional text positions for the text-only benchmark.

        This worker, rather than ``AgentLoopBase.run``, constructs the rows
        consumed by actor old-logprob.  Qwen3.5's generic processor advertises
        vision RoPE and would otherwise create a four-channel position tensor
        even though SimpleTIR carries no image/video tokens.
        """
        return text_only_position_ids(attention_mask, multi_modal_inputs)

    def _compute_multi_modal_inputs(self, output: AgentLoopOutput, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keep SimpleTIR rows strictly text-only.

        Qwen3.5's processor can emit RoPE-related auxiliary fields even when
        the input contains no images, videos, or audio. Those fields are not
        part of a SimpleTIR trajectory and must not select the model's
        multimodal position-id path.
        """
        return text_only_multi_modal_inputs(output.multi_modal_data)

    async def generate_sequences(self, batch: TensorDict) -> None:
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if validate:
            sampling_params.update(
                {
                    "top_p": config.val_kwargs.top_p,
                    "top_k": config.val_kwargs.top_k,
                    "temperature": config.val_kwargs.temperature,
                }
            )
        if "agent_name" not in batch:
            batch["agent_name"] = NonTensorData(config.agent.default_agent_loop)

        trajectories = await get_trajectory_info(batch["global_steps"], batch["index"], validate)
        for index in range(len(batch)):
            prompt: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[index]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[index].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                else:
                    raise TypeError(f"SimpleTIR unsupported TQ prompt field {key!r}: {type(value)!r}")
            task = asyncio.create_task(self._run_prompt(prompt, sampling_params, trajectory=trajectories[index]))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict[str, Any], sampling_params: dict[str, Any], trajectory: dict[str, Any]) -> None:
        uid, partition_id = prompt["uid"], "val" if trajectory["validate"] else "train"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks: list[asyncio.Task[Any]] = []
        try:
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.val_kwargs.n if trajectory["validate"] else config.n)
            do_sample = prompt.pop("__do_sample__", True)
            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)
            for session_id in range(n):
                tasks.append(
                    asyncio.create_task(
                        self._run_agent_loop(run_sampling_params, trajectory=trajectory, session_id=session_id, **prompt)
                    )
                )
            errors = await _settle_session_tasks(tasks)
            if errors:
                for error in errors:
                    logger.error("SimpleTIR session failed for uid=%s", uid, exc_info=(type(error), error, error.__traceback__))
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception:
            logger.exception("SimpleTIR prompt failed for uid=%s", uid)
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        session_id: int,
        **kwargs,
    ) -> None:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=False,
        ):
            if agent_name not in _agent_loop_registry:
                raise RuntimeError(f"unknown SimpleTIR agent loop {agent_name!r}")
            loop_cfg = _agent_loop_registry[agent_name]
            loop = hydra.utils.instantiate(
                config=loop_cfg,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                hf_model_type=self.hf_model_type,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output = await loop.run(sampling_params, is_validation=bool(trajectory["validate"]), **kwargs)
            await self._agent_loop_postprocess(output, bool(trajectory["validate"]), session_id=session_id, **kwargs)

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate: bool, **kwargs
    ) -> None:
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            raise RuntimeError(f"SimpleTIR emitted no rows for {uid}_{session_id}")
        await self._compute_score(outputs, kwargs=kwargs)
        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        if final_output.reward_score is None:
            raise RuntimeError(f"SimpleTIR terminal reward missing for {uid}_{session_id}")
        reward_extra_info = final_output.extra_fields.get("reward_extra_info")
        if not isinstance(reward_extra_info, dict):
            raise RuntimeError(f"SimpleTIR terminal reward metadata missing for {uid}_{session_id}")
        for previous in outputs[:-1]:
            previous.reward_score = final_output.reward_score
            previous.extra_fields["reward_extra_info"] = reward_extra_info

        keys, fields, tags = [], [], []
        for turn_index, turn in enumerate(outputs):
            prompts = torch.tensor(turn.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(turn.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(turn, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)
            keys.append(f"{uid}_{session_id}_{turn_index}")
            field = turn.as_dict()
            field.update(kwargs)
            field.pop("multi_modal_data", None)
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompts.numel(),
                    "response_len": responses.numel(),
                    "seq_len": prompts.numel() + responses.numel(),
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )
        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="val" if validate else "train",
        )


class SimpleTIRAgentLoopManagerTQ(AgentLoopManager):
    """V1 manager selecting :class:`SimpleTIRAgentLoopWorkerTQ`."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = SimpleTIRAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)]
        )
