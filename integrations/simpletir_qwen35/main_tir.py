"""Launch the isolated Qwen3.5/SimpleTIR overlay on pinned modern Verl.

This entrypoint deliberately composes Verl's installed ``ppo_trainer`` base
configuration without copying it.  The launch script supplies all experiment
overrides, while importing ``trainer`` registers the strict three-way trainer
before Ray workers are created.
"""

from __future__ import annotations

import sys
import os
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
import ray

# Registration has to happen in the controller process before TaskRunnerV1
# resolves trainer.v1.trainer_mode inside the remote Ray task.
from integrations.simpletir_qwen35 import trainer as _simpletir_trainer  # noqa: F401
from integrations.simpletir_qwen35.data_contract import validate_simpletir_files
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class SimpleTIRTaskRunner:
    """V1 runner that registers the overlay inside the remote Ray process."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()
        import transfer_queue as tq

        # Ray workers are separate interpreters: importing here registers the
        # custom trainer before the standard registry is queried.
        from integrations.simpletir_qwen35.trainer import SimpleTIRTrainer

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = SimpleTIRTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def compose_config(overrides: list[str]):
    import verl

    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="simpletir"):
        return compose(config_name="ppo_trainer", overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    config = compose_config(list(sys.argv[1:] if argv is None else argv))
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.use_v1:
        raise ValueError("SimpleTIR overlay requires modern Verl V1 / TransferQueue")
    if str(config.trainer.v1.trainer_mode) != "sync":
        raise ValueError("SimpleTIR overlay currently supports only trainer.v1.trainer_mode=sync")
    if not config.transfer_queue.enable:
        raise ValueError("SimpleTIR overlay requires transfer_queue.enable=true")
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1 or int(rollout.data_parallel_size) != 1:
        raise ValueError("SimpleTIR 8-GPU recipe requires rollout tensor/data parallel size = 1/1")
    if int(config.actor_rollout_ref.actor.fsdp_config.fsdp_size) != int(config.trainer.n_gpus_per_node):
        raise ValueError("SimpleTIR requires actor FSDP size to equal trainer.n_gpus_per_node")
    print(
        "simpletir_parallelism="
        f"actor_fsdp={config.actor_rollout_ref.actor.fsdp_config.fsdp_size} "
        f"rollout_tp={rollout.tensor_model_parallel_size} rollout_dp={rollout.data_parallel_size} "
        f"trainer_gpus={config.trainer.n_gpus_per_node}"
    )
    # Validate the leakage boundary before starting Ray or allocating a GPU.
    # This returns schema-only metadata and intentionally is not sent to the
    # tracker because a tracker may preserve arbitrary objects indefinitely.
    train_files = config.data.train_files
    val_files = config.data.val_files
    train_files = list(train_files) if isinstance(train_files, Sequence) and not isinstance(train_files, str) else [train_files]
    val_files = list(val_files) if isinstance(val_files, Sequence) and not isinstance(val_files, str) else [val_files]
    validate_simpletir_files([*train_files, *val_files])
    if os.environ.get("SIMPLE_TIR_PREFLIGHT_ONLY") == "1":
        print("simpletir_preflight=ok")
        return
    run_ppo(config, task_runner_class=SimpleTIRTaskRunner)


if __name__ == "__main__":
    main()
