"""Launch the isolated Qwen3.5/SimpleTIR overlay on pinned modern Verl.

整个训练的进程入口（`python -m integrations.simpletir_qwen35.main_tir`），
由 scripts/run_simpletir_qwen35_4b.sh 调用。启动顺序刻意安排为：

1. 用 Hydra 组装"已安装 Verl 的 ppo_trainer 基配置 + 命令行覆盖项"——
   不复制配置文件，避免与基座版本漂移；
2. 纯 CPU 阶段完成全部可前置的校验（V1/同步模式/TQ/并行度约束、数据
   契约与防泄漏检查），失败时不占任何 GPU；
3. SIMPLE_TIR_PREFLIGHT_ONLY=1 时到第 2 步即返回（预检模式）；
4. run_ppo 把控制权交给 SimpleTIRTaskRunner（Ray 远程类）：在 worker
   进程里注册 SimpleTIRTrainer、初始化 TransferQueue 与 agent loop
   manager，然后进入训练主循环。

trainer 的 import 必须发生在控制器进程里（见下方注释）：注册表要在
TaskRunnerV1 解析 trainer_mode 之前就位。
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
# 在控制进程先导入自定义 Trainer，使其注册到 Verl 的训练器表。
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

        # 优先读取配置指定的 agent manager；未指定时使用 Verl 默认实现。
        manager_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
        # 将训练器提供的 rollout、teacher 与奖励句柄交给循环管理器。
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

        # 此集成依赖 Verl V1 的 TransferQueue，因此在入口处强制开启。
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            # 构造覆盖原生 PPO 流程的 SimpleTIR 三路训练器。
            self.trainer = SimpleTIRTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            # 进入主训练循环，并由 agent manager 持续产出多轮轨迹。
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    # 无论成功或异常都结束实验跟踪会话，避免日志悬挂。
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def compose_config(overrides: list[str]):
    import verl

    # 复用已安装 Verl 的基础 Hydra 配置，避免复制一份易漂移的配置。
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="simpletir"):
        return compose(config_name="ppo_trainer", overrides=overrides)


def main(argv: list[str] | None = None) -> None:
    # 将命令行覆盖项合并到 Verl 的 ppo_trainer 基础配置。
    config = compose_config(list(sys.argv[1:] if argv is None else argv))
    # 根据配置选择并写入当前控制进程使用的设备。
    auto_set_device(config)
    # 在分配 GPU 前执行 Verl 的全局配置一致性检查。
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
    # 并行度约束是本 overlay 的实现边界：rollout 每 GPU 一个独立 vLLM 服务
    # （TP=DP=1），actor FSDP 覆盖全部卡。违反时早期失败，避免训练中途
    # 才在分布式路径上炸掉。
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
    # 启动前校验训练集和验证集的字段契约与防泄漏边界。
    validate_simpletir_files([*train_files, *val_files])
    # 预检模式只验证配置和数据，不启动 Ray 或训练。
    if os.environ.get("SIMPLE_TIR_PREFLIGHT_ONLY") == "1":
        print("simpletir_preflight=ok")
        return
    # 通过 Verl 标准入口启动，但替换为本集成的远程任务运行器。
    run_ppo(config, task_runner_class=SimpleTIRTaskRunner)


if __name__ == "__main__":
    main()
