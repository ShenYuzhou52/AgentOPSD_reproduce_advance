# GRPO 代码走读：后训练推理引擎、verl 框架与训练循环（from code to code）

> 项目：Qwen3.5-4B + SimpleTIR + verl(v1/sync) + vLLM colocate。
> 本文沿"一条训练数据"的视角，从 launcher 敲回车到一次权重更新，逐层贴真实代码。

## 0. 全景

```
bash launcher（配置组装，无 GPU）
  → python -m integrations.simpletir_qwen35.main_tir（控制进程：hydra 校验）
    → Ray: SimpleTIRTaskRunner.run()（真正占 GPU 的远端进程）
      → SimpleTIRTrainer(PPOTrainerSync).fit()          ← 训练框架层（verl v1）
          每一步 step():
            ① replay_buffer.sample() 取 prompt
            ② agent_loop_manager.generate_sequences()   ← Agent 层（多轮工具循环）
                 └→ vLLM HTTP servers（×4 副本）         ← 后训练推理引擎层
                 └→ sandbox 执行 Python、打分
            ③ 旧 log-prob / 参考 log-prob
            ④ GRPO 优势（组内标准化）
            ⑤ actor.update_actor()（FSDP2 反传）
            ⑥ on_step_end → 新权重同步回 vLLM、休眠引擎
```

三层分工：verl 管"训练循环 + 权重"，agent loop 管"轨迹怎么生成"，vLLM 管"token 怎么吐"。

## 1. 入口层：launcher（bash → hydra 配置）

`scripts/run_simpletir_qwen35_4b.sh` 只做三件事：算配置、做预检、起 python。

```bash
ARGS=( "model_engine=dp" "trainer.use_v1=True" "trainer.v1.trainer_mode=sync"
       "algorithm.adv_estimator=grpo" "actor_rollout_ref.rollout.name=vllm" ... )
"${PYTHON_BIN}" -m integrations.simpletir_qwen35.main_tir "${ARGS[@]}" 2>&1 | tee -a train.log
```

`main_tir.py` 的 hydra 组装——不复制配置文件，叠加在已安装 verl 的基配置上：

```python
def compose_config(overrides):
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), job_name="simpletir"):
        return compose(config_name="ppo_trainer", overrides=overrides)
```

`data.train_files=[...]` 这类字符串是 dot-path 覆盖项，按 `ppo_trainer.yaml` schema 解析成嵌套
config；`validate_config()` 在纯 CPU 阶段拦住配置错误。

## 2. 控制层：Ray TaskRunner + TransferQueue

```python
@ray.remote
class SimpleTIRTaskRunner:
    def run(self, config):
        from integrations.simpletir_qwen35.trainer import SimpleTIRTrainer
        config.transfer_queue.enable = True
        tq.init(config.transfer_queue)                  # 数据总线初始化
        self.trainer = SimpleTIRTrainer(config=config)  # 注册表时机: 必须在worker内import
        self.trainer.init()
        self.agent_loop_manager = ...create(...)        # rollout/teacher/reward 句柄注入
        self.trainer.fit(self.agent_loop_manager)       # 主循环
```

TransferQueue（TQ）是 verl v1 的 KV 数据总线：驱动进程按 key（`{uid}_{session}_{turn}`）
读写数据，rollout 与验证共用，`partition_id="train"/"val"` 分区。

## 3. 数据层：parquet → prompt

数据契约（`data_contract.py`）：`prompt` 是 chat 消息列表、`reward_model.ground_truth`
与 prompt 平级（绝不进 prompt 文本，防答案泄漏）。`enable_thinking` 通过模板开关控制
`<think>` 解码形态——同一份权重两种形态。

## 4. 推理引擎层：vLLM 嵌入训练（colocate / hybrid engine）

### 4.1 拓扑：4 卡 = 4 个独立 vLLM 副本 + 全局负载均衡

`verl/workers/rollout/llm_server.py`:

```python
rollout_world_size = tp * dp * pp                  # 1×1×1
num_replicas = world_size // rollout_world_size    # 4
self.rollout_replicas = [replica_class(replica_rank=i) for i in range(num_replicas)]
self._init_global_load_balancer()
```

每个副本是完整 vLLM 引擎（自带 KV）。`GPU_MEMORY_UTILIZATION` 调的就是每副本 KV 预算。

### 4.2 权重同步与休眠（`trainer_sync.py`）

```python
@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    def on_step_end(self):
        self.checkpoint_manager.update_weights(self.global_steps)  # FSDP→gather→广播→load
    def on_sample_end(self):
        self.checkpoint_manager.sleep_replicas()   # 卸权重、丢KV，显存还给训练
```

时序即显存分工：rollout 时 vLLM 占大头；采样完休眠；反传；再唤醒灌新权重。

### 4.3 采样双面性

训练 `temperature=1.0`（探索），验证 `val_kwargs.temperature=0`（确定性）——同一批引擎两套
SamplingParams，是后训练引擎与普通推理服务的本质区别。

## 5. Agent 层：SimpleTIR 循环

```python
def generate_sequences(self, prompts):            # agent_loop_tq.py
    chunkes = prompts.chunk(len(self.agent_loop_workers))
    ray.get([w.generate_sequences.remote(c) for w, c in zip(...)])
```

worker 内 `SimpleTIRPythonAgentLoop` 逐轮（`simpletir_agent_loop.py`）：

```python
for turn_step in range(self.max_turns):
    remaining = self.max_episode_response_tokens - episode_response_tokens
    turn_sampling_params["max_tokens"] = min(requested, remaining)   # 总预算硬顶
    generated = await self.server_manager.generate(prompt_ids=...)
    merged, response_mask, _ = await self.ct_merge_assistant_token(...)
    output = AgentLoopOutput(prompt_ids=..., response_ids=response_ids,
                             response_mask=response_mask, ...)        # 每轮一行训练行
    obs = run_python(code, timeout=5)
    runtime_prompt_ids = merged.token_ids + obs_tokens               # 观察进 prompt 侧
```

要点：每轮一行训练样本（prompt=完整历史，response=本轮生成，观察 token mask=0 不进损失）；
episode 末 `score_simpletir_math(...)` 打分回填 `rm_scores`。

## 6. 训练循环层：fit → step → _step_once

```python
while ... and self.global_steps <= self.total_training_steps:
    batch = self.step(metrics, timing_raw)          # ① 采样+训练
    if step % save_freq == 0: self._save_checkpoint()
    self.on_step_end()                              # 权重同步回vLLM
    if step % test_freq == 0: val_metrics = self._validate()
    self._compute_metrics(batch, metrics, ...)      # 那行指标日志的来源
    tq.kv_clear(keys=batch.keys, ...)
    self.logger.log(data=metrics, step=...)         # → wandb/console
```

`_step_once()` 七步：sample（按 global_steps 防陈旧，保 on-policy）→ reward → balance →
old_log_prob → ref_log_prob → advantage → update_actor（FSDP2）。

## 7. 算法层：GRPO 三段真代码

优势（`core_algos.py:268`）——无 critic，baseline=同组均值：

```python
scores = token_level_rewards.sum(dim=-1)
for idx in id2score:                    # 按 prompt 分组(同题 n=8 条rollout)
    id2mean[idx], id2std[idx] = mean, std
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
scores = scores.unsqueeze(-1) * response_mask
```

组全对/全错 → std=0 → 零梯度（训练集校准 0.425 的代码依据）。`norm_adv_by_std_in_grpo=False`
即 Dr.GRPO 变体。

损失（`core_algos.py:1210`）：

```python
ratio = torch.exp(clamp(log_prob - old_log_prob, -20, 20))
pg1 = -advantages * ratio
pg2 = -advantages * torch.clamp(ratio, 1-ε_low, 1+ε_high)   # 0.2/0.2
loss = masked_mean(torch.max(pg1, pg2), response_mask)       # 只对模型token平均
# + kl_loss_coef=0.01 × (log_prob - ref_log_prob) 锚定基座
```

我们的扩展点（`integrations/.../trainer.py`）：`_compute_advantage` 先跑原生 GRPO，
`method=grpo` 臂纯透传——四臂消融共享 rollout/奖励/数据，唯二分岔在优势与损失。

## 8. 显存编排：FSDP2 + colocate

`engine_workers.py`：设备网格 `(dp, infer_tp, infer_pp)`；`fsdp_size=4`、
`reshard_after_forward=True`、`ppo_micro_batch_size_per_gpu=1`（40k 长序列）、
`use_remove_padding=True`（Qwen3.5 的 FA packed 要求）。
单卡 80G 四阶段轮转：rollout(vLLM 55%) → 休眠 → 反传 → 唤醒。

## 9. 周边层

验证：TQ val 分区 + `data_source` 分组指标 + `validation_data_dir` 全量落盘 +
`log_val_generations` wandb 表格。checkpoint：FSDP 分片 + `latest_checkpointed_iteration.txt`
+ `resume_mode=auto`。监控：指标行 → wandb + `monitor_overlong.py` 增量正则 + 熔断。

## 10. 参数 → 代码位置速查

| launcher 参数 | 位置 |
|---|---|
| trainer_mode=sync | PPOTrainerSync.on_step_end/on_sample_end（§4.2） |
| rollout.name/mode | LLMServerManager 副本与负载均衡（§4.1） |
| gpu_memory_utilization / max_model_len | vLLM 引擎 KV 预算/上下文 |
| agent.num_workers | AgentLoopManagerTQ.generate_sequences chunk（§5） |
| rollout.n | GRPO index 分组（§7.1） |
| temperature / val_kwargs | SamplingParams 双面性（§4.3） |
| +simpletir.max_episode_response_tokens | agent loop 逐轮 min(requested, remaining)（§5） |
| adv_estimator=grpo | compute_grpo_outcome_advantage（§7.1） |
| clip_ratio / kl_loss_coef | compute_policy_loss（§7.2） |
| fsdp_size / remove_padding / micro | engine_workers 设备网格（§8） |
| test_freq / save_freq / validation_data_dir | fit() 分支（§6、§9） |
