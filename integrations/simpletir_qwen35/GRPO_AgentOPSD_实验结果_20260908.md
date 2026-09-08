# Qwen3.5-4B SimpleTIR：GRPO 与 AgentOPSD 实验结果

> 记录日期：2026-09-08。本文只包含已完成的 GRPO 与 AgentOPSD 对比；两者均取 `step 200` 最终 checkpoint。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| 基座模型 | `Qwen3.5-4B` |
| 训练任务 | SimpleTIR Python 工具增强数学推理 |
| 训练集 | `simplelr_math_35/train.parquet` |
| 训练步数 / 随机种子 | 200 steps / seed 42 |
| 训练资源 | 4 GPUs，FSDP2，同步 V1 trainer |
| 每个训练 batch | 16 个题目、每题 8 条 rollout |
| 优化器学习率 | `1e-6` |
| 训练 agent 限制 | 最多 5 回合；每回合最大响应长度 2048 token；Python sandbox 超时 5 秒 |
| GRPO | 保留 Verl 原始 GRPO advantage 与 actor 损失 |
| AgentOPSD | 在 GRPO advantage 基础上，使用答案条件 teacher log-prob 进行跨 turn credit 重整；`lam=0.5`、`b=0.2`、`gamma=0.95` |

训练中 probe 使用两个集合：

- `simplelr_math_35/test_fixed100_s42.parquet`：从 Math500 测试集固定抽取的 100 题。
- `deepscaler/aime25.parquet`：AIME25，30 题。

## 2. 训练结束时的内置验证

下表为训练过程最后一次验证（step 200）。这里的验证由训练 rollout 路径执行，主要用于检查训练健康和趋势，不作为最终主结果。

| 指标 | GRPO | AgentOPSD | AgentOPSD - GRPO |
|---|---:|---:|---:|
| SimpleLR-100 answer accuracy | 83.00% | 87.00% | +4.00 pp |
| SimpleLR-100 reward | 83.00% | 85.50% | +2.50 pp |
| SimpleLR-100 boxed ratio | 92.00% | 93.00% | +1.00 pp |
| AIME25 answer accuracy / reward | 40.00% | 36.67% | -3.33 pp |
| AIME25 boxed ratio | 50.00% | 56.67% | +6.67 pp |
| AIME25 substantive tool-use ratio | 76.67% | 86.67% | +10.00 pp |
| 平均 agent 回合数（两 probe 合并） | 3.62 | 2.51 | -1.12 |

AgentOPSD 在 step 200 的 credit 诊断正常：`reshape_applied=1`、`adv_nonfinite_ratio=0`、`adv_large_ratio=0`，collapse alert、credit dead 与数值 alert 均为 0。

## 3. 最终独立评测：三次重复均值

最终评测先将两个 FSDP checkpoint 合并为 Hugging Face 推理权重，再用独立 vLLM evaluator 评测。协议完全固定：TIR 模式、最多 5 回合、每回合最多 3072 token、`temperature=0.0`。每个模型与数据集组合独立运行 3 次。

| 数据集 | 题数 | GRPO 三次 answer accuracy | GRPO 均值 | AgentOPSD 三次 answer accuracy | AgentOPSD 均值 | 差值 |
|---|---:|---|---:|---|---:|---:|
| SimpleLR Math test | 500 | 70.0%, 71.6%, 70.4% | **70.67%** | 90.0%, 90.0%, 90.6% | **90.20%** | **+19.53 pp** |
| AIME24 | 30 | 40.0%, 40.0%, 43.33% | **41.11%** | 53.33%, 50.0%, 53.33% | **52.22%** | **+11.11 pp** |

按 SimpleTIR 的无实质工具调用半分规则计算的 mean score：

| 数据集 | GRPO | AgentOPSD | 差值 |
|---|---:|---:|---:|
| SimpleLR Math500 | 70.67% | 89.20% | +18.53 pp |
| AIME24 | 41.11% | 50.56% | +9.44 pp |

三次运行的样本标准差：Math500 上 GRPO 为 0.83 pp、AgentOPSD 为 0.35 pp；AIME24 上两者均约为 1.92 pp。Math500 的 19.53 pp 差距显著大于本协议下的重复解码波动。

## 4. 固定验证集的独立复测

为区分测试集构成与评测路径的影响，额外在与训练验证完全相同的固定 100 题上，以独立 evaluator 复测最终 checkpoint，并将每回合上限锁定为训练时的 2048 token：

| 设置：SimpleLR fixed-100，TIR 5 回合，2048 token | GRPO | AgentOPSD | 差值 |
|---|---:|---:|---:|
| answer accuracy | 76.00% | 88.00% | +12.00 pp |
| boxed ratio | 81.00% | 95.00% | +14.00 pp |
| score with halving | 76.00% | 87.00% | +11.00 pp |

因此，最终 500 题上的提升并非仅由更换测试集或从 2048 增至 3072 token 引起：在相同固定 100 题、相同 2048 token 的独立评测中，AgentOPSD 仍领先 12 pp。最终 500 题中差距进一步扩大，可能同时包含其余 400 题的题目组成与更长生成预算的影响；两者的贡献尚未通过 Math500-2048 对照完全分离。

## 5. 当前结论与限制

- 在本次固定配置、seed 42 与冻结外部评测协议下，AgentOPSD 相对 GRPO 在 Math500 和 AIME24 都有明显优势。
- 训练内 100 题 / 30 题 probe 没有显示同等量级的差距，不能将其直接作为最终模型能力排名；它更适合用于训练健康监控。
- 三次评测重复验证了推理解码的稳定性，但两组训练各只有一个训练随机种子。因此，当前结果支持“本配置与该 seed 下的强提升”，尚不足以宣称方法跨 seed 的普遍优势。
- 若要形成正式实验结论，应补充至少 3 个训练 seed，并在每个 checkpoint 上使用固定的独立 evaluator 进行 probe 与最终评测。

## 6. 原始结果位置

- GRPO checkpoint：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/grpo/grpo_formal200_s42/checkpoints/global_step_200`
- AgentOPSD checkpoint：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/agentopsd/agentopsd_formal200_s42/checkpoints/global_step_200`
- 最终评测总汇总：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/final_eval_20260908/final_eval_summary.md`
- 100 题 / 2048 token 对照：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/final_eval_20260908/audit_fixed100_2048/`