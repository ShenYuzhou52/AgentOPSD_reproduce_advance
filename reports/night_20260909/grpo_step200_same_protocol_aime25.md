# GRPO step 200：AIME25 同口径独立复测

复测时间：2026-09-09。

## 目的

核验训练内验证在 step 200 报出的 AIME25 `12/30 = 40.0%`，是否由独立评测原先使用的更长生成预算（每回合 3072 token）或更短上下文窗口（16384）造成。

## 固定配置

| 项目 | 本次复测 |
|---|---|
| 权重 | `merged/grpo_200`（由 `global_step_200` FSDP actor checkpoint 合并） |
| 数据集 | `deepscaler/aime25.parquet`（30 题） |
| 模式 | SimpleTIR，Python sandbox |
| thinking | `false` |
| 回合上限 | 5 |
| 每回合生成上限 | 2048 token |
| 模型上下文上限 | 18432 token |
| temperature | 0.0 |
| GPU | CUDA device 4 |

除推理引擎仍为独立 vLLM evaluator 外，生成预算与训练内验证一致。

## 结果

| 指标 | 训练内验证（step 200） | 本次独立复测 |
|---|---:|---:|
| answer accuracy | 12/30 = **40.0%** | 6/30 = **20.0%** |
| boxed ratio | 15/30 = 50.0% | 10/30 = 33.3% |
| substantive tool use | 23/30 = 76.7% | 未汇总 |

结果目录：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/night_20260909/results/grpo_200_aime25_trainproto_2048_ctx18432/`。

## 结论

本次结果与原独立评测（3072 token、16384 context）的总分完全一致，均为 `6/30 = 20.0%`。因此，**2048/3072 token 和 18432/16384 context 的差异不能解释训练内 40% 与独立评测 20% 的落差**。

训练器代码的 step 200 时序为：`update actor → save checkpoint → update rollout weights → validate`，所以验证也不是在该步更新前的旧 actor 上运行。下一步应直接用原训练验证路径加载 `global_step_200` 重跑，逐题保留输出；重点核对训练内异步 rollout/agent-loop 与独立 vLLM evaluator 的请求、终止和响应解析是否一致。
