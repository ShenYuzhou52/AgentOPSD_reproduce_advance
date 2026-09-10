from pathlib import Path
import math
import matplotlib.pyplot as plt

root = Path('reports/night_20260909')
src = (root / 'training_validation_comparison.md').read_text(encoding='utf-8')
a = src.index('| step |')
b = src.index('\n\n## 读法', a)
table = src[a:b]
rows = []
for line in table.splitlines()[2:]:
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    if len(c) != 7: continue
    rows.append([int(c[0])] + [math.nan if x == '—' else float(x) for x in c[1:]])
steps = [r[0] for r in rows]
colors = {'GRPO':'#D55E00','AgentOPSD':'#0072B2','OPSD':'#009E73'}
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
for ax, title, cols in [
    (axes[0], 'SimpleLR fixed-100 validation (reward@1)', [(1,'GRPO'),(2,'AgentOPSD'),(3,'OPSD')]),
    (axes[1], 'AIME25 training probe (reward@1)', [(4,'GRPO'),(5,'AgentOPSD'),(6,'OPSD')]),
]:
    for col, label in cols:
        ax.plot(steps, [r[col] for r in rows], marker='o', markersize=3, linewidth=1.8, label=label, color=colors[label])
    ax.set_title(title, loc='left', weight='bold')
    ax.set_ylabel('reward@1'); ax.set_ylim(0, 1); ax.grid(alpha=.25); ax.legend(ncol=3, frameon=False)
axes[0].axvline(190, color='#888', linestyle='--', linewidth=1)
axes[0].annotate('GRPO 0.300 anomaly', xy=(190,.3), xytext=(150,.16), arrowprops={'arrowstyle':'->','color':'#666'}, color='#555', fontsize=9)
axes[1].set_xlabel('training step'); axes[1].set_xticks(range(0,201,20))
fig.savefig(root/'training_validation_curves.png', dpi=200, bbox_inches='tight')
plt.close(fig)

report = f'''# Qwen3.5-4B SimpleTIR 三臂实验数据报告

本文汇总 GRPO、AgentOPSD 与 OPSD 的训练中指标、独立终评和 AgentOPSD checkpoint 评测。除特别注明外，比例均为 answer accuracy；括号内为独立评测次数。

## 一、实验内容、模型与参数

实验比较三种训练信号对 Qwen3.5-4B 数学工具推理的影响。训练集为 `simplelr_math_35/train.parquet`：8,523 道单轮数学题，提供标准最终答案与元数据，不提供教师解题轨迹或工具调用轨迹；模型通过在线 rollout 和结果奖励学习。

| 项目 | 设置 |
|---|---|
| 基座模型 | Qwen3.5-4B，`enable_thinking=false` |
| 训练规模 | 200 steps；每步 16 prompts × 8 rollouts；学习率 1e-6 |
| 训练环境 | 最多 5 轮 Python 工具；每轮最大响应 2,048 tokens；上下文 18,432；沙箱超时 5 秒 |
| 训练/验证解码 | rollout temperature=1；训练内验证 temperature=0；thinking 关闭 |
| 最终独立评测 | 合并 HF 权重 + vLLM evaluator；TIR、5 轮、每轮 3,072 tokens、context 16,384、temperature=0 |
| 奖励 | `\\boxed{{}}` 最终答案匹配；正确但无实质工具使用按 SimpleTIR 规则记半分 |

| 算法 | 训练信号 |
|---|---|
| GRPO | 任务结果奖励驱动的标准 GRPO 策略梯度 |
| AgentOPSD | GRPO 加答案条件 teacher log-prob 的回合级 advantage 重塑 |
| OPSD | 置信度门控自蒸馏，无策略梯度项 |

## 二、最终评测数据

主表使用同一 TIR 独立评测协议。Math500 的基线、OPSD 和 GRPO step 195 是单次完整运行；GRPO 与 AgentOPSD step 200 的 Math500 为三次独立运行。AIME24 的 GRPO/AgentOPSD 为三次均值、OPSD 为单次；AIME25 三臂为夜间单次，基线为三次均值。

| 数据集 / 指标 | 未训练 Qwen3.5-4B | GRPO step 200 | AgentOPSD step 200 | OPSD step 200 |
|---|---:|---:|---:|---:|
| Math500 answer accuracy（500 题） | **85.00%**（1） | **70.67%**（3：70.0/71.6/70.4） | **90.20%**（3：90.0/90.0/90.6） | **84.20%**（1） |
| Math500 SimpleTIR score | 81.10%（1） | 70.67% | 89.20% | 78.30%（1） |
| Math500 boxed ratio | 87.80%（1） | 75.13%（3 次均值） | 98.47%（3 次均值） | 88.40%（1） |
| AIME24 answer accuracy（30 题） | 33.33%（TIR，3） | 41.11%（3） | 52.22%（3） | 36.67%（1） |
| AIME25 answer accuracy（30 题） | 24.44%（TIR，3） | 20.00%（1） | 43.33%（1） | 20.00%（1） |

无工具 CoT 基线为 AIME24 40.00%、AIME25 26.67%（均 3 次）。主表采用 TIR 基线以匹配三臂工具评测。Math500 基线设置 `VLLM_USE_FLASHINFER_SAMPLER=0` 以避开服务器无 nvcc 时的 FlashInfer JIT，模型和解码参数未改变。

### GRPO checkpoint 对照

| checkpoint | Math500 | AIME24 | AIME25 | 说明 |
|---|---:|---:|---:|---|
| step 195 | **86.40%**（432/500，1；boxed 90.60%，score 86.40%） | 46.67%（14/30） | 43.33%（13/30） | 高于基线，且全面高于 step 200 |
| step 200 | 70.67%（3 次均值） | 43.33%（夜间单次；三次均值见主表） | 20.00%（6/30） | 训练末段退化 |

## 三、训练中指标数据

以下为训练内 temperature=0、单 rollout 的 `reward@1` 大表。SimpleLR 为固定 100 题开发集，AIME25 为训练中挂载的 30 题 probe；`—` 表示该阶段没有落盘该 probe。同一 global step 的 resume 重复记录取最后一条。训练内指标用于监控趋势，不能替代第二部分独立终评。

{table}

![训练中验证曲线](training_validation_curves.png)

图中上半部分为 fixed-100，下半部分为 AIME25 probe。GRPO 的 AIME25 指标仅从 step 145 开始落盘；fixed-100 的 step 190 出现一次 0.300 异常点，后续立即恢复。

### step 200 训练 rollout 摘要

| 算法 | rollout acc | sbx | code | void | grad norm | KL |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 0.878 | 0.916 | 0.618 | 0.012 | 1.064 | 0.087 |
| AgentOPSD | 0.856 | 0.973 | 0.718 | 0.043 | 1.796 | 0.065 |
| OPSD | 0.685 | 0.880 | 0.580 | 0.252 | 0.009 | 0.004 |

`sbx` 为沙箱成功率，`code` 为代码产出率，`void` 为空转轮率。三臂训练日志的 collapse signal 均为 0。

## 四、AgentOPSD checkpoint 评测数据

AgentOPSD 保存了 60/65/70/75 四个中期完整 checkpoint，均已完成 AIME24/AIME25 各三次 TIR 独立评测。195 为夜间单次；200 的 AIME24 采用三次均值，AIME25 为夜间单次。中期 checkpoint 未测 Math500。

| step | AIME24 | 运行次数 | AIME25 | 运行次数 | 训练内 fixed-100 |
|---:|---:|---:|---:|---:|---:|
| 基线 TIR | 33.33% | 3 | 24.44% | 3 | 0.735 |
| 60 | 43.33% | 3 | 40.00% | 3 | 0.825 |
| 65 | 43.33% | 3 | 33.33% | 3 | ~0.830 |
| 70 | **53.33%** | 3 | 36.67% | 3 | 0.865 |
| 75 | 43.33% | 3 | **46.67%** | 3 | 0.865 |
| 195 | 53.33% | 1 | 50.00% | 1 | 0.835 |
| 200 | 52.22% | 3 | 43.33% | 1 | 0.855 |

60–75 的 24 个三次重复评测均完全一致。中期 AIME 增益在 step 60 已出现，后续主要为 1–2 题量级波动；它们不能与只保留 195/200 的 GRPO 或仅有夜间末段终评的 OPSD 构成同密度曲线。

## 五、结论

- **AgentOPSD 的 step 200 是本轮最强终评 checkpoint**：Math500 90.20%，高于基线 5.20 pp；AIME24 三次均值 52.22%，高于 GRPO 11.11 pp；AIME25 夜间单次为 43.33%。
- **GRPO 对 checkpoint 很敏感**：step 195 的 Math500 为 86.40%，高于基线 1.40 pp；继续训练至 step 200 后降至 70.67%，AIME24/AIME25 由 46.67%/43.33% 降至 43.33%/20.00%，显示末段退化。
- **OPSD 的 Math500 answer accuracy 接近基线**（84.20% vs 85.00%），但 score 低 2.80 pp，AIME24/AIME25 仅为 36.67%/20.00%；其低 rollout acc、高 void 与极小梯度和终评弱势一致。
- 训练内 fixed-100/AIME25 probe 显示 AgentOPSD 后程较稳、GRPO 固定验证波动较大、OPSD 平台化；但训练 reward 不能替代独立终评，checkpoint 选择必须依赖外部评测。
- 当前仅一个训练 seed，且部分 AIME/Math500 点为单次运行。正式结论需补足多 seed，并对候选 checkpoint 使用同一 evaluator 重复评测。
'''
(root / 'data_report.next.md').write_text(report, encoding='utf-8')
print('generated')