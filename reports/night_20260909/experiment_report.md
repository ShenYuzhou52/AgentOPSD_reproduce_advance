# 消融实验结果（GRPO / AgentOPSD / OPSD）


## 一、实验设置

| 项 | 值 |
|---|---|
| 模型 / 数据 | Qwen3.5-4B（`enable_thinking=false`）/ simplelr_math_35 train.parquet |
| 规模 | 200 步 × 16 prompt × 8 rollout，lr 1e-6，KL 系数 0.01，响应 2048 / 上下文 18432 |
| 工具环境 | 5 轮 Python 工具预算，bwrap 沙箱（sb_venv 科学计算库），沙箱超时 5s |
| 奖励 | 真实沙箱全文提取 `\boxed{}` → math_verify 二值分；无实质工具使用的正确回答记半分 |
| 验证 | 每 5 步在 `test_fixed100_s42`（种子 42 固定 100 题）上 temperature = 0 评估 |
| GPU/耗时 | 各 4 卡；GRPO/AgentOPSD 并行（9/8 当日完成），OPSD 15:52 auto-resume 后于 23:43 完成，约 7.6h |

基线参考（同模型、未训练）：test_fixed100 CoT 81% / TIR 78%；**AIME25 TIR 30%**（30 题）。

---

## 三、终评汇总

评测口径：合并后 HF 权重、温度 0、max-tokens 3072、5 轮工具、AIME24/AIME25 各 30 题。

| checkpoint | AIME24 acc | AIME25 acc | 备注 |
|---|---|---|---|
| **agentopsd_200** | **56.7%** | **43.3%** | boxed 76.7% / 83.3% |
| agentopsd_195 | 53.3% | 50.0% | |
| **grpo_200** | 43.3% | 20.0% | boxed 53.3% / 33.3% |
| grpo_195 | 46.7% | 43.3% | |
| **opsd_200** | 36.7% | 20.0% | |
| opsd_195 | 40.0% | 23.3% | |
| opsd_190 | 30.0% | 23.3% | |
| 基座（无训练） | — | 30.0% | `base4b_aime25_tir` |

读法（n=30，单点标准误约 ±9%，趋势重于单点）：
- **AgentOPSD 全面领先**，AIME24 比 GRPO 高 13.4 分、比 OPSD 高 20 分， boxed 率也最高（83%）。
- GRPO 的 step-200 在 AIME25 上回落到 20%（195 步还有 43.3%），与其验证曲线末段的 0.300 异常点相互印证——**GRPO 末段稳定性存疑**，AgentOPSD 的优势重塑可能正起到稳定作用。
- OPSD 增益最小（AIME24 36.7% vs 基座 AIME25 30% 口径不同不可直接比，但其验证曲线几乎无增长与终评弱势一致）。
- 中间 checkpoint 曲线受限：GRPO/OPSD 只保留了末 3 个 actor 格式断点（155/190 之前的 `data.pt` 新格式无法用现有 merger 合并），AgentOPSD 的 100/150 亦同——如需完整"分数-步数"曲线，白天需用 verl v1 loader 转换（见第七节遗留）。

---

## 四、训练过程观测（metrics.jsonl 每 25 步采样；全量在服务器 metrics.jsonl）

`acc`=训练 rollout 二值答对率；`sbx`=沙箱成功率；`code`=代码产出率；`void`=空转轮率；
`gnorm`=梯度范数；`kl`=KL 损失；`adv`=优势 [min,max]；`col`=collapse 信号。

### GRPO（grpo_formal200_s42）

| step | acc | sbx | code | void | gnorm | kl | adv | col |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.779 | 0.883 | 0.744 | 0.099 | 1.186 | 0.011 | ±2.475 | 0 |
| 25 | 0.951 | 0.863 | 0.561 | 0.028 | 0.752 | 0.033 | +0.94/-2.48 | 0 |
| 50 | 0.870 | 0.849 | 0.551 | 0.054 | 0.924 | 0.043 | ±2.475 | 0 |
| 100 | 0.847 | 0.764 | 0.610 | 0.032 | 0.779 | 0.064 | ±2.475 | 0 |
| 150 | 0.870 | 0.860 | 0.553 | 0.018 | 0.262 | 0.073 | +0.35/-2.48 | 0 |
| 200 | 0.878 | 0.916 | 0.618 | 0.012 | 1.064 | 0.087 | +0.73/-2.48 | 0 |

### AgentOPSD（agentopsd_formal200_s42）

| step | acc | sbx | code | void | gnorm | kl | adv | col |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.779 | 0.883 | 0.797 | 0.064 | 1.262 | 0.009 | +0.80/-2.72 | 0 |
| 25 | 0.876 | 0.939 | 0.786 | 0.110 | 1.404 | 0.026 | +2.72/-2.48 | 0 |
| 50 | 0.855 | 0.966 | 0.814 | 0.117 | 1.452 | 0.039 | +2.65/-2.48 | 0 |
| 100 | 0.794 | 0.964 | 0.775 | 0.046 | 0.914 | 0.045 | +1.78/-2.72 | 0 |
| 125 | 0.917 | 0.972 | 0.834 | 0.053 | 0.644 | 0.051 | +1.78/-2.72 | 0 |
| 200 | 0.856 | 0.973 | 0.718 | 0.043 | 1.796 | 0.065 | +1.78/-2.72 | 0 |

### OPSD（opsd_formal200_s42）

| step | acc | sbx | code | void | gnorm | kl | adv | col |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.757 | 0.791 | 0.764 | 0.125 | 0.010 | 0.002 | ±2.475 | 0 |
| 25 | 0.781 | 0.763 | 0.708 | 0.197 | 0.013 | 0.003 | ±2.475 | 0 |
| 75 | 0.635 | 0.763 | 0.585 | 0.264 | 0.008 | 0.003 | ±2.475 | 0 |
| 125 | 0.787 | 0.745 | 0.691 | 0.206 | 0.010 | 0.004 | ±2.475 | 0 |
| 175 | 0.540 | 0.816 | 0.555 | 0.372 | 0.010 | 0.004 | +2.18/-2.48 | 0 |
| 200 | 0.685 | 0.880 | 0.580 | 0.252 | 0.009 | 0.004 | ±2.475 | 0 |

观测要点：
- **全程零 collapse 信号**（三组 600 步合计 `agentopsd/collapse/signal=0`），无数值异常告警。
- **AgentOPSD 的沙箱成功率显著更高**（0.94-0.98 vs GRPO 0.76-0.92 / OPSD 0.75-0.88），代码产出率也更稳（0.62-0.83）——回合级信用分配下模型更倾向写"可执行"的代码。
- OPSD 的 grad_norm 小两个数量级（≈0.01）符合其设计（纯门控蒸馏、无策略梯度项）；其 rollout acc 反而最低且 void 轮最多（0.20-0.37），蒸馏信号更新慢的特征。
- 训练 rollout acc 三组都在 0.65-0.95 高位震荡（该数据集对 4B 已近天花板，与基线 78-81% 一致），**区分度主要体现在验证集与 AIME 终评**。

---

## 五、开发验证集曲线（val-core reward@1，100 题，每 5 步）

摘录（每 20 步 + 峰值；全量序列在 `train.log`，提取脚本 `extract_metrics.py`）：

| step | GRPO | AgentOPSD | OPSD |
|---|---|---|---|
| 0（初始） | 0.730 | 0.735 | 0.715 |
| 25 | 0.825 | 0.815* | 0.740 |
| 50 | 0.870* | 0.820 | 0.760 |
| 75 | 0.790 | 0.865 | 0.760 |
| 100 | 0.870* | 0.845 | 0.730 |
| 125 | 0.800 | **0.890** | 0.750 |
| 150 | 0.860 | 0.835 | 0.720 |
| 175 | 0.860* | 0.875 | 0.775 |
| 200（末值） | 0.830 | 0.855 | 0.720 |
| **峰值** | 0.910（step 45） | **0.915（step 129）** | 0.790（step 169） |

\* GRPO/OPSD 的采样步因 resume 步号错位取了邻近点（GRPO 序列步号为 0,5,9,15,19,… 的混合；OPSD/AgentOPSD 有 14/15、74/75 重叠，均为 resume 前后重复记录）。

**异常标记（GRPO）**：step-190 验证 = **0.300**（前后 0.820/0.900）——单点崩塌后立即恢复，与 grpo_200 在 AIME25 的 20% 回落相互印证，建议白天查看该步的验证明细/权重状态。
**AgentOPSD 末段最稳**：最后 30 步保持在 0.835-0.910；GRPO 末段 0.800-0.880 波动更大；OPSD 全程 0.71-0.79 无起色。

---

## 六、OPSA 第四臂进展（arXiv:2608.31046，"On-Policy Self-Adaptation"）

- **方法**：无教师/无奖励/无 KL——每条 rollout 取采样 log-prob 最低的 20% token，施加熵自适应负优势
  `A = -0.75 - 0.25·r`（r∈[-1,1] 由选点内熵归一化，δ=1），loss = -(1/|S|)Σ ratio·A（论文式 4/5）。官方代码是 slime 框架，本仓库为 verl 移植实现。
- **已完成**：`opsa_loss.py`（纯函数+verl 入口）、trainer 路由（`METHOD=opsa`）、teacher 前向豁免、启动脚本超参、
  单测 **11/11**、全库回归 **41/41**、preflight 通过；本地提交 `b480d5c` + `55c1df8`。
- **冒烟战果**：GPU 2 步冒烟在 step-1 actor 更新抓到真 bug——`ppo_micro_batch_size_per_gpu=1` 时 micro-batch 可能只含
  平衡补行的全零 padding 行，原守卫误判为异常。已修复（全零 mask → 连接计算图的零损失；"有真实 token 却选不出点"保留报错）。
- **未完成**：修复后的 GPU 冒烟 + 200 步正式启动——01:25 起 8 卡全被其他用户占用
  （0,1 ruiwenhu-sglang / 2,3 cljx / 4-7 weixuan-vllm），按 04:00 预案放弃今晚启动。
- **恢复步骤**（有空卡即可执行，预计 7.7h）：
  1. GPU 冒烟（修复后代码还没上过 GPU）：
     `cd /data2/ssd/yixinshen/AgentOPSD-tir && METHOD=opsa EXPERIMENT=opsa_smoke2_s42 RUN_ROOT=<night>/opsa_smoke CUDA_VISIBLE_DEVICES=<两空闲卡> N_GPUS=2 TRAIN_STEPS=2 TRAIN_PROMPTS=8 ROLLOUT_N=4 TEST_FREQ=200 SAVE_FREQ=2 RESUME_MODE=disable VAL_BEFORE_TRAIN=false bash scripts/run_simpletir_qwen35_4b.sh`
     判定：metrics.jsonl 落盘、`opsa/selected_ratio≈0.2`、奖励/梯度非 NaN。
  2. 正式：同 playbook 第三节命令（`EXPERIMENT=opsa_formal200_s42`，4 卡 200 步，RESUME_MODE=auto）。

---

## 七、夜间值守时间线（详见 timeline.md）

| 时间 | 事件 |
|---|---|
| 22:00-23:00 | 准备：修复本机 ssh 中文路径 bug（`/c/ssh-agentops/`）、部署夜间脚本、merger 三轮补丁（transformers 5.9 / megatron / 参数）、预合并 4 ckpt、评测链冒烟（结论：不可与训练共存挤卡） |
| 23:00 | OPSD 184/200 巡检正常；OPSA 实现开跑 |
| 23:43 | **OPSD 干净跑完 200 步**；链路自动 merge + 评测启动 |
| 23:56 | grpo/agentopsd 8 个评测完成（`EVALS_DONE`） |
| 00:00 | 发现链路 `sort -t_ -k3` 排序键 bug → OPSD 评测遗漏；手动补 merge 190/195/200 + 挂补评测链；OPSA 冒烟启动 |
| 00:14 | OPSD 6 个补跑评测完成，**14/14 评测全齐** |
| ~00:20 | OPSA 冒烟崩溃于 step-1 actor 更新（padding micro-batch bug） |
| 01:00 | 定位并修复 bug（11/11 回归），GPU 已全被占用 |
| 01:30-07:30 | 例行巡检：GPU 持续满员，按 04:00 预案放弃 OPSA 今晚启动 |

---

## 八、遗留事项（建议白天处理，均未自动执行）

1. **OPSA 恢复启动**（第六节两步）；活动窗口每晚 23:00-9:00 还剩 11 天。
2. **step 155 checkpoint 状态**：GRPO 的 `global_step_155` 目录只余 6.8 KB `data.pt` 元数据；训练日志显示 checkpoint manager 已自动删除其 `actor/` 分片，且远端未发现备份，因此无法转换、合并或评测。AgentOPSD 的旧 `data.pt` checkpoint 仍需另行确认其权重是否完整。
3. **GRPO step-190 验证 0.300 异常**：查该步验证明细与权重；grpo_200 的 AIME25 回落是否与此相关。
4. **agentopsd 旧 checkpoint 未裁剪**（global_step_5..200 全在，占约数百 GB）：确认无用后可手动清理（夜间未删任何文件）。
5. **overlay git 对账**：夜间代码走 scp 直推（overlay HEAD `da074fb` 与本地 `a632432` 历史有分叉），OPSA 两个 commit（`b480d5c`、`55c1df8`）尚在本地，白天统一 push + overlay pull。
6. 服务器 `/data2` 磁盘 71%（1.5T 空闲），夜间无风险。


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

最终评测先将 GRPO 与 AgentOPSD 的 FSDP checkpoint 合并为 Hugging Face 推理权重，再用独立 vLLM evaluator 评测。协议为 TIR 模式、最多 5 回合、每回合最多 3072 token、temperature=0.0；GRPO 与 AgentOPSD 各独立运行 3 次，Qwen3.5-4B Math500 基线当前为 1 次完整运行。

| 数据集 | 题数 | Qwen3.5-4B 基线 | GRPO 三次 answer accuracy | GRPO 均值 | GRPO - 基线 | AgentOPSD 三次 answer accuracy | AgentOPSD 均值 | AgentOPSD - 基线 | AgentOPSD - GRPO |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| SimpleLR Math test | 500 | **85.00%**（1 次） | 70.0%, 71.6%, 70.4% | **70.67%** | **-14.33 pp** | 90.0%, 90.0%, 90.6% | **90.20%** | **+5.20 pp** | **+19.53 pp** |
| AIME24 | 30 | — | 40.0%, 40.0%, 43.33% | **41.11%** | — | 53.33%, 50.0%, 53.33% | **52.22%** | — | **+11.11 pp** |

### GRPO checkpoint 对照：step 195 与 step 200

`grpo_195` 已按原始独立评测路径完成 AIME 评测（TIR、temperature=0、thinking 关闭、30 题）；该 checkpoint **未测 Math500**，因此不能将 step 200 的 Math500 结果外推给它。下表的 step 200 为同一批原始单次输出；上表中的 step-200 AIME24 三次均值保留为主结果。

| GRPO checkpoint | Math500 | AIME24 answer accuracy | AIME24 boxed | AIME25 answer accuracy | AIME25 boxed |
|---|---:|---:|---:|---:|---:|
| step 195 | **86.40%**（432/500；1 次） | **46.67%**（14/30） | 73.33% | **43.33%**（13/30） | 60.00% |
| step 200 | 已见主表：70.67%（3 次均值） | 43.33%（13/30） | 53.33% | 20.00%（6/30） | 33.33% |

step 195 在两套 AIME 上都高于 step 200，尤其 AIME25 高 23.33 pp；新增的完整 Math500 也达到 86.40%（boxed 90.60%，score 86.40%），比冻结基线高 1.40 pp、比 step-200 三次均值高 15.73 pp。AIME 的两个 checkpoint 各只有一轮 30 题，step 195 的 Math500 也只有一轮；它们共同构成“step 200 末段退化”的强证据，但仍应补齐 step 195 多次重复。
按 SimpleTIR 的无实质工具调用半分规则计算的 mean score：

| 数据集 | Qwen3.5-4B 基线 | GRPO | GRPO - 基线 | AgentOPSD | AgentOPSD - 基线 | AgentOPSD - GRPO |
|---|---:|---:|---:|---:|---:|---:|
| SimpleLR Math500 | 81.10% | 70.67% | -10.43 pp | 89.20% | +8.10 pp | +18.53 pp |
| AIME24 | — | 41.11% | — | 50.56% | — | +9.44 pp |

Qwen3.5-4B 基线在 Math500 上为单次完整评测：TIR、5 回合、每回合 3072 token、context 16384、temperature 0、thinking 关闭；因服务器缺少 nvcc，使用 vLLM 原生 sampler（VLLM_USE_FLASHINFER_SAMPLER=0）避开 FlashInfer JIT，模型和解码参数不变。三次运行的样本标准差：Math500 上 GRPO 为 0.83 pp、AgentOPSD 为 0.35 pp；AIME24 上两者均约为 1.92 pp。

## 4. 固定验证集的独立复测

为区分测试集构成与评测路径的影响，额外在与训练验证完全相同的固定 100 题上，以独立 evaluator 复测最终 checkpoint，并将每回合上限锁定为训练时的 2048 token：

| 设置：SimpleLR fixed-100，TIR 5 回合，2048 token | GRPO | AgentOPSD | 差值 |
|---|---:|---:|---:|
| answer accuracy | 76.00% | 88.00% | +12.00 pp |
| boxed ratio | 81.00% | 95.00% | +14.00 pp |
| score with halving | 76.00% | 87.00% | +11.00 pp |

因此，最终 500 题上的提升并非仅由更换测试集或从 2048 增至 3072 token 引起：在相同固定 100 题、相同 2048 token 的独立评测中，AgentOPSD 仍领先 12 pp。最终 500 题中差距进一步扩大，可能同时包含其余 400 题的题目组成与更长生成预算的影响；两者的贡献尚未通过 Math500-2048 对照完全分离。

## 5. 当前结论与限制

- 在本次固定配置、seed 42 与冻结外部评测协议下，AgentOPSD 相对 GRPO 在 Math500 和 AIME24 都有明显优势；Math500 上相对冻结 Qwen3.5-4B 基线为 **+5.20 pp**。
- GRPO 在 Math500 上为 **70.67%**，低于同协议冻结基线 **85.00%** 达 **14.33 pp**。这不是正常的训练增益，应作为 GRPO 训练或评测路径异常处理，而不是作为基线对照结果。
- GRPO 的 checkpoint 对照显示：step 195 的 Math500 为 **86.40%**（基线 +1.40 pp），AIME24/AIME25 为 **46.67%/43.33%**；step 200 对应为 70.67%（三次均值）与 43.33%/20.00%。因此异常集中在 **step 200 的训练末段退化**，而非 step 195 已经整体失效。
- 训练内 100 题 / 30 题 probe 没有显示同等量级的差距，不能将其直接作为最终模型能力排名；它更适合用于训练健康监控。
- 三次评测重复验证了推理解码的稳定性，但两组训练各只有一个训练随机种子。因此，当前结果支持“本配置与该 seed 下的强提升”，尚不足以宣称方法跨 seed 的普遍优势。
- 若要形成正式实验结论，应补充至少 3 个训练 seed，并在每个 checkpoint 上使用固定的独立 evaluator 进行 probe 与最终评测。

## 6. 原始结果位置

- GRPO checkpoint：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/grpo/grpo_formal200_s42/checkpoints/global_step_200`
- AgentOPSD checkpoint：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/agentopsd/agentopsd_formal200_s42/checkpoints/global_step_200`
- 最终评测总汇总：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/final_eval_20260908/final_eval_summary.md`
- 100 题 / 2048 token 对照：`/data2/ssd/yixinshen/experiments/qwen35-simpletir/final_eval_20260908/audit_fixed100_2048/`