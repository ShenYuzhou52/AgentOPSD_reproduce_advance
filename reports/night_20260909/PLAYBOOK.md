# 夜间值守剧本 2026-09-08 → 09-09（Qwen3.5-SimpleTIR 三方消融 + OPSA 第四臂）

本文件是今晚所有无人值守轮次的**唯一行动依据**。每轮：跑快照 → 查决策表 → 行动 → 追加 timeline → 结束本轮。**不要在本轮内长睡眠等待**，等不到的事留给下一轮（30 分钟后）。

## 0. 连接方式（已验证）

```bash
SSH="ssh -F /c/ssh-agentops/config -o BatchMode=yes 124.128.251.62"
# 状态快照（只读，每轮第一步）：
$SSH 'bash /data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/night_20260909/status_snapshot.sh'
```

关键路径：
- 正式实验根：`$R = /data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907`
- 夜间目录：`$NIGHT = $R/night_20260909`（markers/ merged/ results/ + 各脚本）
- overlay 仓库：`/data2/ssd/yixinshen/AgentOPSD-tir`（OPSA 代码写这里）
- 训练入口：`$OVERLAY/scripts/run_simpletir_qwen35_4b.sh`（METHOD 环境变量切换方法臂）
- 本地报告：`C:\Desktop\Thu-CS\Temp_for_ssh\AgentOPSD\reports\night_20260909\`（timeline.md 每轮追加；morning_report.md 最后生成）

## 1. 决策表（按快照输出分派）

| 快照状态 | 行动 |
|---|---|
| OPSD 进程在跑，无异常 | 本轮只记录 step/磁盘。异常=：`agentopsd/collapse/signal=1` 连续 2 轮、metrics NaN、磁盘 <40G、step 连续 2 轮不推进。异常只记录不动作（除非"进程消失"） |
| OPSD 进程消失，且无 `markers/OPSD_EXIT` | **已授权自动续跑**：确认 `$R/opsd/.../checkpoints/latest_checkpointed_iteration.txt` 存在后，`$SSH 'bash $NIGHT/auto_resume_opsd.sh'`，记录到 timeline。若 latest 文件不存在则不动作，标记报警 |
| `markers/OPSD_EXIT` + `OPSD_CLEAN200` | 正常完成。检查 `$NIGHT/night_chain.log` 与 `merged/merge_opsd_*.log` 是否有 merge 失败；等 `markers/MERGE_DONE` → `EVALS_DONE`（链路自动跑评测，约 30-60 分钟）。本轮转去做第 2 节 OPSA 实现（若未完成） |
| `markers/OPSD_EXIT` + `OPSD_UNCLEAN` | 链路已停。按"OPSD 进程消失"行处理（续跑授权有效）。续跑后 chain 需要重启：`$SSH 'cd $NIGHT && nohup bash night_chain.sh >> night_chain.log 2>&1 &'` |
| `markers/EVALS_DONE` 且 OPSA 尚未启动 | 执行第 3 节：OPSA 冒烟 → 正式启动 200 步 |
| OPSA (`opsa/opsa_formal200_s42`) 在跑 | 看护同 OPSD 行。OPSA 崩溃且 step<200 → 同样**授权续跑**：原命令重发（见第 3 节启动命令，RESUME_MODE=auto） |
| `markers/EVALS_DONE` 且 OPSA 在跑且时间 ≥08:00 | 生成晨报（第 4 节）后结束本轮 |

## 2. OPSA 实现指南（23:00-评测结束之间的空档做掉）

**论文**：arXiv 2608.31046 "Does On-Policy Distillation Really Distill?"，方法 OPSA（官方代码是 slime 框架的 github.com/DripNowhy/On-Policy-Self-Adaptation，**不适用本 verl 管线，移植损失即可**）。

**方法本质**（按论文式 4/5，逐字实现）：
- 每个 rollout 的学生 token 中，取 **log-prob 最低的 20%**（下标集合 S，按 response 内排序），只用这些位置计算损失
- 这些位置的优势：`A_i = -0.75 - 0.25 * r_i`，其中 `r_i = 2*(H_i - H_min)/(H_max - H_min) - 1`（A∈[-1,-0.5]，熵越高负优势越强；H_min/H_max 是 S 内熵的最小/最大值）
- 损失：`L = -(1/|S|) * Σ_{i∈S} A_i * log π(y_i|...)`。**无教师、无任务奖励、无参考模型/KL**（opsa 臂设 `use_kl_loss=False`）
- H_i = 该位置策略的 token 熵，需要 actor forward 给 entropy。verl 中 `entropy_coeff != 0` 才算 entropy——给 opsa 臂设 `entropy_coeff=1e-8`（量级无害、强制计算），或在 trainer 钩子里直接取 forward 的 entropy；以 verl 实际代码为准（`verl/trainer/ppo/core_algos.py`、`verl/workers/actor/dp_actor.py`，先读再写）
- 验证集照常打分（用于报告），但奖励**不得**进入优势

**实现位置**（全部在 overlay，新文件 + 最小分支，不动 verl 基座、不动已有三臂行为）：
1. 读 `integrations/simpletir_qwen35/trainer.py` 与 `opsd_loss.py`，弄清 `sdar_only_loss` 怎么挂进 actor loss、method 分支在哪路由
2. 新增 `integrations/simpletir_qwen35/opsa_loss.py`（纯函数 + 单测）
3. `run_simpletir_qwen35_4b.sh` 的 case 行加 `opsa`；trainer/method 路由加 opsa 分支；opsa 臂专属 config（kl_loss_coef=0 / entropy 强制计算 / 优势替换）走 `+simpletir.method=opsa` 一致风格
4. 单测（仿 `test_*.py`，CPU 可跑）：20% 选取正确性、A_i 端点值（熵最低 → -0.5，最高 → -1）、mask 外 token 无梯度、奖励不泄漏进优势
5. 验证链（逐级，都过才进下一级）：
   a. CPU 单测全绿
   b. `$SSH 'cd $OVERLAY && PREFLIGHT_ONLY=1 SIMPLE_TIR_PREFLIGHT_ONLY=1 METHOD=opsa ... bash scripts/run_simpletir_qwen35_4b.sh'`（配置预检）
   c. 冒烟 2 步：同启动命令但 `TRAIN_STEPS=2 EXPERIMENT=opsa_smoke2_s42 SAVE_FREQ=2 TEST_FREQ=2`，确认奖励/指标落盘、无 NaN（注意：冒烟也要等 GPU 0-3 空；若与评测链冲突，等 EVALS_DONE）
6. 每完成一步，把代码（本地仓库）+ 进度写到 timeline.md；**代码同步先落本地 git commit，再 git push + overlay pull**（`bash scripts/git_sync.sh` 语义），或直接 scp 到 overlay 对应路径（次选，morning 补 commit）

## 3. OPSA 正式启动（EVALS_DONE 后，GPU 0-3 全空时）

```bash
$SSH 'cd /data2/ssd/yixinshen/AgentOPSD-tir && \
METHOD=opsa EXPERIMENT=opsa_formal200_s42 RUN_ROOT=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/opsa \
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 TRAIN_STEPS=200 TRAIN_PROMPTS=16 ROLLOUT_N=8 MAX_TURNS=5 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=18432 ACTOR_MINI_BATCH=8 \
GPU_MEMORY_UTILIZATION=0.40 TEST_FREQ=5 SAVE_FREQ=5 MAX_CKPTS=3 RESUME_MODE=auto \
VAL_BEFORE_TRAIN=true DATALOADER_WORKERS=0 \
nohup bash scripts/run_simpletir_qwen35_4b.sh >> /data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907/night_20260909/opsa_train.log 2>&1 & echo "OPSA pid $!"'
```
- 预计 139 s/step，200 步 ≈ 7.7h。00:30 启动 → 08:15 左右完成；更晚启动就跑过 9:00，无人在场也安全（auto-resume 已授权，白天用户可查）
- 若到 04:00 仍未达到"单测+预检+冒烟全过"，**放弃今晚启动**，把进度写清楚留给明天——绝不带病上线 8 小时训练

## 4. 晨报（08:00 后的第一轮或 EVALS_DONE 后 OPSA 已在跑时）

生成 `C:\Desktop\Thu-CS\Temp_for_ssh\AgentOPSD\reports\night_20260909\morning_report.md`：
1. **终评表**：`$SSH 'cat $NIGHT/results/summary.tsv'` → grpo_195/200、agentopsd_195/200、opsd_195/200 × {aime, aime25}；对照基座 30%（base4b_aime25_tir）
2. **训练曲线摘要**：三方 metrics.jsonl 的 step∈{50,100,150,200} 的 `critic/score/mean`、`answer_accuracy`、`val` 分数（若有），OPSA 到当前 step
3. **OPSA 状态**：step、score 趋势、预计完成时间
4. **异常时间线**：从 timeline.md 汇总（续跑、崩溃、磁盘告警）
5. **遗留事项**：data.pt 格式 ckpt（agentopsd_100/150、grpo_155）未能 merge——建议白天用 verl v1 loader 转；agentopsd 旧 ckpt 未裁剪占磁盘——建议清理（等用户确认，不自动删）

## 5. 红线（任何情况下不得违反）

1. **不碰 GPU 4-7**：PID 2038123+ 是别人/其他任务，不动、不评论、不等待其结束
2. **不删除任何文件**：checkpoint 裁剪、磁盘清理只写建议
3. **不修改** verl 基座（`benchmarks/verl-qwen35-base`）、SimpleTIR 上游、已验证的三臂脚本行为；OPSA 只新增文件 + 最小 method 分支
4. **不 kill 任何训练进程**；唯一允许的"启动"类动作 = 剧本给出的续跑/启动命令
5. 服务器上只写 `$NIGHT`、`$R/opsa`、overlay 内 OPSA 新文件
6. 每轮动作（含"无异常"）都追加到本地 `reports/night_20260909/timeline.md`：时间、快照要点、动作、结果

## 6. 已验证事实（免得重新发现）

- merge 链已通：`model_merger_patched.py`（transformers 5.9 兼容 + megatron 回退 + `--hf_model_path` 必填、无 `merge` 子命令）；**仅 actor/ 分片格式可 merge**；agentopsd_100/150、grpo_155 是 data.pt 单文件格式，跳过
- 评测链已冒烟：`eval_ckpt.sh <HF目录> <parquet> <GPU> <输出> <标签> [gpu_mem]`，温度 0、max-tokens 3072、5 轮工具，结果追加 `results/summary.tsv`
- **冒烟重要结论**：合并目录（权重+config+tokenizer+processor）vllm 加载全部正常；但**绝不能在训练占用的卡上挤占跑评测**——gpu_mem 0.20 时 CUDA graph 预算超限（KV 为负）引擎直接起不来。评测只等 GPU 空后按默认 0.80 跑（今天下午验证过的同款配置）
- **评测矩阵第一票否决**：若当晚第一个评测 `answer_accuracy=0.0`，立即停掉矩阵（`pkill -f eval_baseline`），在 timeline 标红"merge 权重可疑"，等下一轮判读——防止坏权重白跑 14 个评测
- 基座基线：AIME25 TIR = 30%（n=30，单卡约 4 分钟）；AIME(24) 同目录 `aime.parquet`
- OPSD 参数（续跑/OPSA 沿用）：16 prompt × 8 rollout，lr 1e-6，kl 0.01（opsa 臂除外），2048 响应/18432 上下文，SAVE/TEST_FREQ=5
- 服务器磁盘 /data2 已 90%（约 550G free）：merge 产物 ~8GB/个，OPSD 今晚还会存 ckpt——每轮看 `df`
- SSH 必须走 `/c/ssh-agentops/config`（中文用户名 home 路径的 ssh 读取 bug 已绕过）
