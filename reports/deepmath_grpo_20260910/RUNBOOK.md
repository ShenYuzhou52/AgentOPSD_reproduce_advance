# 正式 GRPO 重跑（DeepMath 混合难度训练集）— 2026-09-10 运行手册

**当前状态（15:55 更新）**：正式 200 步训练已启动（GPU 4-7），wandb 实时看板：
<https://wandb.ai/17621741876-tsinghua-university/qwen35_simpletir/runs/6peo8dzi>

## 1. 训练集更换依据（为什么是 DeepMath-103K）

上一轮 GRPO 训练增长有限的根因之一：`simplelr_math_35` 对 Qwen3.5-4B(thinking+TIR, 15轮, 32k总预算) 太易，
基线 reward ≈ 0.71，rollout 大多全对 → GRPO 组内优势趋零。

联网调研候选：DeepMath-103K（逐题难度 3–9 标注、答案可 math_verify 验证、对 AIME/MATH500 等做过去污染，
[HF](https://huggingface.co/datasets/zwhe99/DeepMath-103K)、[arXiv:2504.11456](https://arxiv.org/abs/2504.11456)）、
Big-Math-RL-Verified（250K 无难度标注）、OpenMathReasoning（306K AoPS 偏易分层）、DeepScaleR-Preview（服务器已有 40K）。
选 DeepMath：唯一带细粒度难度标注，可精确配平到目标 reward。

### 实测难度曲线（Qwen3.5-4B，thinking，TIR，15 轮，greedy，总预算 32768，每档 n=128）

| 难度 | reward(score) | answer_acc | overlong 率 |
|---|---|---|---|
| 5.0 | 0.586 | 0.633 | 35.9% |
| 5.5 | 0.566 | 0.602 | 42.2% |
| 6.0 | 0.512 | 0.555 | 47.7% |
| **6.5** | **0.406** | 0.492 | 45.3% |
| DeepScaleR 随机 | 0.492 | 0.531 | 35.2% |
| （对照 simplelr_math_35） | ≈0.71 | ≈0.79 | — |

### 最终训练集：d6.0 × 17.8% + d6.5 × 82.2% → 期望 reward 0.425

- 训练集：`/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepmath/train_deepmath_mix_r425_s42.parquet`（9300 题，
  d6.5 池 7750 题所限；200 步 × 16 prompt = 3200 次采样仍远小于一个 epoch）
- 验证集（随机抽取，与训练集零重叠）：`val_deepmath100_s42.parquet`（100 题，难度构成 87×d6.5 + 13×d6.0）
- AIME24：`val_aime24.parquet`（30 题）；AIME25：`val_aime25.parquet`（30 题）
- 去污染：AIME24/25 归一化文本双向匹配，命中 0 题
- 全部数字见 `datasets/deepmath/build_manifest.json`（含 sha256）

## 2. 正式训练配置（相对上一轮的全部变化）

| 项 | 上一轮 | 本轮 | 说明 |
|---|---|---|---|
| 训练集 | simplelr_math_35（reward 0.71） | DeepMath 混合（reward≈0.425） | 见上 |
| 总 token 预算 | 每轮 2048 | **每 episode 总计 32768**（`simpletir.max_episode_response_tokens`） | 单轮上限=剩余预算 |
| thinking | 关 | **开**（`ENABLE_THINKING=true`） | |
| Python 步数 | 5 | **15**（`MAX_TURNS=15`） | 在要求 10~20 内；overlong 全部发生在第 1 轮超长思考，轮数不是瓶颈 |
| 验证频率 | 5 步 | **20 步**（`TEST_FREQ=20`） | |
| 验证集 | fixed100 | **val100(deepmath) + AIME24 + AIME25** 三集分别出指标 | `data_source` 分组 |
| wandb | 无 | **开**（project `qwen35_simpletir`） | 需先部署 api key（见 §4） |
| 验证 response 留存 | 无 | **每轮验证全量落盘** `<run>/val_generations/<step>.jsonl`（input/output/gt/score） | +wandb 表格随机 24 条 |
| overlong 监控 | 无 | `scripts/monitor_overlong.py` 常驻：≥10% 告警、≥25% 连续 3 步或 ≥35% 单步自动停训 | 见 §5 |
| MAX_MODEL_LEN | 18432 | 45056 | 4096 prompt + 32768 生成 + 观察余量 |
| rollout 吞吐 | 0.40 显存 / 4 workers | **0.55 显存 / 16 workers** | 单步 26min → ~11min（2.2×），200 步约 40h |
| 其余 | — | 不变 | 4×A800(4,5,6,7)、16 prompt × 8 rollout、lr 1e-6、KL 0.01、200 步、seed 42 |

### smoke 实测（正式启动前全链路验证，全部通过）

- step-1 训练（temp 1.0）：reward **0.473**、overlong 38.3%、aborted 0、episode 平均 25,280 tokens
- 三源验证一次到位：deepmath_val100 reward 0.48 / AIME24 0.45 / AIME25 0.433（各 30/30/100 题）
- `val_generations/2.jsonl` 160 行完整落盘（input/output/gt/score/answer_accuracy），抽样输出为流利
  数学推理 + `\boxed{}` 收尾（"还在说人话" ✓）

## 3. 编排与状态（服务器）

- 编排器：`/data2/ssd/yixinshen/AgentOPSD-tir/scripts/run_grpo_formal_20260910.py`
  阶段：pilots → build → **smoke(2步)** → **formal(200步)**；状态文件
  `/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/orchestrator_state.json`
- smoke 通过条件：初始 reward∈[0.30,0.55]、clip/overlong<0.45（灾难线）、验证三集产出指标、
  `val_generations/*.jsonl` 落盘成功
- 正式 run 目录：`formal_20260910/grpo_deepmath_think32k_t15_s42_formal200/`
  （train.log、metrics.jsonl、checkpoints 每 20 步、保留 3 个、断点续训 RESUME_MODE=auto）

## 4. wandb（已就绪）

服务器 `~/.netrc` 已有有效凭证，`trainer.logger=['console','wandb']` 已生效，
run 名 `grpo_deepmath_think32k_t15_s42_formal200`，project `qwen35_simpletir`。
若需换账号：
```bash
ssh -p 16022 yixinshen@124.128.251.62
/data2/ssd/yixinshen/benchmarks/verl-qwen35-base/.venv/bin/python -m wandb login <新KEY>
```

关键 wandb 指标名：
- `critic/score/mean`（训练 reward）、`val-core/deepmath_val100|aime24|aime25/reward/mean@1`
- `simpletir/episode_overlong_ratio`、`response_length/clip_ratio`（overlong 两条曲线，预期随训练下降）
- `response_length/max`、`actor/grad_norm`、`actor/kl_loss`
- 验证表格（每轮随机 24 条完整 input/output/score）

## 5. overlong 的现实情况与监控口径（重要）

- 实测：该模型在 0.4~0.45 难度上，**初始 overlong（单轮思考烧穿 32k 预算）≈ 45%**，
  与预算/轮数无关（overlong episode 平均轮数=1，thinking 本身停不下来）。
  用户目标 ≤10% 在"reward 0.4~0.45 + 32k 预算"约束下**开局不可达**，已如实放宽告警线：
  ≥10% 记告警、≥25%×3步 或 ≥35%单步 才熔断。
- GRPO 天然压制 overlong：超长截断 episode reward≈0，持续负优势会训练模型"早写代码、少空想"，
  预期 overlong 曲线随步数下降（wandb 上看 `simpletir/episode_overlong_ratio` 与
  `response_length/clip_ratio` 两条曲线）。
- 线上实时查看：`formal_20260910/grpo_deepmath_think32k_t15_s42_formal200/monitor_state.json`
  （每 60 秒刷新，含最近 60 步历史与告警列表）。

## 6. 日常巡检命令

```bash
R=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910
cat $R/orchestrator_state.json | head -20          # 编排阶段
cat $R/grpo_deepmath_think32k_t15_s42_formal200/monitor_state.json | head -30   # overlong/告警
ls $R/grpo_deepmath_think32k_t15_s42_formal200/val_generations/   # 每次验证的完整 response
tail -2 $R/grpo_deepmath_think32k_t15_s42_formal200/train.log     # 最新步指标
```

## 7. 已知运维事实

- 服务器系统盘 `/` 100% 满 → `/tmp` 不可写（昨晚 rerun 基线评测批量离奇死亡的根因）。
  所有新进程已显式 `TMPDIR=/data2/...`；训练 launcher 自带 /data2 tmp，不受影响。
  根治需要 root 清理系统盘。
- OPSD/AgentOPSD 两目录旧 checkpoint 已清理（各留 `global_step_200`，释放约 640G，/data2 现 62%）。
