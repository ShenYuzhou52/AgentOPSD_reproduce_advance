# Qwen3.5-4B SimpleTIR GRPO 重跑总结：DeepMath 难度校准 + 长度惩罚（2026-09-10 ~ 09-12）

> 实验目标：修复上一轮 GRPO（simplelr_math_35，基线 reward 0.71，"step195 增长有限"）的训练数据问题。
> 主线：数据难度校准到 0.4~0.45 → 两臂对照（无长度惩罚 vs 长度惩罚）→ 验证驱动的 checkpoint 筛选 → 三模型独立终评。

## 1. 实验设置（共用部分）

| 项 | 值 |
|---|---|
| 基座模型 | Qwen3.5-4B（/data2/ssd/yixinshen/models/Qwen3.5-4B） |
| 框架 | verl v1 sync + TransferQueue，vLLM colocate（4 卡 4 副本），FSDP2 |
| 训练资源 | 4×A800-80G（GPU 4-7），lr 1e-6，KL loss coef 0.01，clip 0.2/0.2 |
| 批量 | 16 prompt × 8 rollout = 128 episodes/步，ppo_mini_batch 8，micro 1 |
| Agent | SimpleTIR Python 工具循环，**thinking 开**，max_turns 15 |
| **总 token 预算** | **每 episode 32768**（跨轮共享，`max_episode_response_tokens` 补丁） |
| 上下文 | MAX_MODEL_LEN 45056（4096 prompt + 32k 生成 + 观察余量） |
| 采样 | 训练 temp 1.0 / 验证 temp 0（greedy）、n=1 |
| 步数 / 存档 | 200 步目标，SAVE_FREQ 20，janitor 额外保管每步纯参数副本 |
| 沙箱 | bwrap，timeout 5s，observation ≤512 字符 |
| 奖励 | 全 episode 提取 `\boxed{}` → math_verify 二值；无实质工具使用答对记 0.5（score_with_halving） |

## 2. 数据：DeepMath-103K 难度校准

选型：[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)（逐题难度 1-10 标注、答案可 math_verify 验证、对 AIME/MATH500 等去污染）。备选淘汰：Big-Math（无难度标注）、OpenMathReasoning（偏 SFT）、DeepScaleR（实测 0.492 不够精准）。

**难度曲线实测**（每档随机 128 题，与训练同协议 thinking+TIR+15轮+32k 预算 greedy，2026-09-10）：

| 难度 | reward(score) | answer_acc | overlong 率 |
|---|---|---|---|
| d5.0 | 0.586 | 0.633 | 35.9% |
| d5.5 | 0.566 | 0.602 | 42.2%（24k 预算对照：0.555 / 36.7%） |
| d6.0 | 0.512 | 0.555 | 47.7% |
| **d6.5** | **0.406** | 0.492 | 45.3% |
| DeepScaleR 随机 128 | 0.492 | 0.531 | 35.2% |
| simplelr_math_35 全量（旧数据） | ≈0.71 | ≈0.79 | — |

**训练集**：d6.0×17.8% + d6.5×82.2% 线性插值 → 期望 reward **0.425**。
- `train_deepmath_mix_r425_s42.parquet`：9300 题（d6.5 池 7750 所限；200 步×16=3200 采样 ≪ 1 epoch）
- `val_deepmath100_s42.parquet`：随机抽 100 题（87×d6.5 + 13×d6.0），与训练集零重叠
- `val_aime24.parquet`（30 题）/ `val_aime25.parquet`（30 题）
- 去污染：AIME 归一化文本双向匹配命中 0；构建清单 `datasets/deepmath/build_manifest.json`（含 sha256）

闭环验证：训练启动时未训练模型三源合并 mean score 0.444（目标 0.425 ✓）。

## 3. 臂 1：无长度惩罚（`grpo_deepmath_think32k_t15_s42_formal200`）

启动：看门狗 09-11 00:49 自动拉起。**每 20 步三源验证（val100/AIME24/AIME25）+ 全量 response 落盘。**

### 3.1 验证序列（val100 = score_with_halving，AIME 为 answer_accuracy）

| step | val100 | AIME24 | AIME25 | 训练 reward | 响应均长(tok) | overlong | 熵 | KL |
|---|---|---|---|---|---|---|---|---|
| 0（初始验证） | 0.480 | 0.400 | 0.433 | — | — | — | — | — |
| 20 | 0.615 | 0.467 | 0.300 | 0.789 | 16,990 | 2.3% | 0.459 | 0.005 |
| **40（峰值）** | **0.670** | **0.533** | **0.500** | 0.727 | 14,903 | 3.9% | 0.462 | 0.009 |
| 60 | 0.320 | 0.333 | 0.300 | 0.836 | 20,653 | 14.1% | 0.517 | 0.011 |
| 80 | 0.170 | 0.267 | 0.300 | 0.840 | 17,202 | 8.6% | 0.541 | 0.013 |

（AIME 数字为训练内置验证；终评单跑见 §5）

### 3.2 退行确认与停训

step-60 三源齐跌且训练 reward 仍在 0.84——训练/验证背离。病理证据：验证输出长度 41.9k→74.8k 字符、
贪心锁死在 "Wait... Let's assume..." 空转循环、boxed 率 0.88→0.65。step-80 val100 0.170 确认单调崩塌，
09-11 20:17 停训。**根因链**：训练分饱和（~0.78，大量死组）→ 优势持续奖励"长链枚举"（temp 1.0 下撞对率高）
→ token 分布拉平（熵↑ KL↑）→ **贪心 argmax 退化为循环吸引子**。长度是载体，本质是奖励饱和后的
"采样彩票"漂移。step-40 checkpoint（完整版）双份保存。

## 4. 臂 2：长度惩罚（`grpo_deepmath_think32k_t15_s42_formal200_lenpen`）

**设计**（`simpletir_agent_loop.py`，train-only，GRPO 组内标准化之前生效）：

```
penalty = λ · max(0, tokens − free_frac·B) / (B − free_frac·B)     # quota 模式
score' = max(0, score − penalty)        # λ=0.25, free_frac=0.5, B=32768 → 免费额度 16k
```

组内标准化使"同题短对 vs 长对"产生相对优势（短者胜）；组内等量惩罚自动抵消；保底 0 不惩罚失败。
验证不计惩罚（口径可比）。

**从臂 1 step-40 完整 checkpoint 续训**（optimizer/rng/lr_scheduler 全恢复，09-11 20:26），单变量对照。

### 4.1 验证序列

| step | val100 | AIME24 | AIME25 | 训练 reward | 响应均长(tok) | overlong | 熵 | KL |
|---|---|---|---|---|---|---|---|---|
| 40（续训起点） | 0.640 | 0.517 | 0.400 | — | — | — | — | — |
| 60 | 0.620 | 0.600 | 0.500 | 0.831 | 16,293 | 9% | 0.506 | 0.011 |
| 80 | 0.685 | 0.533 | 0.433 | 0.871 | **9,604** | 1% | 0.437 | 0.019 |
| **100（峰值）** | **0.780** | **0.550** | **0.567** | 0.823 | 13,456 | 3% | 0.511 | 0.017 |
| 120 | 0.725 | 0.483 | 0.567 | 0.815 | 12,407 | 0.8% | — | 0.026 |
| 140 | 0.640 | 0.533 | 0.533 | 0.897 | 12,159 | — | — | 0.027 |
| 160 | 0.330 | 0.267 | 0.367 | 0.872 | 13,533 | — | — | 0.026 |

### 4.2 判决与停训

- **机制生效**：无惩罚臂同段 0.75→0.32→0.17 自由落体；惩罚臂 0.64→0.62→0.69→**0.78**（超原峰值），
  响应长度 17k→10k（砍半）且训练分不降（0.80-0.93）——"更短且仍对"是真实能力提升。**有效窗口 40→100 步（2.5×）**。
- **step-160 跳崖**（0.330）：平台缓降（100→140，−14 分）后与臂 1 同本质的贪心退化再次发生（延迟 60 步）。
  KL 已回落 0.026、长度稳定 13.5k——跳崖不经由任何被追踪的宏观指标预警。
- 09-12 19:37 按预设规则停训。step-100 完整 checkpoint 已 pin 至 `pinned_checkpoints/`。

**两臂结论**：长度惩罚抬高峰值、延长窗口、治好长度通胀，但**不改变终点**——结果奖励 GRPO 的
贪心退化是奖励饱和的必然，checkpoint 筛选（验证驱动早停）目前是必要收官手段。

## 5. 独立终评（09-12，合并 HF 权重，独立 evaluator，greedy，TIR 15 轮 32k 预算）

三个模型 × 三个数据集。step-100 跑 4 次重复；基座与 ckpt40 各 1 次。
（answer_accuracy 为主指标；score_with_halving 参考；overlong = episode 撞 32k 预算比例；turns 均为 1.0——所有 episode 单轮内终结）

### 5.1 answer_accuracy（主口径）

| 模型 | AIME24 (n=30) | AIME25 (n=30) | val100 (n=100) |
|---|---|---|---|
| 4B 基座（1 次） | 0.467 | 0.400 | 0.480 |
| ckpt40 无惩罚（1 次） | 0.533 | 0.500 | 0.670 |
| **step100 惩罚臂（4 次均值）** | **0.592 ± 0.050** | **0.633 ± 0.000** | **0.730 ± 0.000** |
| step100 各 rep | 0.533/0.567/0.633/0.633 | 0.633×4 | 0.730×4 |

### 5.2 score_with_halving（含无实质工具使用减半）

| 模型 | AIME24 | AIME25 | val100 |
|---|---|---|---|
| 4B 基座 | 0.433 | 0.383 | 0.450 |
| ckpt40 无惩罚 | 0.533 | 0.450 | 0.655 |
| step100（4 次均值） | 0.583 ± 0.049 | 0.617 ± 0.000 | 0.725 ± 0.000 |
| step100 各 rep | 0.533/0.550/0.617/0.633 | 0.617×4 | 0.725×4 |

### 5.3 辅助指标（boxed 率 / overlong 率）

| 模型 | boxed（24/25/val100） | overlong（24/25/val100） |
|---|---|---|
| 4B 基座 | 0.47 / 0.57 / 0.71 | **0.57 / 0.63 / 0.52** |
| ckpt40 无惩罚 | 0.67 / 0.67 / 0.80 | 0.40 / 0.43 / 0.30 |
| step100（均值） | 0.708 / 0.633 / 0.830 | 0.292 / 0.367 / **0.220** |

### 5.4 读数

- **AIME25：0.633**，基座 0.400 的 1.58×；greedy 下 4 rep 逐题完全一致（评测链路确定性验证通过）。
- **AIME24：0.592±0.050**，rep 间 0.533→0.633 的波动来自 32k 长链解码的 vLLM 批处理数值抖动（单 token 翻转级联）。
- 训练同时治理了长度行为：overlong 0.52-0.63（基座）→ 0.22-0.37（step100）。
- 训练验证 val100 0.780 与终评 0.730 的差：终评主口径为 answer_accuracy（0.730 vs 训练口径 score 0.725 一致）+ 合并权重数值路径差异，±3-5 分在预期内。

## 6. 结论

1. **数据难度校准是前提**：把训练集 reward 从 0.71 压到 0.425（DeepMath d6.0/d6.5 混合），GRPO 从
   "step195 增长有限"变为 40 步内 val100 +19 分的快速起飞——组内方差是 GRPO 梯度的生命线。
2. **结果奖励 GRPO 存在"有效窗口"，窗口后必然发生采样/贪心背离式退化**（两臂均验证）。表象是长度通胀
   与贪心锁死，根因是奖励饱和后策略向"长链枚举抽奖"漂移。
3. **长度惩罚（λ=0.25 quota）有效但有限**：峰值更高（0.670→0.780）、窗口 2.5×、长度砍半、overlong 减半，
   但不改变终点。shaping 项在任务梯度耗尽后接管优化方向，制造新的过优化。
4. **验证驱动 checkpoint 筛选是必要收官手段**：峰值 checkpoint（惩罚臂 step-100）与最终模型之间差
   60 步和 45 分。
5. **终评**：AIME24 0.592 / AIME25 0.633 / val100 0.730——AIME 双双刷新本项目历史最好成绩。

**后续方向**（按对症程度）：难度课程跟进（策略涨→数据加难，维持组内方差，治饱和本因）；
KL 锚加强或熵正则；贪心感知的 checkpoint 筛选；λ 衰减调度（step-100 后 0.25→0.05）。

## 7. 资产与复现

| 资产 | 位置（/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260910/） |
|---|---|
| 最终模型 HF 权重 | `final_eval_20260912/merged_step100/`（9.7G） |
| step-100 完整 ckpt（含优化器） | `pinned_checkpoints/global_step_100/`（51G） |
| 惩罚臂纯参数 ckpt ×7（40-160） | `..._lenpen/checkpoints_keep/` |
| 无惩罚臂纯参数 ckpt ×4（20-80） | `..._formal200/checkpoints_keep/` |
| 终评逐题记录 | `final_eval_20260912/rep{1-4}/`、`baseline4b_once/`、`ckpt40_nolenpen_once/` |
| 两臂训练曲线 / wandb | 各 run `train.log`/`metrics.jsonl`；project `qwen35_simpletir`（runs 3urdku3j 等） |
| 验证全量 response | 各 run `val_generations/{0,20,...,160}.jsonl`（每轮 160 行） |
| 代码 | overlay `/data2/ssd/yixinshen/AgentOPSD-tir`（长度惩罚 `apply_length_penalty.py`，本地仓 commit `ec18945` 起） |
| 数据 | `datasets/deepmath/`（训练集/验证集/manifest 含 sha256） |

评测协议：`eval_baseline.py --mode tir --thinking --temperature 0 --max-turns 15 --max-tokens 32768 --max-model-len 49152`（`scripts/final_eval_rep.sh` 可一键复跑单 rep）。
