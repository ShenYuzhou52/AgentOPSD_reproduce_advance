# Qwen3.5-4B × SimpleTIR 三组消融（GRPO / OPSD / AgentOPSD）文件地图

日期：2026-09-07。服务器 `yixinshen@124.128.251.62:16022`（8×A800 80GB）。
本文件回答两个问题：**服务器上哪些文件与本实验相关**，以及**当前预跑状态与已修复问题**。

## 一、实验定义

同一 Qwen3.5-4B 初始化、同一 `simplelr_math_35` 训练集、同组采样（n=8）、
同一提示词与 5 轮工具预算、同一二值奖励与验证节奏、同一种子，仅切换训练信号：

| 组名 (`simpletir.method`) | 策略梯度损失 | Teacher 信号 |
|---|---|---|
| `grpo` | GRPO | 无 |
| `opsd_author_code` | 关闭（`sdar_coef` 门控自蒸馏替代） | 置信度门控自蒸馏（作者公开代码路径） |
| `agentopsd` | GRPO | 含金答案的私有 teacher 前向，仅用于有界的回合级优势重塑 |

奖励语义（与上游 SimpleTIR 一致）：从**真实沙箱 stdout** 中提取 `\boxed{...}`，
经 `math_verify` 判等得二值分；无实质工具使用的正确回答记半分。标准答案
`reward_model.ground_truth` 只进入终局打分与 AgentOPSD 私有 teacher 前向，
永不进入学生可见文本、观察、`extra_fields`、指标或日志。

## 二、服务器目录地图（`/data2/ssd/yixinshen/`）

### 直接相关（本实验的全部代码与数据）

| 路径 | 内容 |
|---|---|
| `AgentOPSD-tir/` | 本实验 overlay 仓库（git，基线提交 `7be3026`），详见第三节 |
| `benchmarks/verl-qwen35-base/` | 钉死的 Verl 基座（提交 `5b79827`，含 `.venv` 训练环境与 `platform_cuda.py` 等补丁） |
| `benchmarks/SimpleTIR/` | 上游 SimpleTIR 参考实现（`recipe/`、`sandbox/`、`verl/`）与数据集 `datasets/simplelr_math_35/` |
| `models/Qwen3.5-4B/` | 基座模型权重（Qwen3_5 ForConditionalGeneration，bfloat16） |
| `experiments/qwen35-simpletir/` | 全部运行产物：`precheck_20260907/` 下按修复阶段分子目录，含 console.log、metrics.jsonl、checkpoints、`val4.parquet`（4 题开发验证集） |
| `r/<hash>`、`t/<hash>` | 每实验独立 Ray 临时目录与 TMPDIR（避免与既有任务冲突） |
| `sb_venv/` | **本次新建**：沙箱专用科学计算 venv（见第五节） |

### 同仓库但属于上一阶段实验（ALFWorld / SDAR，本实验不触碰）

`AgentOPSD/`（ALFWorld 版工作副本）、`agentopsd.bundle`（git bundle 备份）、
`initial_RL/`、`rsp-opd/`、`sft/`、`GeoGuessant_train/`、`benchmarks/` 其余子目录。

### 其余为环境/缓存（无关）

`cuda-12.8-*`、`micromamba/`、`hf_cache/`、`train_tmp/`、`wandb/`、安装包与历史调试输出等。

## 三、`AgentOPSD-tir` 仓库内相关文件

### `integrations/simpletir_qwen35/` —— Qwen3.5/SimpleTIR 专用入口（核心）

| 文件 | 作用 |
|---|---|
| `main_tir.py` | Hydra 入口，组装 Verl `ppo_trainer` 基配置 + 本实验覆盖项；先注册自定义 trainer 再启动 Ray |
| `trainer.py` | `SimpleTIRTrainer(PPOTrainerSync)`：三组共用的 V1 训练钩子。校验 turn 键/元数据一致、写入 teacher log-prob、AgentOPSD 优势重塑、批平衡防重复守卫、`simpletir/*` 与 `agentopsd/*` 指标、metrics.jsonl 落盘；OPSD 组在此挂 `sdar_only_loss` |
| `simpletir_agent_loop.py` | `simpletir_python` agent loop：构造学生提示→逐轮生成→解析代码围栏→沙箱执行→观察回填→终局打分。AgentOPSD 私有 teacher 前向（含金答案的 system 注入 + prompt_logprobs 对齐校验）也在此 |
| `simpletir_sandbox.py` | bwrap + systemd-run 无特权沙箱：网络/文件系统隔离、内存/任务/CPU 限额、超时分类。**本次修复点**（见第五节） |
| `trajectory.py` | 纯函数：代码围栏与 `\boxed{}` 解析、`final_answer` 助手注入、观察格式化、`score_simpletir_math` 二值奖励（math_verify，**线程感知** + Python 风格 sympify 回退，见第五节） |
| `prompting.py` | 上游 SimpleTIR 工具使用契约前缀（逐字复制），只注入一次 |
| `opsd_loss.py` | `sdar_only_loss`：作者公开代码的门控自蒸馏损失（脱离 SDAR 依赖移植） |
| `agent_loop_manager.py` | TransferQueue worker 适配：向 agent loop 注入 `is_validation` 位，保证 teacher 只在训练时运行；不改行数 |
| `data_contract.py` | 训练/验证 parquet 的 schema 校验 |
| `agent_loop_config.yaml` | 注册 `simpletir_python` loop 的 Hydra 配置 |
| `test_*.py`（9 个） | 沙箱安全、提示词、轨迹解析、teacher 对齐（真实返回结构回归）、传输层、OPSD 反传、agent loop 辅助函数的单测，共 28 项 |
| `inspect_dataset_schema.py`、`remote_preflight.py` | 数据检查与远端预检工具 |

### `agentopsd/` —— 方法核心（框架无关）

| 文件 | 作用 |
|---|---|
| `credit.py` | `reshape_advantages`：按论文 Algorithm 1 将序列级 GRPO 优势重塑为回合级（token gap → 回合信用 → 乘子裁剪 → 重归一化） |
| `trainer/monitor.py` | 训练监控指标聚合与 JSONL 追加 |
| `trainer/main_agentopsd.py`、`patch.py` | ALFWorld/SDAR 阶段的入口与补丁（本实验不经过，但为同方法实现） |

### `scripts/` —— 启动与运维

| 文件 | 作用 |
|---|---|
| `run_simpletir_qwen35_4b.sh` | **三组统一入口**：`METHOD=grpo|opsd_author_code|agentopsd`；全部状态放 /data2，2~8 卡可配，防 teacher 信息泄漏（rollout dump 关闭） |
| `train.sh`、`smoke_test.sh`、`eval.sh`、`run_stability_matrix.sh` 等 | ALFWorld 阶段脚本（本实验不用） |
| `configs/agentopsd.env`、`lib.sh`、`git_sync.sh` 等 | 辅助配置 |

### 其他

`patches/sdar-reproduction-fixes.patch`（上游 SDAR 复现修复，已含在构建里）、
`docs/01–04`（ALFWorld 阶段设计文档）、`pyproject.toml`/`uv.lock`（依赖钉版）。

## 四、外部依赖关系

- **Verl 基座**：`PYTHONPATH` 为 `AgentOPSD-tir:verl-qwen35-base`，overlay 只新增不改基座文件；教师 log-prob 的单例维度（`[[id],...]`）等行为以该提交为准。
- **数据**：`simplelr_math_35/train.parquet`（训练）+ `test_fixed100_s42.parquet`（开发验证）；预跑用 4 题子集 `val4.parquet`。终评用不相交的 `deepscaler/aime`、`aime25`。
- **模型**：`Qwen3.5-4B`，`enable_thinking=false`（首动作必须是围栏代码，1k 动作预算不被思考块吃掉）。

## 五、预跑问题与修复记录（2026-09-07）

### 已修复（上一会话，代码在仓库中）

1. **Teacher token 对齐误报**：Verl 把 `prompt_ids` 返回成 `[[id],…]`，旧代码直接与 `[id,…]` 比较导致首批 rollout 失败 → 单例解包 + 保留全部严格校验。
2. **OPSD teacher 分数传输**：`TransferQueue` 字段元数据未随新字段更新，actor 看不到 `teacher_response_log_probs` → 修复元数据写入。
3. **AgentOPSD nested/padded mask**：稠密 mask 误传给要求 nested 的 `to_padded_tensor`/`response_to_nested` → 修复转换路径。

### 本次修复（零奖励的三层根因）

上一轮三组预跑全部 `critic/score/mean=0`、`answer_accuracy=0`、GRPO
`grad_norm=0`。逐层排查（沙箱直测 → 学生侧转储分析 → 线程复现）确认了
**三个互相叠加的根因**，任意一个都足以让奖励恒零：

1. **沙箱缺科学计算库**：沙箱用系统 `/usr/bin/python3`（3.10.12，仅标准库），
   模型的数学代码 `import sympy/numpy/scipy` 全部 `ModuleNotFoundError`。
   **修复**：新建 `/data2/ssd/yixinshen/sb_venv`（numpy 2.2.6 / sympy 1.14.0 /
   scipy 1.15.3），`simpletir_sandbox.py` 以只读方式挂载到沙箱内 `/opt/sb_venv`
   并切换解释器；`/data2` 对模型程序保持不可见；venv 缺失时自动回退。
2. **OpenBLAS 线程崩溃**：沙箱内 `sched_getaffinity` 仍见 112 CPU，OpenBLAS
   尝试开 64 线程被 systemd `TasksMax=32` 拒绝 → numpy 段错误（rc=139）。
   **修复**：bwrap 环境固定 `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`（CPUQuota 本就 1 核）。
3. **math_verify 在 worker 线程中静默失败（最终根因）**：agent loop 在 worker
   线程运行，math_verify 的 `SIGALRM` timeout 在非主线程抛 `ValueError`，被
   打分器的异常保护吞掉——**即使答案完全正确也恒判 0**（线程内复现：同一输入
   主线程 1.0、worker 线程 0.0）。上游 SimpleTIR 为此自写了
   `verify_without_timeout`。**修复**：`trajectory.py` 线程感知——主线程保持
   原 signal 超时，worker 线程走 math_verify 官方线程模式
   （`parsing_timeout=None` / `timeout_seconds=None`）。
4. **Python 风格答案的系统性误判**：`final_answer(str(sympy_expr))` 打印
   `5*x**2/2` 这类 Python 语法，math_verify 的 latex 优先解析将其保留为未解析
   字符串，与 sympy 形式的金标准比较恒 False（符号假设也不同：金标准
   `real=True` 且未展平）。**修复**：判 False 后对仍为字符串的预测做
   `sympify` + 按金标准符号假设对齐 + `sympy_expr_eq`（上游同款 grader、同参数）
   的回退比较；只补齐等价表达式的可判分性，不放宽错误答案。
5. **1024 响应预算截断级联**（参数修复）：Qwen3.5-4B 非思考模式写详细 LaTeX
   推理，1024 token 处截断 → 代码围栏未闭合 → 解析为 void turn → episode 终止。
   转储显示 74% episode 因此在第一轮就死亡、仅 27% 曾产出代码。**修复**：
   三组统一 `MAX_RESPONSE_LENGTH=2048`（`MAX_MODEL_LEN=18432`）——历史上 2048
   也是零奖励，但当时沙箱与打分器都是坏的，不构成对 2048 的反证。修复后
   代码产出率 0.27→0.83，void 率 0.74→0.19。
6. **新增学生侧调试转储**（`SIMPLETIR_DEBUG_DUMP=<dir>` 开启）：只转储学生
   可见内容（问题、各轮文本、观察、标量奖励诊断），不含 ground_truth，默认关闭。

修复后仓库单测 **31 项全部通过**（含线程安全打分、Python 风格等价性、
沙箱 venv/只读/数据盘隔离等新回归）。

## 六、预跑结果（final 轮，2026-09-07）

目录 `precheck_20260907/final_{grpo,agentopsd,opsd}/`，各 2 GPU（0,1 / 2,3 / 6,7）、
2 step、8 prompt × 8 rollout、响应 2048、step2 验证 + checkpoint，全部退出码 0。

| 指标 | GRPO | OPSD(作者代码) | AgentOPSD |
|---|---|---|---|
| `critic/score/mean` step1→step2 | 0.038 → **0.159** | 0.064 → **0.183** | 0.012 → **0.146** |
| `actor/grad_norm` (step2) | **0.662** | 0.014（仅蒸馏损失，PG 按设计关闭） | **1.169** |
| 优势范围 (step2) | ±1.62 | [-1.21, +1.62] | [-0.88, +2.47] |
| `answer_accuracy` (step2 rollout) | 0.159 | 0.183 | 0.153 |
| `sandbox_ok_ratio` (step2) | 0.95 | 0.77 | 0.69 |
| 验证 reward@1 (4 题, step2) | 0.50 | 0.00 | **0.75** |
| checkpoint global_step_2 | 51 GB | 51 GB | 51 GB |

方法专属信号验证：

- **AgentOPSD**：`teacher_forward_applied=1.0`、`reshape_applied=1.0`、
  `group_success_mean=0.172`、**`group_success_mixed_ratio=0.625`**（62.5% 的组
  内有成功有失败——优势重塑有真实的信用分配对象）、`credit_abs_mean=0.146`、
  `pivotal_turn_ratio=0.028`、multiplier ∈ [0.80, 1.20]、`credit_dead=0`。
- **OPSD**：`teacher_forward_applied=1.0`，梯度来自门控自蒸馏损失；2 step 内
  验证无提升属预期（蒸馏信号更新慢），学习信号以 rollout 奖励上升为准。
- **GRPO**：纯任务奖励驱动，奖励逐 step 上升、梯度非零、验证过半。

**结论：预跑通过**——三组链路完整（rollout → 工具 → 奖励 → 优势 → 更新 →
验证 → checkpoint），奖励与梯度均非零，方法分支各自激活，消融对比条件成立。
注意 2 step 的数值只用于链路验证，不是方法结论。

## 七、TIR-Bench baseline（2026-09-07，正式实验前的门槛验证）

官方参考（Qwen3.5-4B 模型卡 "Tool Calling" 分区，`with CI / without CI` = 带/不带
Code Interpreter）：**with CI 38.9 / without CI 29.9**。评测集为
`simplelr_math_35/test.parquet`（500 题），按用户要求用种子 42 的固定 100 题抽样
（即项目现成的 `test_fixed100_s42.parquet`，与全量 subject 分布成比例）。
脚本：`scripts/eval_baseline.py`（`--mode cot` 单轮纯推理 / `--mode tir` 复用训练
同款工具循环；温度 0、非思考模式；需要 `source env.sh` 并把 verl `.venv/bin`
加入 PATH 以提供 nvcc/ninja）。

| 设定 | answer_accuracy | 备注 |
|---|---|---|
| **without CI**（cot，用户门槛 ≥25%） | **81%** | 85% 产出 boxed；15 个错误全部是 3072 token 截断，无打分错误（人工抽查核对） |
| with CI（tir 工具循环，修正打分后） | **78%** | boxed 83%；含半分规则得分 74.5% |
| with CI（修正前，stdout-only 错误打分） | 11% | 见下方奖励修正 |

**门槛大幅通过**。两点说明：

1. 官方 29.9/38.9 来自其多模态 TIR-Bench，难度高于本纯文本数学集；本集对
   Qwen3.5-4B 的纯推理已近天花板（81%），故 with CI 无增益（78%）符合预期——
   工具收益要在更难题目（AIME）与 RL 训练中体现。
2. **顺带发现并修正了第四个 bug（奖励来源错误）**：overlay 原实现只从沙箱
   stdout 提取答案，但上游 `simplelr_math_35` 走 `hf_math_verify.compute_score`，
   `extract_solution` 从**完整多轮文本**（各 assistant 轮 + 观察）提取最后一个
   `\boxed{}`；只有 LeetCode 类数据才用"代码 stdout 精确匹配"。模型常正确执行
   工具后在**文本**里写 boxed（temp 0 下 89% 的正确轨迹如此），stdout-only 把
   它们全部判 0。已改为全文打分（`simpletir_agent_loop.py` 与评测脚本同步修改），
   半分规则与防泄漏边界不变。

修正奖励后的 2-step GRPO 预跑（`precheck_20260907/rewardfix_grpo/`，GPU 1,2）：
`critic/score/mean` 0.565→0.571（修正前 0.038→0.159）、`grad_norm` 0.76/0.65、
优势 [-2.47, +1.62]、`sandbox_ok_ratio` 0.93、验证 reward@1 0.75、checkpoint
正常保存。**训练信号比修正前丰富约 15 倍，正式启动实验的条件全部满足。**

## 八、遗留观察（非阻塞）

- 仍有 33–44% 回合撞 2048 上限（`clip_ratio`），cot 模式 15% 撞 3072；随训练
  改善，如需调整须三组同参。
- OPSD 验证集 4 题 0 分：温度 0 确定性评估 + 仅 2 step 蒸馏，不构成异常。
- 沙箱超时 5s 下 sympy 执行 0.25s，余量充足；未观察到超时。
- 正式实验建议沿用 `MAX_RESPONSE_LENGTH=2048`；baseline 脚本与结果在
  `experiments/qwen35-simpletir/baseline_tirbench_20260907/`。

