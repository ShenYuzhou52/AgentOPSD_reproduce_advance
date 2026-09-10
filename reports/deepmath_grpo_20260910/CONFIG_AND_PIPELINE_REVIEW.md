# GRPO 配置与启动链路复盘（2026-09-10）

## 〇、当前状态

- 4 卡正式 run 在 step-1 后被 overlong 熔断器误杀（阈值定低，见 §五）；2 卡重启已按用户要求停止。
- GPU 4/5 空闲，6/7 被其他用户占用。无训练在跑。
- 数据/脚本全部就绪，一条命令即可重启（见 §六）。

## 一、GRPO 配置逐项目的是什么

### A. 数据（这次改动的核心）

| 参数 | 值 | 目的 |
|---|---|---|
| TRAIN_FILE | DeepMath-103K 难度 6.0/6.5 混合（17.8%/82.2%），9300 题 | 把基线 reward 从 0.71 压到 **0.425** |
| VAL_FILE | val100（同分布随机抽）+ AIME24 + AIME25 | 三源分别出指标：val100 测分布内进步，AIME 测泛化 |
| data.shuffle=True, seed=42 | 固定 | 可复现 |

**为什么必须 0.4~0.45**：GRPO 的优势是组内标准化 `A_i = (r_i - mean(group)) / std(group)`。
以 8 条 rollout 一组：如果某题基线 reward≈1，8 条几乎全对 → std=0 → advantage=0 → **该题零梯度**。
上一轮 simplelr_math_35 基线 0.71~0.79，大量组全对，这就是"step 195 增长有限"的主要原因。
基线 0.42 时单题 8 条采样大概率"有对有错"，组内方差最大，梯度信号最密。

**难度怎么选的**：DeepMath 每题带 1~10 难度标注（官方用小模型通过率标定）。我在 4 个难度档
各随机抽 128 题用"正式训练同协议"（thinking + TIR + 15 轮 + 32k 总预算 + greedy）实测：

| 难度 | 5.0 | 5.5 | 6.0 | 6.5 |
|---|---|---|---|---|
| reward | 0.586 | 0.566 | 0.512 | **0.406** |

目标 0.425 落在 6.0 和 6.5 之间，混合权重由线性插值解出：
`w = (0.425 − r_6.5) / (r_6.0 − r_6.5) = (0.425−0.406)/(0.512−0.406) = 17.8%`（取 6.0 的比例）。
实测闭环：训练启动时未训练模型在三源验证集上 mean score 0.444，val100 单独 0.52（n=100 抽样噪声内）。

### B. Rollout / 生成侧

| 参数 | 值 | 目的 |
|---|---|---|
| MAX_EPISODE_RESPONSE_TOKENS | **32768** | **每 episode 总 token 预算**（跨轮共享）。这是本次新增的补丁：agent loop 每轮生成前把 `max_tokens` 截到 `min(请求值, 剩余预算)`，预算耗尽即终局 |
| MAX_RESPONSE_LENGTH | 32768 | verl 层的单轮响应上限。与总预算相等 → 单轮最多吃光全部预算（thinking 可能一轮就很长） |
| MAX_MODEL_LEN | 45056 | vLLM 上下文上限。推导：4096(prompt) + 32768(生成) + 15×~130(observation) + 模板 ≈ 40k，留 5k 余量 |
| MAX_TURNS=15 | python 步数上限 | 用户要求 10~20 取中。实测 overlong episode 平均轮数=1（全是第一轮 thinking 烧穿预算），轮数不是瓶颈 |
| ENABLE_THINKING=true | | 用户要求；也是难度校准协议的一部分 |
| temperature=1.0, top_p=1.0, top_k=-1 | 训练采样 | RL 探索需要；top_k=-1 表示不做 top-k 过滤 |
| rollout.n=8 | GRPO 组大小 | 每题 8 条 rollout 组内比优劣 |
| val_kwargs: temperature=0, n=1 | 验证采样 | 确定性验证，排除采样噪声 |
| GPU_MEMORY_UTILIZATION | 0.55（4卡）/0.50（2卡） | vLLM 预留显存比例。越高 KV 越大 → 并发解码序列越多。4 卡时 0.40→0.55 实测单步 26min→11min |
| AGENT_WORKERS(num_workers) | 16 | agent loop 并发驱动数，决定同时在飞的 episode 数 |
| sandbox_timeout_seconds=5 | | 单次 Python 执行限时（与上一轮一致） |
| max_observation_chars=512 | | 工具输出截断，控制上下文增长 |

**rollout 并行结构**：`tensor_model_parallel_size=1, data_parallel_size=1` 时，
verl 自动起 `num_replicas = world_size / (tp×dp)` 个独立 vLLM 引擎——4 卡=4 副本，负载均衡。
训练阶段引擎休眠（sleep_replicas 释放权重和 KV）， rollout 阶段唤醒，这就是 colocate 模式。

### C. 训练侧（与上一轮保持一致）

| 参数 | 值 | 目的 |
|---|---|---|
| TRAIN_PROMPTS=16 × ROLLOUT_N=8 | 128 episodes/步 | 每步数据量 |
| ppo_mini_batch_size=8, ppo_epochs=1 | | 128 条每步分 16 个 minibatch，单轮更新 |
| ppo_micro_batch_size_per_gpu=1 | | 显存安全：单条 40k token 的序列一次前向 |
| lr=1e-6 | | RL 微调小步长，防止破坏基座 |
| use_kl_loss=True, kl_loss_coef=0.01 | | 约束策略不漂移太远（防止"奖励黑客"式退化） |
| clip_ratio_low/high=0.2 | | PPO 裁剪，限制单步策略更新幅度 |
| entropy_coeff=0.0 | | 不额外加熵正则（KL 已有约束） |
| use_remove_padding=True | | Qwen3.5 的 FSDP/FlashAttention 路径要求 packed（去 padding）批次 |
| strategy=fsdp2, reshard_after_forward=True | | 4B 模型分片训练，省显存 |
| TRAIN_STEPS=200, SAVE_FREQ=20, MAX_CKPTS=3 | | 每 20 步存档留 3 个，支持断点续训 |

### D. 奖励口径（"reward 0.7" 指的是什么）

`score_simpletir_math`：拼接整个 episode 的助手文本和 observation，提取 `\boxed{}`，
用 math_verify 对答案做**二值判定**；若答对但**全程没有实质性工具使用**（纯 CoT 做对），
按 SimpleTIR 规则**减半记 0.5**。所有校准数字（0.71、0.406、0.425）都是这个带减半的分数。

### E. 验证与监控（本次新增）

| 机制 | 值 | 目的 |
|---|---|---|
| TEST_FREQ=20 | 每 20 步 | 用户要求；验证 160 题（100+30+30）本身要 ~15min，不宜更密 |
| 验证指标分组 | `val-core/deepmath_val100|aime24|aime25/reward/mean@1` | verl 按 parquet 的 data_source 字段自动分组 |
| validation_data_dir | `<run>/val_generations/` | **每轮验证全量落盘** `{step}.jsonl`（input/output/gt/score），看"还在不在说人话" |
| log_val_generations=24 | | wandb 上每轮随机 24 条完整表格，不用登服务器 |
| trainer.logger=['console','wandb'] | | 所有训练指标（含 overlong 两条曲线）自动上 wandb |
| monitor_overlong.py | 常驻 | 见 §四.5；≥10% 记告警，失控才熔断 |

## 二、完整启动链路（做了什么、为什么这么排）

```
联网调研 → DeepMath-103K 选型
   ↓ (hf-mirror 下载到服务器)
难度切片 pilot ×4 + DeepScaleR 对照 ×1 + 24k 预算对照 ×1   ——GPU 2-7 六卡并行
   ↓ (eval_baseline.py, 同训练协议 greedy 版)
难度→reward 曲线 → 锚点混合 → build_deepmath_mix.py
   ↓
train 9300 + val100 + AIME24/25（去污染=0 命中，train/val 零重叠）
   ↓
launcher 改造（多 val 集 / wandb / val dump）
   ↓
smoke 2 步（全链路：训练 step + 三源验证 + 落盘）→ 门控通过
   ↓
吞吐调优 smoke（0.55 显存 + 16 workers，1 步，验证不 OOM + 测速）
   ↓
编排器启动 formal 200 步（wandb 已连） + monitor
   ↓
事故：step-1 overlong 38.3% 触发 0.35 单步熔断线 → 被误杀（阈值失误，见 §五）
```

**为什么先 smoke 再 formal**：改了三处高风险点（32k 预算的 token 账本、3 个验证文件、
wandb 后端），2 步冒烟能在 40 分钟内暴露所有链路问题，代价是正式 run 的 1/20。
冒烟门控：初始 reward∈[0.30,0.55]（数据校准 0.425 的宽容区间）、aborted<10%、验证 dump 必须产生。

## 三、关键脚本与 bash 语句讲解

### 1. 六卡并行 pilot（bash 函数 + 后台任务模式）

```bash
launch() {
  name=$1; gpu=$2; data=$3; budget=$4
  out=$ROOT/eval_$name
  mkdir -p $out
  if [ -f "$out/summary.json" ]; then echo "$name already done"; return; fi   # 幂等：重跑跳过已完成的
  CUDA_VISIBLE_DEVICES=$gpu \          # 只让这个进程看见一张卡（vLLM 自动用它）
  VLLM_USE_FLASHINFER_SAMPLER=0 \      # 服务器没 nvcc，绕开 FlashInfer JIT
  PYTHONPATH=/data2/.../AgentOPSD-tir \ # eval 脚本 import 我们的 overlay 模块
  TMPDIR=/data2/... \                  # 关键：系统盘满、/tmp 不可写，必须重定向临时目录
  setsid nohup $PY $EV ... > $out/eval.log 2>&1 < /dev/null &
  echo $! > $out/pid                   # 记 pid 供巡检（注意 setsid 下 $! 是其父，判断存活看日志/GPU 更可靠）
}
launch d5_0_b32k 2 ... 32768   # 函数式调用，一张卡一个任务
```

`setsid nohup cmd > log 2>&1 < /dev/null &` 五件套逐个拆：
- `setsid`：新建会话/进程组，脱离当前终端——SSH 断开（我们这次断了很多次）不影响它；
- `nohup`：免疫挂断信号（双保险）；
- `> log 2>&1`：stdout 和 stderr 都进日志（`2>&1` 必须写在 `> log` 之后）；
- `< /dev/null`：占住 stdin，防止它等待交互输入把任务挂死；
- `&`：放后台。

### 2. build_deepmath_mix.py（校准的核心 30 行）

```python
# 锚点选择：找相邻难度对，使 r_lo >= target >= r_hi（难度越高 reward 越低）
for lo, hi in zip(difficulties[:-1], difficulties[1:]):
    if reward[lo] >= args.target >= reward[hi]:
        anchors = (lo, hi)
# 线性插值解混合权重：w*r_lo + (1-w)*r_hi = target
w = (args.target - reward[hi]) / (reward[lo] - reward[hi])
# 采样：两个池无放回抽满 9400（100 验证 + 9300 训练），rng.permutation 打乱后前 100 条做验证
# → 天然保证 train/val 零重叠、同分布
idx_lo = np.sort(rng.choice(len(pool_lo), size=n_lo, replace=False))
```

去污染：把 AIME24/25 题干做 `re.sub(r'[^a-z0-9]+',' ',q.lower())` 归一化后取前 180 字符做键，
与候选题双向包含匹配（DeepMath 官方已去污染，实测命中 0，这层是防御性的）。

### 3. launcher（run_simpletir_qwen35_4b.sh）里的 bash 技巧

**a) 环境变量默认值模式**——所有可调项同一写法，调用方 `VAR=x bash script.sh` 覆盖，不传用默认：
```bash
TRAIN_STEPS="${TRAIN_STEPS:-100}"
MAX_EPISODE_RESPONSE_TOKENS="${MAX_EPISODE_RESPONSE_TOKENS:-${MAX_RESPONSE_LENGTH}}"
```

**b) 冒号分隔的多验证集拆分**：
```bash
IFS=':' read -r -a VAL_FILE_LIST <<< "${VAL_FILE}"
```
`IFS=':'` 让 read 用冒号切分；`-a` 存入数组；`<<< "..."` 是 here-string（把字符串当 stdin）。
比 for 循环切字符串安全（路径含空格也不碎）。

**c) 拼 hydra 列表字面量**：
```bash
VAL_FILES_HYDRA="[$(printf "'%s'," "${VAL_FILE_LIST[@]}" | sed 's/,$//')]"
```
`printf` 对数组每个元素输出 `'xxx',`，`sed 's/,$//'` 去掉末尾逗号 → `['a','b','c']`。
这个字符串最终作为 hydra 命令行覆盖项 `data.val_files=['a','b','c']` 传给 python 入口。

**d) heredoc 内嵌 python 预检**（防 wandb 交互式登录把训练挂死）：
```bash
if ! "${PYTHON_BIN}" - <<'PYCHECK'
import os, sys
key = os.environ.get("WANDB_API_KEY")
if not key:
    try:
        import wandb; key = wandb.api.api_key
    except Exception:
        key = None
sys.exit(0 if key else 1)
PYCHECK
then
  echo "USE_WANDB=1 but no wandb credential ..." >&2; exit 2
fi
```
`<<'PYCHECK'` 引号很关键：**带引号的定界符不做变量展开**，python 代码里的 `$` 原样保留；
不带引号会被 bash 先展开，容易出隐蔽 bug。

**e) 参数数组 + 逐元素展开**：
```bash
ARGS=( "model_engine=dp" "trainer.total_training_steps=${TRAIN_STEPS}" ... )
"${PYTHON_BIN}" -m integrations.simpletir_qwen35.main_tir "${ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"
```
`"${ARGS[@]}"` 加引号展开时每个元素仍是独立参数（路径带空格不会碎）；`tee -a` 同时上屏和落盘。

### 4. 编排器（run_grpo_formal_20260910.py）的设计

- **阶段状态机**：pilots → build → smoke → formal，每步把决策依据写进 `orchestrator_state.json`；
  崩溃后可用 `--start-phase` 从任意阶段续。
- **原子写状态**（防半截文件）：
  ```python
  tmp = state_path.with_suffix(".tmp"); tmp.write_text(...); tmp.replace(state_path)
  ```
- **启动训练**：`subprocess.Popen([...], start_new_session=True)` —— python 版的 setsid，
  编排器死了训练也活着；返回的 pid 同时是进程组长（PGID），给监控用。
- **从日志抽数**（与监控同一套正则）：
  ```python
  clip_m = re.search(r"response_length/clip_ratio:([-+0-9.eE]+)", line)
  ```
  verl 每个 step 往 train.log 打一行超长指标行，正则按 key 抠值，比解析结构化输出稳。

### 5. monitor_overlong.py（overlong 熔断器）

- **增量读日志**：记住上次读到的字节偏移，`handle.seek(since)` 只读新增部分，`tell()` 更新偏移；
- **连续状态机**：`overlong_streak` 连续计数，低于阈值就清零；
- **熔断**：先 `os.killpg(pid, SIGTERM)` 杀整个进程组（bash→python→ray 全链条），
  等 120 秒不退再 `SIGKILL`；
- 每 60 秒把最近 60 步历史 + 告警写进 `monitor_state.json`，人随时可读。

### 6. SSH 侧惯用法（我这边怎么操作服务器的）

```bash
ssh -F /c/ssh-agentops/config 124.128.251.62 'bash -s' <<'EOF'   # 本地写脚本、远程 bash 整段执行
  ...完整脚本...
EOF
```
`/c/ssh-agentops/` 是绕开 Windows 中文用户路径导致 ssh 崩溃的独立配置目录。
scp 用同一个 `-F`。远程长任务一律 setsid 脱离，SSH 闪断（今天发生了七八次）零影响。

## 四、（补）wandb 与"说人话"检查的具体位置

- 看板：wandb.ai → project `qwen35_simpletir`；指标名见 RUNBOOK §4。
- 完整 response：服务器 `<run>/val_generations/<step>.jsonl`，160 行/轮，
  字段 input/output/gts/score/answer_accuracy/substantive_tool_use。
- 抽查 step-0 的输出：流利数学英语、正常 LaTeX、`\boxed{(1, 1)}` 收尾——基座行为正常。

## 五、事故复盘：熔断阈值误杀

时间线：15:44 正式启动 → 16:09 初始验证（0.444）→ 16:2x step-1 完成（reward 0.477、overlong 38.3%）
→ 16:26 monitor 判定 `single-step overlong 0.383 >= 0.35` → SIGTERM 杀掉整个训练。

失误点：我在定阈值时把"单步灾难线"定为 0.35，但**两轮 smoke 的实测开局水位就是 38~42%**
（难度 0.42 + thinking 的物理长度），数据明明在手，外推阈值时没对齐。属于"自动化熔断阈值
必须从已观测分布推导"的反面教材。已改为：**≥55% 连续 3 步 或 ≥65% 单步才熔断**，
≥10% 只记告警（那是你的目标线，但开局达不到——GRPO 对超长 episode 的 reward≈0 惩罚
会随训练把 overlong 压下来，smoke 里 step-2 已从 38% 降到 22%）。

## 六、下一步选项（无训练在跑，随时可执行）

1. **等 4 卡**（推荐，速度 2×）：GPU 4-7 空出来后一条命令：
   ```bash
   cd /data2/ssd/yixinshen/AgentOPSD-tir && \
   TMPDIR=/data2/ssd/yixinshen/tmp-pilot setsid nohup \
   /data2/.../.venv/bin/python scripts/run_grpo_formal_20260910.py --start-phase formal \
   > /data2/.../formal_20260910/orchestrator.log 2>&1 < /dev/null &
   ```
   （编排器会先等 wandb key、再等 GPU 空闲，然后起训练 + 修好阈值的监控）
2. **2 卡先跑**：同前，环境变量换 `CUDA_VISIBLE_DEVICES=4,5 N_GPUS=2 GPU_MEMORY_UTILIZATION=0.50`，
   吞吐约减半（~22min/步，200 步 ≈ 3.5 天）。
3. 被杀的 4 卡 run 没产生 checkpoint（SAVE_FREQ=20，step-1 不存档），重启=从头；数据 seed 固定，完全可复现。
