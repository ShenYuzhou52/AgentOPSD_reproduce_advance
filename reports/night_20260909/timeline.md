# 夜间值守时间线 2026-09-08 → 09-09

## 准备阶段（22:00-23:00，交互模式）

- 22:05 摸底：GRPO formal200 ✅（200/200）、AgentOPSD formal200 ✅（200/200）、OPSD formal200 🔄 ~165/200（139 s/it，预计 23:35 完成，GPU 0-3）。GPU 4-7 为其他任务（PID 2038123+），禁碰。/data2 磁盘 90%（554G free）。
- 22:08 修复本机 ssh：Git Bash ssh 读不了中文用户名 home 路径，自包含配置落到 `/c/ssh-agentops/`，免密非交互登录验证通过。
- 22:15 部署夜间脚本到服务器 `$NIGHT=/data2/.../formal_20260907/night_20260909/`：status_snapshot / merge_one / eval_ckpt / auto_resume_opsd / night_chain（已挂后台，等 OPSD 退出后自动 merge+评测）。
- 22:20 发现 checkpoint 两种格式：grpo_200、agentopsd_195/200 有 `actor/` 分片（可 merge）；grpo_155、agentopsd_100/150 是 `data.pt` 单文件（今晚跳过，晨报建议白天转格式）。
- 22:22-22:33 merger 三轮补丁（`model_merger_patched.py`）：① transformers 5.9 无 AutoModelForVision2Seq → 回退 AutoModelForImageTextToText；② megatron 导入失败 → try/except 回退内联路径函数；③ `--hf_model_path` 必填、无 `merge` 子命令。4 个 ckpt 合并成功（grpo_200/195、agentopsd_200/195，各 ~10GB 单文件 safetensors）。
- 22:37 补齐合并目录辅助文件（tokenizer/processor/chat_template，来自 checkpoint 的 huggingface/ 与基座模型目录）。
- 22:40-22:55 冒烟评测（merged grpo_200 → AIME25，GPU3 挤占训练卡，gpu_mem 0.20）：**引擎初始化失败 = CUDA graph KV 预算为负**。结论：与训练共存不可行，但权重/config/processor/tokenizer 加载全部正常；今晚正式评测等 GPU 空后用 0.80（下午验证过的同款配置）。第一票否决规则写入剧本。
- 22:58 safetensors 结构校验通过：724 张量、bf16、vocab 248320×2560、语言塔+视觉塔齐全、Qwen3_5ForConditionalGeneration。
- 22:59 定时任务已挂：automation-c685d870，23:00-08:30 每 30 分钟，共 21 轮，仅今晚。

## 值守阶段（23:00 起，每 30 分钟一轮）

- 23:00 轮：OPSD 184/200（134 s/it，ETA 23:36）、collapse=0、磁盘好转至 71%（1.5T free，约 900G 被释放——疑似其他任务清理或 ckpt 裁剪）。链路存活。无异常。转入 OPSA 实现：
  - 发现 overlay 未提交改动：启动脚本已加 `calculate_entropy=True`（15:52 的 entropy resume 即为此），另有未跟踪 `scripts/eval_checkpoint_tir.py`（现成的 checkpoint 评测 CLI，今晚管线未用它）。
  - 实现 `opsa_loss.py`（纯函数 `opsa_token_advantage` + verl 入口 `opsa_loss`，镜像 sdar_only_loss 模式）、trainer 路由（`_METHODS`+`on_init_end`+`_compute_advantage` 走 grpo 式无 teacher 路径）、agent loop teacher 门控豁免 opsa、启动脚本加 opsa 分支与 `+simpletir.opsa_*` 超参。
  - 测试三轮迭代：① 非 2D nonzero 断言写法、② `no_padding_2_padding` 契约（flat 须含 prompt+response、且 logp 按"写在前一位"左移布局）、③ **opsa_loss 真 bug：`clamp(1.0+clip_low, 1.0+clip_high)` 把 ratio 恒钳成 1.2、梯度恒零**（clamp 上下界写错符号）——修复为 `clamp(1.0-clip_low, 1.0+clip_high)`。该 bug 由新单测抓出，验证了 fail-loud 设计。
  - 结果：OPSA 单测 10/10；全库回归 41/41（31 旧 + 10 新）；PREFLIGHT `simpletir_preflight=ok`。
  - 本地提交 `b480d5c`（仅 OPsa 相关 5 文件；overlay 的 git 对账与 push 留给早上——overlay HEAD da074fb 与本地 a632432 历史有分叉，夜间接 scp 直推文件系统）。
- 23:29 收尾快照：OPSD 195/200（98%，约 23:43 完成），无异常。OPSA 冒烟（2 步）与 200 步正式启动等 `EVALS_DONE` 标记（预计 00:30-00:40），由下一轮执行。
- 23:30 轮：OPSD 196/200（ETA 23:41）、collapse=0、磁盘 71%、链路存活。GPU 0-3 显存升高为 step-195 验证 rollout 的正常开销。无异常，无需行动——OPSD 退出后链路自动 merge+评测；OPSA 冒烟与启动等 EVALS_DONE（预计 00:30-00:40 轮次执行）。
- 00:00 轮：**OPSD 干净跑完 200 步**（OPSD_CLEAN200）；链路 23:43–23:56 全自动完成 grpo/agentopsd 的 merge+评测（`EVALS_DONE`）。**发现链路 bug**：`sort -t_ -k3` 在完整路径上排序键取错字段，OPSD merge 误选了字母序尾部的 85/90/95（重启前的 data.pt 格式，无 actor/）→ OPSD 的 6 个评测整体遗漏。已手动修复：确认 190/195/200 有 actor/ → 重发 3 个 merge（CPU）→ 挂"merge 完成后自动跑 6 个评测"的后台链（GPU 0/1，`opsd_eval_after_merge.sh`）。
- 00:05 首批终评结果（n=30，基座 AIME25=30%）：agentopsd_200 aime **56.7%** / aime25 **43.3%**；agentopsd_195 53.3%/50.0%；grpo_200 43.3%/**20.0%**（aime25 末段回落）；grpo_195 46.7%/43.3%。AgentOPSD 末段优于 GRPO。
- 00:08 **OPSA 2 步冒烟已启动**（GPU 2,3，`opsa_smoke2_s42`，8 prompt×4 rollout，无验证集，pid 3441214，日志 `$NIGHT/opsa_smoke.log`）。下一轮：查冒烟结果（奖励非零/无 NaN/正常落盘）→ 通过则启动 200 步正式训练。
- 00:30 轮：OPSD 6 个补跑评测已于 00:14 全部完成（opsd_200 aime 36.7%/aime25 20.0%；opsd_195 40.0%/23.3%；opsd_190 30.0%/23.3%）。**三方 step-200 终评齐**：AgentOPSD 56.7/43.3 > GRPO 43.3/20.0 > OPSD 36.7/20.0（AIME24/25，基座 aime25=30%）。OPSA 冒烟仍在跑（00:08 启动，GPU 2,3 满载，vllm/JIT 正常，无 metrics 无错误——启动期属预期节奏）。判定与正式启动留待 01:00 轮。
- 01:00 轮：**冒烟抓到真 bug 并已修复**：step-1 actor 更新时 `OPSA selected no tokens` 崩溃——`ppo_micro_batch_size_per_gpu=1` 下 micro-batch 可能只含 `_balance_batch` 的全零 padding 行，选点为空属合法（零损失 padding 不变量），原 fail-loud 守卫误伤。修复为：mask 全零 → 返回连接计算图的零损失；"有真实 token 却选不出点"保留 raise（构造上不可能，出现即布局被改）。新增回归用例，OPSA 单测 11/11。
  - 冒烟其余路径全部正常：rollout ✓、teacher 豁免 ✓、损失被调用 ✓、Ray/沙箱无残留。
  - **GPU 全占**：0,1=ruiwenhu(sglang，00:56 起)、2,3=cljx(01:25 起)、4-7=weixuan(vllm，已 5h49m)。按红线不碰。**按剧本 04:00 规则：OPSA 200 步今晚放弃启动**，除非 04:00 前有卡释放（后续轮次继续查）。修复后的 GPU 冒烟 + 正式启动留给白天/明晚窗口。
- 01:30 轮：GPU 无变化，8 卡仍全被占用（0,1 ruiwenhu / 2,3 cljx / 4-7 weixuan）。按红线不碰，维持 04:00 放弃启动预案，继续巡检。无其他动作。
- 02:00 轮：GPU 仍无空卡。cljx 在 2,3 换了新进程（3783962/3，旧进程结束新进程顶上），ruiwenhu/weixuan 不变。继续按 04:00 规则巡检，无其他动作。
- 02:30 轮：GPU 仍无空卡，且 ruiwenhu 的 sglang（0,1）已进入 100% 满载计算——8 卡全部活跃占用。04:00 放弃启动预案基本确定生效，OPSA 启动顺延至后续窗口；继续巡检至 08:00 出晨报。
- 03:00 轮：GPU 无变化（0,1 ruiwenhu 占满、2,3 cljx 满载、4-7 weixuan）。无空卡，维持放弃启动预案，无其他动作。
- 03:30 轮：GPU 无变化（0,1 满载 89%/88%，2,3 满载）。无空卡，放弃启动预案正式生效（过了 04:00 即不再尝试启动），下轮起仅做例行巡检，08:00 出晨报。
- ~06:10 用户交互请求：提前生成晨报。已从服务器抽取三方全量训练曲线（metrics.jsonl 里程碑 + 全序列 val-core 验证分）与 14 个终评结果，整理为 `reports/night_20260909/morning_report.md`（提取脚本 `extract_metrics.py`）。
- 09:10（9/10，收尾轮）：定时任务末次触发。三组 formal 全部 step-200 完结、无训练在跑；**GPU 2-7 已全空**（0,1 仍驻留 ruiwenhu 的 sglang 显存但 0% 利用率）。夜间全部目标已完成：三方 200 步 + 14 终评 + 基座双 baseline 3-run + agentopsd 中期 4 ckpt 3-run（24/24 range=0）+ OPSA 就绪待启动。OPSA 白天启动未获用户指示（计划范围"只排今晚"），不擅自启动——等用户决定（白天跑占付费额度，或今晚 23:00 免费窗口）。定时任务配额（maxRuns=21）已基本耗尽，自然结束。
- 04:00 轮：GPU 无变化（0,1 ruiwenhu 瞬时空闲利用率但显存占满、2,3 cljx 满载），磁盘 71% 稳定。**04:00 规则生效：OPSA 今晚启动正式放弃**，代码保持就绪状态待后续窗口。剩余轮次仅例行巡检，08:00 出晨报。
- 04:30 轮：GPU 无变化。例行巡检，无动作。
- 05:00 轮：GPU 无变化（0,1 ruiwenhu 占显存、2,3 cljx 满载、4-7 weixuan）。例行巡检，无动作。
- 05:30 轮：GPU 无变化。例行巡检，无动作。
- 06:00 轮：GPU 无变化。例行巡检，无动作。
