# AgentOPSD：8B 训练稳定性实测

本项目**实测 AgentOPSD（Recursive Self-Distillation for Agentic RL, arXiv:2608.05987）在 8B 模型上的训练稳定性**。默认模型 Qwen3-8B-Instruct，默认环境 ALFWorld，训练复用 SDAR（ZJU-REAL，verl 系框架）作为基座，AgentOPSD 核心算法在本仓库独立实现并通过最小钩子接入训练循环。

本地仓库**只保存代码**（`.gitignore` 已排除数据/权重/断点/日志/vendor）；代码通过 **ssh git** 推到服务器，服务器环境由 **uv** 维护，复用已有 `CUDA-12-conda`（vllm、flash-attn、deepspeed、IPC），训练脚本适配 **2~4 张 A800**，默认使用**第 2/3/4/5 卡**，支持**断点续训**。

## 快速开始

```bash
# 0) 本地：提交并推送代码到服务器
git remote add origin ssh://<user>@<server>/~/git/AgentOPSD.git
bash scripts/git_sync.sh

# 1) 服务器：环境（uv + conda CUDA-12-conda + SDAR 克隆）
bash scripts/setup_server.sh

# 2) 服务器：模型（默认 hf-mirror，备选 ModelScope/GitHub 镜像，见 model/download_model.md）
bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct

# 3) 服务器：数据（ALFWorld）
bash scripts/prepare_data.sh alfworld

# 4) 服务器：训练（4 卡 2,3,4,5，150 步；断点续训默认 auto）
bash scripts/train.sh

# 5) 评测与稳定性报告
bash scripts/eval.sh --ckpt checkpoints/<实验>/global_step_40
python3 scripts/stability_report.py --root logs --out report.md
```

常用变体：

```bash
bash scripts/train.sh --gpus 2,3 --steps 50            # 2 卡
bash scripts/train.sh --resume-ckpt checkpoints/xxx/global_step_40   # 指定断点续训
bash scripts/train.sh --disable-credit                 # 纯 GRPO 对照
SEEDS="0 1 2" LAM_SWEEP="0.5 0.25" bash scripts/run_stability_matrix.sh
bash scripts/smoke_test.sh                             # 2 卡 2 步链路冒烟
```

## 环境配置（服务器）

服务器已有：`CUDA-12-conda`（conda 环境）、vllm、flash-attn、deepspeed、IPC（共享内存）。本项目用 uv 在同一 conda 环境内管理依赖：

```bash
conda activate CUDA-12-conda
python -m pip install uv
bash scripts/setup_server.sh   # 内部完成 uv pip install -e .（本项目）+ -e vendor/SDAR
```

ALFWorld 额外依赖由 `prepare_data.sh` 自动安装（gymnasium、stable-baselines3、alfworld）。WebShop（Python≤3.10）与 Search-QA（faiss-gpu 检索服务）见 [docs/03-训练全流程](docs/03-训练全流程.md)。

## 文档导航

- [docs/01-项目目标与背景](docs/01-项目目标与背景.md)：目标、术语、仓库结构
- [docs/02-AgentOPSD方法与奖励信用分配](docs/02-AgentOPSD方法与奖励信用分配.md)：**RLVR 奖励信用分配**的完整推导与实现细节
- [docs/03-训练全流程](docs/03-训练全流程.md)：数据→训练→续训→评测全流程
- [docs/04-训练稳定性评测方案](docs/04-训练稳定性评测方案.md)：稳定性指标、预警、实验矩阵
- [model/download_model.md](model/download_model.md)：GitHub/HF/ModelScope 镜像下载指南

## 目录

```text
agentopsd/     AgentOPSD 算法（credit.py）与训练集成（patch/main）
scripts/       数据/训练/评测/合并/稳定性脚本（bash + python）
docs/          技术文档（方法、流程、稳定性方案）
model/         模型镜像下载指南
vendor/SDAR    基座框架（setup_server.sh 克隆，不提交）
```

## 引用

```bibtex
@article{wang2026agentopsd,
  title={AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning},
  author={Wang, Zi-Han and others},
  journal={arXiv preprint arXiv:2608.05987}, year={2026}}
@article{lu2026sdar,
  title={Self-distilled agentic reinforcement learning},
  author={Lu, Zhengxi and others},
  journal={arXiv preprint arXiv:2605.15155}, year={2026}}
```

