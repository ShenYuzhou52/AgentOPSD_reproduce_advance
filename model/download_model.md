# 模型参数下载指南（镜像方案）

本文档解决一个问题：**在服务器上把 8B 模型权重稳定、可校验地下载到本地目录**。默认模型 `Qwen/Qwen3-8B-Instruct`（约 16GB bf16），推荐目录 `$HOME/models/Qwen3-8B-Instruct`。

三种方案按优先级排列，全部可由 `scripts/download_model.sh` 一键完成。

## 方案 A：HuggingFace 镜像（hf-mirror，推荐）

服务器无法直连 huggingface.co 时，用镜像站 `hf-mirror.com`（HF 官方支持的社区镜像）。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface

# 方式 1：huggingface_hub CLI（推荐，支持断点续传/多线程）
hf download Qwen/Qwen3-8B-Instruct --local-dir $HOME/models/Qwen3-8B-Instruct

# 方式 2：老版 CLI
huggingface-cli download Qwen/Qwen3-8B-Instruct --local-dir $HOME/models/Qwen3-8B-Instruct
```

仓库脚本自动执行以上逻辑：

```bash
bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct $HOME/models
```

下载后设置 `HF_HUB_ENABLE_HF_TRANSFER=1` 可启用 `hf_transfer` 高速下载（需 `uv pip install hf_transfer`）。

## 方案 B：ModelScope（国内直连，HF 不通时的首选备选）

Qwen 官方同时在 ModelScope 发布权重，国内直连快。

```bash
uv pip install modelscope
MODELSCOPE_ENABLED=1 bash scripts/download_model.sh Qwen/Qwen3-8B-Instruct $HOME/models

# 等价手工命令：
modelscope download --model Qwen/Qwen3-8B-Instruct --local_dir $HOME/models/Qwen3-8B-Instruct
```

## 方案 C：GitHub 镜像 / GitHub LFS

适用两类场景：

1. **权重托管在 GitHub LFS 的仓库**（部分开源模型以 git-lfs 形式发布，或你想从 GitHub 镜像站拉）：先用 GitHub 加速前缀，再正常 `git clone`。
2. **只想通过 GitHub 镜像加速下载**本项目的其他资源（SDAR 等），或权重发布方提供了 GitHub Release 包。

GitHub 加速前缀（任选一个可用节点）：

```bash
# gh-proxy 系加速（把原 URL 拼在域名后）
git clone https://ghproxy.net/https://github.com/<org>/<repo>.git
git clone https://ghfast.top/https://github.com/<org>/<repo>.git
```

对 git-lfs 仓库（例如某些镜像仓库把 safetensors 用 LFS 托管）：

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=0 git clone https://ghproxy.net/https://github.com/<org>/<repo>.git $HOME/models/<repo>
```

> ⚠️ 安全提醒：GitHub 上的权重仓库**不一定来自官方**。优先使用 Qwen 官方 HF/ModelScope 仓库（`Qwen/Qwen3-8B-Instruct`）；使用第三方 GitHub 镜像前，务必校验文件与官方一致（见下）。

## 校验与放置

```bash
MODEL_DIR=$HOME/models/Qwen3-8B-Instruct

# 1) 目录完整性：必须出现 config.json 与 safetensors
ls -lh $MODEL_DIR | head -30
[[ -f $MODEL_DIR/config.json ]] && echo OK

# 2) 校验索引：model.safetensors.index.json 里声明的文件都应存在
python3 - <<'PY'
import json, os
idx = json.load(open("$MODEL_DIR/model.safetensors.index.json"))
missing = [f for f in idx["weight_map"].values() if not os.path.exists("$MODEL_DIR/" + f)]
assert not missing, missing
print("safetensors 索引校验通过，分片数:", len(set(idx["weight_map"].values())))
PY

# 3) 权重加载冒烟（会真正读入内存，8B 约 16GB）
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('$MODEL_DIR', trust_remote_code=True)
print('load ok, params:', sum(p.numel() for p in m.parameters())/1e9, 'B')
"

# 4) 记录 sha256（可选，留档）
sha256sum $MODEL_DIR/*.safetensors > $MODEL_DIR/SHA256SUMS.txt
```

训练脚本通过 `MODEL_NAME` 指向该目录：

```bash
MODEL_NAME=$HOME/models/Qwen3-8B-Instruct bash scripts/train.sh
```

## 常见问题

| 问题 | 处理 |
|---|---|
| `hf` 命令不存在 | `uv pip install "huggingface_hub[cli]"` 后重试 |
| 镜像 403 / 限速 | 换 `HF_ENDPOINT=https://hf-mirror.com` 其他镜像节点，或切 ModelScope |
| 磁盘不足 | 8B bf16 约 16GB，需 ≥ 40GB 余量（含下载缓存 `~/.cache/huggingface`），`HF_HOME` 指向大盘目录 |
| 与训练脚本路径不一致 | 训练脚本自动找 `$MODEL_CACHE/<模型名>`；确认 `MODEL_CACHE`（默认 `$HOME/models`）一致 |
| vLLM 加载报 safetensors 缺失 | 权重没下全，重跑 download_model.sh；不要手工改文件名 |

