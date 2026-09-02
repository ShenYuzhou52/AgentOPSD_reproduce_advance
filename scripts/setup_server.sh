#!/usr/bin/env bash
# 服务器端一键环境初始化（在 CUDA-12-conda 环境内用 uv 维护依赖）
# 用法: bash scripts/setup_server.sh [--env CUDA-12-conda] [--sdar-commit <hash>]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# This pinned upstream revision is the base for patches/sdar-reproduction-fixes.patch.
# Override with --sdar-commit only together with a compatible patch revision.
SDAR_COMMIT="${SDAR_COMMIT:-80ce06909d665bfb88ac93ec6db8fa8d82631655}"
SDAR_PATCH="${AGENTOPSD_REPO_ROOT}/patches/sdar-reproduction-fixes.patch"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) CONDA_ENV="$2"; shift 2 ;;
    --sdar-commit) SDAR_COMMIT="$2"; shift 2 ;;
    *) die "未知参数: $1" ;;
  esac
done

log "==> 激活 conda 环境 ${CONDA_ENV}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  log "未检测到 conda，假定当前 shell 已激活 ${CONDA_ENV}"
fi

log "==> 安装/确认 uv"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install -U uv
fi
uv --version

log "==> 克隆 SDAR 基座框架（AgentOPSD 官方代码未发布，算法实现在本仓库 agentopsd/ 下，训练复用 SDAR/verl）"
mkdir -p "$(dirname "${SDAR_ROOT}")"
if [[ ! -d "${SDAR_ROOT}/.git" ]]; then
  git clone https://github.com/ZJU-REAL/SDAR.git "${SDAR_ROOT}"
fi
git -C "${SDAR_ROOT}" fetch --all --tags
if git -C "${SDAR_ROOT}" diff --quiet; then
  git -C "${SDAR_ROOT}" checkout "${SDAR_COMMIT}"
  git -C "${SDAR_ROOT}" apply --check --ignore-space-change "${SDAR_PATCH}"
  git -C "${SDAR_ROOT}" apply --ignore-space-change "${SDAR_PATCH}"
  log "已应用本仓库的 SDAR 复现修复补丁"
elif git -C "${SDAR_ROOT}" apply --reverse --check --ignore-space-change "${SDAR_PATCH}"; then
  log "SDAR 复现修复补丁已应用，保留当前工作区"
else
  die "${SDAR_ROOT} 存在无法识别的本地修改；请提交/清理后再执行 setup_server.sh"
fi

log "==> 用 uv 把项目依赖装进当前 conda 环境（不会动已装好的 torch/vllm/flash-attn/deepspeed）"
uv pip install --python "$(command -v python3)" -e "${AGENTOPSD_REPO_ROOT}"
uv pip install --python "$(command -v python3)" -e "${SDAR_ROOT}"

log "==> 环境自检"
check_env

log "==> 完成。下一步:"
log "    bash scripts/download_model.sh ${MODEL_NAME}"
log "    bash scripts/prepare_data.sh ${ENV_NAME}"
log "    bash scripts/train.sh"
