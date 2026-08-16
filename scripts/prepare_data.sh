#!/usr/bin/env bash
# 拿取数据集并预处理成 verl/SDAR 可读的 parquet。
# 用法: bash scripts/prepare_data.sh [alfworld|webshop|search]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ENV_NAME="${1:-${ENV_NAME:-alfworld}}"
require_sdar

prepare_alfworld() {
  log "==> ALFWorld：安装/确认依赖"
  set +e
  python3 - <<'PY'
import importlib
for m in ("gymnasium", "stable_baselines3", "alfworld"):
    try:
        importlib.import_module(m)
    except Exception:
        print(f"missing {m}")
        raise SystemExit(1)
PY
  _dep_check=$?
  set -e
  if [[ ${_dep_check} -ne 0 ]]; then
    log "缺少 ALFWorld 依赖，正在安装（uv 装进当前 conda 环境）..."
    uv pip install --python "$(command -v python3)" gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
  fi

  log "==> 下载 ALFWorld PDDL/Game/预训练检测器（~/.cache/alfworld）"
  require_cmd alfworld-download
  alfworld-download -f

  log "==> 生成占位 parquet（任务在 rollout 时由环境实例化，行数=任务数）"
  export ALFWORLD_DATA="${DATA_ROOT}/alfworld"
  mkdir -p "${ALFWORLD_DATA}"
  (
    cd "${SDAR_ROOT}"
    HF_ENDPOINT="${HF_ENDPOINT}" python3 -m examples.data_preprocess.prepare \
      --mode text \
      --local_dir "${DATA_ROOT}/verl-agent" \
      --train_data_size "${TRAIN_TASKS}" \
      --val_data_size "${VAL_TASKS}"
  )
  ls -lh "${DATA_ROOT}/verl-agent/text/"
  log "==> ALFWorld 数据就绪: ${DATA_ROOT}/verl-agent/text/{train,test}.parquet"
}

prepare_webshop() {
  log "==> WebShop 需要 Python<=3.10 独立环境（SDAR 官方要求），本脚本只给出手工步骤"
  log "    1) conda create -n verl-webshop python==3.10 -y && conda activate verl-webshop"
  log "    2) cd ${SDAR_ROOT}/agent_system/environments/env_package/webshop/webshop && ./setup.sh -d all"
  log "    3) cd ${SDAR_ROOT} && uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
  log "    4) uv pip install -e ${SDAR_ROOT} && uv pip install vllm==0.8.2"
  log "    5) 复用上面的 ALFWorld 占位 parquet（数据源一致）：${DATA_ROOT}/verl-agent/text/"
  die "WebShop 环境未自动搭建，请按上面的步骤在手工环境里执行 train.sh"
}

prepare_search() {
  log "==> Search-QA：需要独立的 faiss-gpu 检索服务（约 6GB/卡），手工步骤："
  log "    1) conda create -n retriever python=3.10 -y && conda activate retriever"
  log "    2) conda install numpy==1.26.4 && uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
  log "    3) uv pip install transformers datasets pyserini huggingface_hub uvicorn fastapi"
  log "    4) conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y"
  log "    5) cd ${SDAR_ROOT} && uv pip install -e . && python3 examples/search/searchr1_download.py --local_dir ${DATA_ROOT}/searchR1"
  log "       cat ${DATA_ROOT}/searchR1/part_* > ${DATA_ROOT}/searchR1/e5_Flat.index && gzip -d ${DATA_ROOT}/searchR1/wiki-18.jsonl.gz"
  log "    6) bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log  # 保持运行"
  log "    7) cd ${SDAR_ROOT} && python3 examples/data_preprocess/preprocess_search_r1_dataset.py"
  log "       数据输出: ~/data/searchR1_processed_direct（train/val parquet）"
  die "Search 检索服务需手工启动，请按上面的步骤执行"
}

case "${ENV_NAME}" in
  alfworld) prepare_alfworld ;;
  webshop)  prepare_webshop ;;
  search)   prepare_search ;;
  *) die "未知环境 ${ENV_NAME}，支持: alfworld | webshop | search" ;;
esac
