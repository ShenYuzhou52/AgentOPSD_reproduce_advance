# 夜间链路公共环境（20260909）— 服务器端脚本均 source 此文件
R=/data2/ssd/yixinshen/experiments/qwen35-simpletir/formal_20260907
NIGHT=$R/night_20260909
OVERLAY=/data2/ssd/yixinshen/AgentOPSD-tir
VERL=/data2/ssd/yixinshen/benchmarks/verl-qwen35-base
PY=$VERL/.venv/bin/python
MERGER=/data2/ssd/yixinshen/benchmarks/SimpleTIR/scripts/model_merger.py
DS=/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets
AIME=$DS/deepscaler/aime.parquet
AIME25=$DS/deepscaler/aime25.parquet
