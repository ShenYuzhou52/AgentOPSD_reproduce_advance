"""Import and schema preflight for the isolated TIR overlay (no GPU allocation).

服务器侧预检入口：一次跑通"控制器能 import 全部组件 + 四个数据文件满足
防泄漏契约"。任何时候改完代码/换数据后先跑它，几秒钟就能发现导入错误或
schema 问题，而不用等 vLLM 初始化几分钟后才失败。
"""

from __future__ import annotations

from integrations.simpletir_qwen35.agent_loop_manager import SimpleTIRAgentLoopManagerTQ
from integrations.simpletir_qwen35.data_contract import validate_simpletir_files
from integrations.simpletir_qwen35.opsd_loss import sdar_only_loss
from integrations.simpletir_qwen35.trainer import SimpleTIRTrainer


def main() -> None:
    summaries = validate_simpletir_files(
        [
            "/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/simplelr_math_35/train.parquet",
            "/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/simplelr_math_35/test.parquet",
            "/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepscaler/aime.parquet",
            "/data2/ssd/yixinshen/benchmarks/SimpleTIR/datasets/deepscaler/aime25.parquet",
        ]
    )
    # Never print source examples.  Names establish all controller/worker
    # imports resolve; summaries are limited to row counts and field names.
    print("imports=ok", SimpleTIRTrainer.__name__, SimpleTIRAgentLoopManagerTQ.__name__, sdar_only_loss.__name__)
    for item in summaries:
        print(f"schema=ok rows={item['rows']} checked={item['checked_rows']} columns={','.join(item['columns'])}")


if __name__ == "__main__":
    main()
