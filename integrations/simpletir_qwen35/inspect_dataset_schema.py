"""Print SimpleTIR Parquet field names/types without emitting question or answer text.

排查数据问题时用的最小工具：只打印 schema 与列类型。刻意不打印任何行
内容——样本文本（尤其 ground_truth）不允许出现在终端记录里。
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    args = parser.parse_args()
    file = pq.ParquetFile(args.parquet)
    print(file.schema)
    table = file.read_row_group(0)
    print({name: str(table.column(name).type) for name in table.column_names})


if __name__ == "__main__":
    main()
