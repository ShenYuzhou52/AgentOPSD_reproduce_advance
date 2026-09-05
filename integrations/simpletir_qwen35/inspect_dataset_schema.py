"""Print SimpleTIR Parquet field names/types without emitting question or answer text."""

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
