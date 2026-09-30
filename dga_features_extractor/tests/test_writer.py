from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dga_features_extractor.writer import SplitParquetWriter


SCHEMA = pa.schema(
    [
        pa.field("DOMAIN", pa.string(), nullable=False),
        pa.field("FEATURE", pa.float64(), nullable=False),
        pa.field("CLASS", pa.string(), nullable=False),
        pa.field("LABEL", pa.int8(), nullable=False),
    ]
)


def _row(index: int, label: int = 0) -> dict[str, object]:
    return {
        "DOMAIN": f"domain-{index}.example",
        "FEATURE": float(index),
        "CLASS": "example" if label == 0 else "test_dga",
        "LABEL": label,
    }


def test_writer_finalizes_single_file(tmp_path: Path) -> None:
    writer = SplitParquetWriter(tmp_path, "dataset", 1_000_000, SCHEMA, row_group_size=2)
    writer.write_row(_row(1), 0)
    writer.write_row(_row(2, 1), 1)

    outputs = writer.finalize()

    assert [item.path.name for item in outputs] == ["dataset.parquet"]
    assert pq.read_table(outputs[0].path).num_rows == 2


def test_writer_rotates_with_small_threshold(tmp_path: Path) -> None:
    writer = SplitParquetWriter(tmp_path, "dataset", 200, SCHEMA, row_group_size=1)
    for index in range(4):
        writer.write_row(_row(index, index % 2), index % 2)

    outputs = writer.finalize()

    assert len(outputs) > 1
    assert all(item.path.name.startswith("dataset_part_") for item in outputs)
    assert sum(pq.read_table(item.path).num_rows for item in outputs) == 4


def test_empty_writer_creates_no_part(tmp_path: Path) -> None:
    writer = SplitParquetWriter(tmp_path, "dataset", 200, SCHEMA, row_group_size=1)
    assert writer.finalize() == []
    assert not list(tmp_path.glob("*.parquet"))
