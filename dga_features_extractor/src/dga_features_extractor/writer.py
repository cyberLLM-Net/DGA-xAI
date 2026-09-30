from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .arff import generate_arff_header


@dataclass
class OutputFileStats:
    path: Path
    part_index: int
    bytes_written: int
    data_rows: int
    class_distribution: Dict[str, int]
    warnings: List[str]
    finalized_at: str
    generated_successfully: bool = True


class SplitParquetWriter:
    """Streaming parquet writer with deterministic size-based rotation.

    Rotation uses an estimated in-memory row footprint to stay safely under threshold,
    since exact parquet size is only known after row groups are flushed and file is closed.
    """

    def __init__(
        self,
        output_dir: Path,
        base_name: str,
        split_threshold_bytes: int,
        parquet_schema: pa.Schema,
        row_group_size: int,
        on_part_closed: Optional[Callable[[OutputFileStats, str], None]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.base_name = base_name
        self.split_threshold_bytes = split_threshold_bytes
        self.parquet_schema = parquet_schema
        self.row_group_size = row_group_size
        self.on_part_closed = on_part_closed

        self._part_index = 0
        self._writer: pq.ParquetWriter | None = None
        self._current_path: Path | None = None
        self._current_rows = 0
        self._current_class_distribution = {"0": 0, "1": 0}
        self._split_used = False

        self._buffer: List[Dict[str, object]] = []
        self._estimated_bytes_in_part = 0

        self.output_files: List[OutputFileStats] = []

    def _part_path(self, part_index: int) -> Path:
        return self.output_dir / f"{self.base_name}_part_{part_index:04d}.parquet"

    def _open_new_part(self) -> None:
        self._part_index += 1
        self._current_path = self._part_path(self._part_index)
        self._writer = pq.ParquetWriter(str(self._current_path), self.parquet_schema)
        self._current_rows = 0
        self._current_class_distribution = {"0": 0, "1": 0}
        self._estimated_bytes_in_part = 0

    def _estimate_row_bytes(self, row: Dict[str, object]) -> int:
        # Conservative estimate to avoid greatly overshooting threshold.
        domain = str(row.get("DOMAIN", row.get("domain", "")))
        n_cols = len(self.parquet_schema.names)
        return max(128, len(domain) + (n_cols * 8) + 64)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        assert self._writer is not None
        table = pa.Table.from_pylist(self._buffer, schema=self.parquet_schema)
        self._writer.write_table(table, row_group_size=self.row_group_size)
        self._buffer.clear()

    def _close_current_part(self, reason: str) -> None:
        if self._writer is None or self._current_path is None:
            return

        self._flush_buffer()
        self._writer.close()

        file_size = self._current_path.stat().st_size if self._current_path.exists() else 0
        stats = OutputFileStats(
            path=self._current_path,
            part_index=self._part_index,
            bytes_written=file_size,
            data_rows=self._current_rows,
            class_distribution=dict(self._current_class_distribution),
            warnings=[],
            finalized_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
        self.output_files.append(stats)

        self._writer = None
        self._current_path = None

        if self.on_part_closed is not None:
            self.on_part_closed(stats, reason)

    def write_row(self, row: Dict[str, object], label: int) -> None:
        if self._writer is None:
            self._open_new_part()

        assert self._writer is not None
        assert self._current_path is not None

        row_estimate = self._estimate_row_bytes(row)

        if self._current_rows > 0 and (self._estimated_bytes_in_part + row_estimate) >= self.split_threshold_bytes:
            self._split_used = True
            self._close_current_part(reason="rotation")
            self._open_new_part()

        self._buffer.append(row)
        self._estimated_bytes_in_part += row_estimate
        self._current_rows += 1
        label_key = str(label)
        self._current_class_distribution[label_key] = self._current_class_distribution.get(label_key, 0) + 1

        if len(self._buffer) >= self.row_group_size:
            self._flush_buffer()

    def finalize(self) -> List[OutputFileStats]:
        self._close_current_part(reason="finalize")

        if len(self.output_files) == 1 and not self._split_used:
            single_target = self.output_dir / f"{self.base_name}.parquet"
            first_part = self.output_files[0]
            os.replace(first_part.path, single_target)
            file_size = single_target.stat().st_size if single_target.exists() else first_part.bytes_written
            self.output_files[0] = OutputFileStats(
                path=single_target,
                part_index=first_part.part_index,
                bytes_written=file_size,
                data_rows=first_part.data_rows,
                class_distribution=dict(first_part.class_distribution),
                warnings=list(first_part.warnings),
                finalized_at=first_part.finalized_at,
                generated_successfully=first_part.generated_successfully,
            )
        return self.output_files


class SampleArffWriter:
    def __init__(
        self,
        output_path: Path,
        relation_name: str,
        feature_names: List[str],
        class_attribute_name: str,
        label_attribute_name: str,
        sample_size: int,
        encoding: str,
        output_errors: str,
        buffer_size: int,
    ) -> None:
        self.output_path = output_path
        self.sample_size = sample_size
        self.rows_written = 0
        self._fh = open(
            output_path,
            "w",
            encoding=encoding,
            errors=output_errors,
            buffering=buffer_size,
            newline="",
        )
        self._fh.write(
            generate_arff_header(
                relation_name=relation_name,
                feature_names=feature_names,
                domain_attribute_name="DOMAIN",
                class_attribute_name=class_attribute_name,
                label_attribute_name=label_attribute_name,
            )
        )

    def write_row(self, row: str) -> None:
        if self.rows_written >= self.sample_size:
            return
        self._fh.write(row)
        self.rows_written += 1

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()
