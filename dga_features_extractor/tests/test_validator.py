from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dga_features_extractor.validator import validate_outputs


COLUMNS = ["DOMAIN", "CLASS", "LABEL"]


def _validate(tmp_path: Path, rows: list[dict[str, object]], expected_columns: list[str] = COLUMNS):
    output_dir = tmp_path / "output"
    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), output_dir / "dataset.parquet")
    return validate_outputs(
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        base_name="dataset",
        split_threshold_bytes=1000,
        encoding="utf-8",
        sample_file_name="sample.arff",
        require_sample=False,
        expected_columns=expected_columns,
        require_balanced=True,
    )


def test_validator_rejects_invalid_label(tmp_path: Path) -> None:
    result = _validate(tmp_path, [{"DOMAIN": "bad.example", "CLASS": "test", "LABEL": 2}])
    assert not result.valid
    assert any("Invalid LABEL" in error for error in result.errors)


def test_validator_rejects_cross_label_overlap(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        [
            {"DOMAIN": "same.example", "CLASS": "example", "LABEL": 0},
            {"DOMAIN": "same.example", "CLASS": "test_dga", "LABEL": 1},
        ],
    )
    assert not result.valid
    assert any("overlap" in error for error in result.errors)


def test_validator_rejects_invalid_class_and_unbalanced_data(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        [
            {"DOMAIN": "one.example", "CLASS": "wrong", "LABEL": 0},
            {"DOMAIN": "two.example", "CLASS": "example", "LABEL": 0},
            {"DOMAIN": "dga.example", "CLASS": "test_dga", "LABEL": 1},
        ],
    )
    assert not result.valid
    assert any("Invalid CLASS" in error for error in result.errors)
    assert any("not balanced" in error for error in result.errors)


def test_validator_rejects_schema_mismatch(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        [
            {"DOMAIN": "one.example", "CLASS": "example", "LABEL": 0},
            {"DOMAIN": "dga.example", "CLASS": "test_dga", "LABEL": 1},
        ],
        expected_columns=["DOMAIN", "FEATURE", "CLASS", "LABEL"],
    )
    assert not result.valid
    assert any("Unexpected columns" in error for error in result.errors)
