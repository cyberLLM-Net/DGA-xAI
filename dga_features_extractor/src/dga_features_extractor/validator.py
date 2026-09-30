from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq

from .schema import CLASS_COLUMN, DOMAIN_COLUMN, LABEL_COLUMN


@dataclass
class ValidationResult:
    valid: bool
    summary: Dict[str, object]
    errors: List[str]


def _find_parquet_files(output_dir: Path, base_name: str) -> List[Path]:
    single = output_dir / f"{base_name}.parquet"
    parts = sorted(output_dir.glob(f"{base_name}_part_*.parquet"))
    if single.exists():
        return [single]
    return parts


def _extract_tld(domain: str) -> str:
    if "." not in domain:
        return ""
    return domain.rsplit(".", 1)[-1]


def _validate_parquet_files(
    files: List[Path],
    expected_columns: Optional[List[str]],
    errors: List[str],
    require_balanced: bool,
) -> Dict[str, object]:
    total_rows = 0
    benign_rows = 0
    dga_rows = 0
    details: List[Dict[str, object]] = []
    benign_domains: set[str] = set()
    dga_domains: set[str] = set()

    schema_reference: List[str] | None = None

    for path in files:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            errors.append(f"Cannot read parquet file {path}: {exc}")
            continue

        columns = table.column_names
        if schema_reference is None:
            schema_reference = columns
        elif schema_reference != columns:
            errors.append(f"Schema mismatch in parquet file: {path}")

        if expected_columns is not None and columns != expected_columns:
            errors.append(f"Unexpected columns in {path}: expected={expected_columns} actual={columns}")

        if "class" in columns:
            errors.append(f"Legacy `class` column is not allowed in {path}; expected `{LABEL_COLUMN}`.")
        if DOMAIN_COLUMN not in columns:
            errors.append(f"Missing `{DOMAIN_COLUMN}` column in {path}")
            continue
        if CLASS_COLUMN not in columns:
            errors.append(f"Missing `{CLASS_COLUMN}` column in {path}")
            continue
        if LABEL_COLUMN not in columns:
            errors.append(f"Missing `{LABEL_COLUMN}` column in {path}")
            continue

        num_rows = table.num_rows
        total_rows += num_rows

        domains = table.column(DOMAIN_COLUMN).to_pylist()
        classes = table.column(CLASS_COLUMN).to_pylist()
        labels = table.column(LABEL_COLUMN).to_pylist()
        b_rows = sum(1 for item in labels if item == 0)
        d_rows = sum(1 for item in labels if item == 1)
        invalid = [item for item in labels if item not in (0, 1)]
        if invalid:
            errors.append(f"Invalid LABEL values in {path}: {invalid[:5]}")

        for domain, class_value, label in zip(domains, classes, labels):
            domain_value = str(domain)
            class_str = str(class_value)
            if label == 0:
                if class_str != _extract_tld(domain_value):
                    errors.append(
                        f"Invalid CLASS for benign domain in {path}: domain={domain_value} class={class_str}"
                    )
                benign_domains.add(domain_value)
            elif label == 1:
                if not class_str.strip():
                    errors.append(f"Empty CLASS for DGA domain in {path}: domain={domain_value}")
                dga_domains.add(domain_value)

        benign_rows += b_rows
        dga_rows += d_rows

        details.append(
            {
                "path": str(path),
                "rows": num_rows,
                "benign_rows": b_rows,
                "dga_rows": d_rows,
                "size_bytes": path.stat().st_size,
                "columns": columns,
            }
        )

    overlap = benign_domains & dga_domains
    if overlap:
        errors.append(f"Cross-label DOMAIN overlap detected ({len(overlap)} domains)")
    if require_balanced and benign_rows != dga_rows:
        errors.append(f"Dataset is not balanced: benign={benign_rows} dga={dga_rows}")

    return {
        "total_rows": total_rows,
        "benign_rows": benign_rows,
        "dga_rows": dga_rows,
        "cross_label_overlap_count": len(overlap),
        "schema_columns": schema_reference or [],
        "files": details,
    }


def _validate_sample_arff(sample_path: Path, expected_columns: Optional[List[str]], errors: List[str]) -> Dict[str, object]:
    sample_rows = 0
    attrs: List[str] = []
    data_started = False

    if not sample_path.exists():
        errors.append(f"Sample file is missing: {sample_path}")
        return {"exists": False, "rows": 0, "attributes": []}

    with open(sample_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split()
                if len(parts) >= 2:
                    attrs.append(parts[1])
            if line.upper() == "@DATA":
                data_started = True
                continue
            if data_started:
                fields = next(csv.reader([line], delimiter=",", quotechar="'", escapechar="\\"))
                sample_rows += 1
                if len(fields) != len(attrs):
                    errors.append(
                        f"Sample ARFF field count mismatch at row {sample_rows}: expected={len(attrs)} actual={len(fields)}"
                    )

    if not data_started:
        errors.append(f"Missing @DATA in sample file: {sample_path}")

    if expected_columns is not None and attrs != expected_columns:
        errors.append(f"Sample ARFF schema mismatch: expected={expected_columns} actual={attrs}")

    return {
        "exists": True,
        "rows": sample_rows,
        "attributes": attrs,
        "path": str(sample_path),
    }


def validate_outputs(
    output_dir: Path,
    metadata_dir: Path,
    base_name: str,
    split_threshold_bytes: int,
    encoding: str,
    sample_file_name: str,
    require_sample: bool,
    expected_format: str = "parquet",
    expected_columns: Optional[List[str]] = None,
    require_balanced: bool = True,
) -> ValidationResult:
    del split_threshold_bytes
    del encoding

    errors: List[str] = []
    files = _find_parquet_files(output_dir, base_name)
    if expected_format != "parquet":
        errors.append(f"Unsupported validation format: {expected_format}")

    if not files:
        return ValidationResult(valid=False, summary={}, errors=[f"No parquet output found for base '{base_name}' in {output_dir}"])

    parquet_summary = _validate_parquet_files(files, expected_columns, errors, require_balanced=require_balanced)

    sample_summary: Dict[str, object] = {
        "exists": False,
        "rows": 0,
        "attributes": [],
        "path": str(output_dir / sample_file_name),
    }
    if require_sample:
        sample_summary = _validate_sample_arff(output_dir / sample_file_name, expected_columns, errors)

    manifest_path = metadata_dir / f"{base_name}_manifest.json"
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        try:
            json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Manifest is not valid JSON: {manifest_path} ({exc})")

    summary = {
        "format": expected_format,
        "file_count": len(files),
        "total_rows": parquet_summary["total_rows"],
        "benign_rows": parquet_summary["benign_rows"],
        "dga_rows": parquet_summary["dga_rows"],
        "cross_label_overlap_count": parquet_summary["cross_label_overlap_count"],
        "schema_columns": parquet_summary["schema_columns"],
        "files": parquet_summary["files"],
        "sample": sample_summary,
        "manifest_exists": manifest_exists,
    }
    return ValidationResult(valid=not errors, summary=summary, errors=errors)
