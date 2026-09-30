from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa

from . import __version__
from .arff import format_arff_row
from .features import compute_domain_features_subset
from .paths import DEFAULT_BENIGN_INPUT, DEFAULT_BENIGN_INPUT_ALT
from .schema import (
    CLASS_COLUMN,
    DOMAIN_COLUMN,
    FIELD_DESCRIPTION_FILE_NAME,
    LABEL_COLUMN,
    SAMPLE_FILE_NAME,
    feature_schema_checksum,
    full_dataset_columns,
    selected_feature_names,
    write_field_descriptions,
)
from .validator import validate_outputs
from .writer import OutputFileStats, SampleArffWriter, SplitParquetWriter

LOGGER = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    benign_input: Path
    dga_input: Path
    output_dir: Path
    base_name: str
    metadata_dir: Path
    split_threshold_bytes: int
    encoding: str
    encoding_errors: str
    log_dir: Path
    log_level: str
    buffer_size: int
    deduplicate: bool
    threads: int
    batch_size: int
    temp_dir: Optional[Path]
    overwrite: bool
    relation_name: str
    manifest: bool
    progress_every: int
    validate_output: bool
    schema_reference: Optional[Path]
    compression_enabled: bool
    compression_format: str
    keep_uncompressed_full_parts: bool
    sample_size: int
    stats_json_name: str
    write_manifest: bool = False
    write_integrity: bool = False
    write_stats: bool = True
    deduplicate_global: bool = False
    split_datasets: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    integrity_json_name: str = "udcdga_dataset_integrity.json"
    dataset_name: str = "udcdga"
    dataset_version: str = "1.0.0"
    audit_sample_size: int = 100
    execution_args: Dict[str, object] = field(default_factory=dict)
    parquet_row_group_size: int = 10_000
    write_schema_description: bool = True
    schema_description_name: str = FIELD_DESCRIPTION_FILE_NAME
    output_format: str = "parquet"


@dataclass
class BuildStats:
    benign_rows_read: int = 0
    dga_rows_read: int = 0
    invalid_benign_rows: int = 0
    invalid_dga_rows: int = 0
    empty_or_invalid_rows: int = 0
    rows_written: int = 0
    benign_rows_written: int = 0
    dga_rows_written: int = 0
    duplicate_rows_dropped: int = 0
    cross_label_conflicts: int = 0
    conflict_rows_excluded: int = 0
    feature_errors: int = 0
    feature_missing: int = 0
    feature_computation_errors: int = 0
    feature_invalid_numeric: int = 0
    benign_overlap_excluded: int = 0
    benign_duplicate_rows_dropped: int = 0
    dga_duplicate_rows_dropped: int = 0
    benign_unique_domains_selected: int = 0
    dga_unique_domains_selected: int = 0
    dga_rows_reduced_for_balance: int = 0
    final_balanced_rows_per_class: int = 0


@dataclass
class DistributionStats:
    rows: int = 0
    length_distribution: Counter = field(default_factory=Counter)
    tld_distribution: Counter = field(default_factory=Counter)
    digit_ratio_histogram: Counter = field(default_factory=Counter)
    entropy_histogram: Counter = field(default_factory=Counter)


@dataclass
class SampleRecord:
    domain: str
    class_value: str
    label: int
    feature_values: List[float]


@dataclass(frozen=True)
class FraudRecord:
    domain: str
    algorithm_code: str


@dataclass(frozen=True)
class FinalRecord:
    domain: str
    class_value: str
    label: int


class ReservoirSampler:
    def __init__(self, size: int, seed: int) -> None:
        self.size = max(0, size)
        self._rng = random.Random(seed)
        self._seen = 0
        self.samples: List[Tuple[str, int, str]] = []

    def consider(self, domain: str, label: int, split_name: str) -> None:
        if self.size == 0:
            return
        self._seen += 1
        row = (domain, label, split_name)
        if len(self.samples) < self.size:
            self.samples.append(row)
            return
        idx = self._rng.randint(1, self._seen)
        if idx <= self.size:
            self.samples[idx - 1] = row


class SampleRecordSampler:
    def __init__(self, size: int, seed: int) -> None:
        self.size = max(0, size)
        self._rng = random.Random(seed)
        self._seen = 0
        self.samples: List[SampleRecord] = []

    def consider(self, domain: str, class_value: str, label: int, feature_values: List[float]) -> None:
        if self.size == 0:
            return
        self._seen += 1
        item = SampleRecord(domain=domain, class_value=class_value, label=label, feature_values=list(feature_values))
        if len(self.samples) < self.size:
            self.samples.append(item)
            return
        idx = self._rng.randint(1, self._seen)
        if idx <= self.size:
            self.samples[idx - 1] = item


class IntegrityJsonWriter:
    def __init__(self, path: Path, metadata: Dict[str, object]) -> None:
        self.path = path
        self._fh = open(path, "w", encoding="utf-8", newline="")
        self._wrote_entry = False
        self._fh.write("{\n")
        self._fh.write('  "metadata": ')
        json.dump(metadata, self._fh, indent=2, sort_keys=True)
        self._fh.write(',\n  "entries": [\n')

    def write_entry(self, domain: str, label: int, split_name: str, entry_sha256: str) -> None:
        payload = {
            "domain": domain,
            "label": label,
            "split": split_name,
            "sha256": entry_sha256,
        }
        if self._wrote_entry:
            self._fh.write(",\n")
        self._fh.write("    ")
        json.dump(payload, self._fh, sort_keys=True)
        self._wrote_entry = True

    def close(self, output_files: List[Dict[str, object]]) -> None:
        self._fh.write("\n  ],\n")
        self._fh.write('  "output_files": ')
        json.dump(output_files, self._fh, indent=2, sort_keys=True)
        self._fh.write("\n}\n")
        self._fh.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, anchor: Path) -> str:
    """Use a relative path for artifacts contained by the dataset workspace."""
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return str(path)


def _extract_tld(domain: str) -> str:
    if "." not in domain:
        return ""
    return domain.rsplit(".", 1)[-1]


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = float(len(value))
    import math

    entropy = 0.0
    for freq in counts.values():
        p = freq / total
        entropy -= p * math.log2(p)
    return entropy


def _digit_ratio(domain: str) -> float:
    if not domain:
        return 0.0
    digits = sum(1 for ch in domain if ch.isdigit())
    return digits / float(len(domain))


def _digit_ratio_bucket(value: float) -> str:
    idx = min(int(value * 10), 10)
    low = idx / 10.0
    high = min((idx + 1) / 10.0, 1.0)
    return f"{low:.1f}-{high:.1f}"


def _entropy_bucket(value: float) -> str:
    idx = min(int(value / 0.5), 20)
    low = idx * 0.5
    high = low + 0.5
    return f"{low:.1f}-{high:.1f}"


def _normalize_domain(raw_line: str) -> str | None:
    domain = raw_line.strip().lower().rstrip(".")
    if not domain:
        return None
    if any(ch.isspace() for ch in domain):
        return None
    if "," in domain:
        return None
    try:
        domain = domain.encode("idna").decode("ascii")
    except Exception:
        return None
    if len(domain) > 253:
        return None
    labels = domain.split(".")
    if any(not item for item in labels):
        return None
    for item in labels:
        if len(item) > 63:
            return None
        if item[0] == "-" or item[-1] == "-":
            return None
        for ch in item:
            if not (ch.isdigit() or ("a" <= ch <= "z") or ch == "-"):
                return None
    return domain


def _normalize_header(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def _is_domain_header_value(value: str) -> bool:
    return _normalize_header(value) == "domain"


def _find_header_key(fieldnames: Sequence[str], expected_normalized_names: set[str]) -> str | None:
    for key in fieldnames:
        if _normalize_header(key) in expected_normalized_names:
            return key
    return None


def _iter_benign_candidates(path: Path, config: BuildConfig) -> Iterator[str]:
    with open(
        path,
        "r",
        encoding=config.encoding,
        errors=config.encoding_errors,
        buffering=config.buffer_size,
        newline="",
    ) as fh:
        reader = csv.reader(fh)
        try:
            first_row = next(reader)
        except StopIteration:
            return

        has_header = any(_is_domain_header_value(value) for value in first_row)
        if has_header:
            fh.seek(0)
            dict_reader = csv.DictReader(fh)
            fieldnames = [item for item in (dict_reader.fieldnames or []) if item is not None]
            domain_key = _find_header_key(fieldnames, {"domain"})
            if domain_key is None:
                raise ValueError(f"Cannot find `domain` column in benign CSV: {path}")
            for row in dict_reader:
                if row is None:
                    yield ""
                    continue
                yield row.get(domain_key, "")
            return

        if first_row:
            first_value = first_row[0]
            if not _is_domain_header_value(first_value):
                yield first_value

        for row in reader:
            if not row:
                yield ""
                continue
            candidate = row[0]
            if _is_domain_header_value(candidate):
                continue
            yield candidate


def _iter_dga_candidates(path: Path, config: BuildConfig) -> Iterator[Tuple[str, str]]:
    algorithm_headers = {
        "algorithmcode",
        "algorithcode",
        "algorithm",
        "algorithmname",
        "algocode",
        "dgafamily",
    }

    with open(
        path,
        "r",
        encoding=config.encoding,
        errors=config.encoding_errors,
        buffering=config.buffer_size,
        newline="",
    ) as fh:
        reader = csv.reader(fh)
        try:
            first_row = next(reader)
        except StopIteration:
            return

        first_row_normalized = {_normalize_header(item) for item in first_row}
        has_header = "domain" in first_row_normalized or bool(first_row_normalized & algorithm_headers)

        if has_header:
            fh.seek(0)
            dict_reader = csv.DictReader(fh)
            fieldnames = [item for item in (dict_reader.fieldnames or []) if item is not None]
            domain_key = _find_header_key(fieldnames, {"domain"})
            if domain_key is None:
                raise ValueError(f"Cannot find `domain` column in fraudulent CSV: {path}")

            algorithm_key = _find_header_key(fieldnames, algorithm_headers)
            if algorithm_key is None:
                raise ValueError(
                    "Cannot find algorithm code column in fraudulent CSV. "
                    "Expected a column like `algorithm_code`."
                )

            for row in dict_reader:
                if row is None:
                    yield "", ""
                    continue
                yield row.get(domain_key, ""), row.get(algorithm_key, "")
            return

        if len(first_row) < 2:
            raise ValueError(
                "Fraudulent CSV without header must contain at least 2 columns: domain and algorithm_code"
            )
        yield first_row[0], first_row[1]
        for row in reader:
            if not row:
                yield "", ""
                continue
            domain_value = row[0] if len(row) >= 1 else ""
            algorithm_value = row[1] if len(row) >= 2 else ""
            yield domain_value, algorithm_value


def _read_fraudulent_records(config: BuildConfig, stats: BuildStats) -> List[FraudRecord]:
    selected: List[FraudRecord] = []
    seen_domains: set[str] = set()

    for raw_domain, raw_algorithm in _iter_dga_candidates(config.dga_input, config):
        stats.dga_rows_read += 1
        domain = _normalize_domain(raw_domain)
        algorithm_code = str(raw_algorithm).strip().lower()

        if domain is None or not algorithm_code:
            stats.empty_or_invalid_rows += 1
            stats.invalid_dga_rows += 1
            continue
        if domain in seen_domains:
            stats.dga_duplicate_rows_dropped += 1
            continue

        seen_domains.add(domain)
        selected.append(FraudRecord(domain=domain, algorithm_code=algorithm_code))

    stats.dga_unique_domains_selected = len(selected)
    return selected


def _read_benign_records(
    config: BuildConfig,
    stats: BuildStats,
    dga_domains: set[str],
    max_rows: int,
) -> List[str]:
    selected: List[str] = []
    seen_domains: set[str] = set()

    for raw_domain in _iter_benign_candidates(config.benign_input, config):
        stats.benign_rows_read += 1
        domain = _normalize_domain(raw_domain)
        if domain is None:
            stats.empty_or_invalid_rows += 1
            stats.invalid_benign_rows += 1
            continue

        if domain in seen_domains:
            stats.benign_duplicate_rows_dropped += 1
            continue
        if domain in dga_domains:
            stats.benign_overlap_excluded += 1
            continue

        seen_domains.add(domain)
        selected.append(domain)
        if len(selected) >= max_rows:
            break

    stats.benign_unique_domains_selected = len(selected)
    return selected


def _deterministic_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _reduce_fraudulent_records_proportionally(
    records: List[FraudRecord],
    target_size: int,
    seed: int,
) -> List[FraudRecord]:
    if target_size >= len(records):
        return list(records)
    if target_size <= 0:
        return []

    grouped: Dict[str, List[FraudRecord]] = {}
    for item in records:
        grouped.setdefault(item.algorithm_code, []).append(item)

    total_records = len(records)
    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for algorithm_code, items in grouped.items():
        expected = (len(items) * target_size) / float(total_records)
        base = int(math.floor(expected))
        allocations[algorithm_code] = min(base, len(items))
        remainders.append((expected - base, algorithm_code))

    remaining = target_size - sum(allocations.values())
    for _, algorithm_code in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocations[algorithm_code] < len(grouped[algorithm_code]):
            allocations[algorithm_code] += 1
            remaining -= 1

    if remaining > 0:
        for algorithm_code in sorted(grouped.keys()):
            if remaining <= 0:
                break
            available = len(grouped[algorithm_code]) - allocations[algorithm_code]
            if available <= 0:
                continue
            take = min(available, remaining)
            allocations[algorithm_code] += take
            remaining -= take

    reduced: List[FraudRecord] = []
    for algorithm_code in sorted(grouped.keys()):
        take_n = allocations.get(algorithm_code, 0)
        if take_n <= 0:
            continue
        ordered = sorted(
            grouped[algorithm_code],
            key=lambda rec: _deterministic_hash(f"{seed}:{algorithm_code}:{rec.domain}"),
        )
        reduced.extend(ordered[:take_n])

    return reduced


def _select_balanced_records(config: BuildConfig, stats: BuildStats) -> List[FinalRecord]:
    fraudulent_records = _read_fraudulent_records(config, stats)
    dga_domains = {item.domain for item in fraudulent_records}
    benign_domains = _read_benign_records(config, stats, dga_domains=dga_domains, max_rows=len(fraudulent_records))

    initial_fraud_count = len(fraudulent_records)
    benign_count = len(benign_domains)
    if benign_count < initial_fraud_count:
        fraudulent_records = _reduce_fraudulent_records_proportionally(
            fraudulent_records,
            target_size=benign_count,
            seed=config.random_seed,
        )
        stats.dga_rows_reduced_for_balance = initial_fraud_count - len(fraudulent_records)
    stats.final_balanced_rows_per_class = min(len(fraudulent_records), benign_count)

    if stats.final_balanced_rows_per_class == 0:
        raise ValueError("Cannot build a balanced dataset: at least one valid domain is required in each class")

    if len(fraudulent_records) > stats.final_balanced_rows_per_class:
        fraudulent_records = fraudulent_records[: stats.final_balanced_rows_per_class]
    if len(benign_domains) > stats.final_balanced_rows_per_class:
        benign_domains = benign_domains[: stats.final_balanced_rows_per_class]

    if len(fraudulent_records) != len(benign_domains):
        raise RuntimeError(
            "Unable to enforce balanced dataset after staged selection: "
            f"dga={len(fraudulent_records)} benign={len(benign_domains)}"
        )

    final_rows: List[FinalRecord] = []
    for item in fraudulent_records:
        final_rows.append(FinalRecord(domain=item.domain, class_value=item.algorithm_code, label=1))
    for domain in benign_domains:
        final_rows.append(FinalRecord(domain=domain, class_value=_extract_tld(domain), label=0))
    return final_rows


def _update_distribution(dst: DistributionStats, domain: str) -> None:
    dst.rows += 1
    dst.length_distribution[str(len(domain))] += 1
    dst.tld_distribution[_extract_tld(domain)] += 1
    dst.digit_ratio_histogram[_digit_ratio_bucket(_digit_ratio(domain))] += 1
    dst.entropy_histogram[_entropy_bucket(_shannon_entropy(domain))] += 1


def _dist_to_json(dst: DistributionStats) -> Dict[str, object]:
    return {
        "rows": dst.rows,
        "length_distribution": dict(sorted(dst.length_distribution.items(), key=lambda kv: int(kv[0]))),
        "tld_distribution": dict(dst.tld_distribution.most_common()),
        "digit_ratio_histogram": dict(sorted(dst.digit_ratio_histogram.items())),
        "entropy_histogram": dict(sorted(dst.entropy_histogram.items())),
    }


def ensure_inputs(config: BuildConfig) -> None:
    benign_alternatives = []
    for filename in (DEFAULT_BENIGN_INPUT.name, DEFAULT_BENIGN_INPUT_ALT.name):
        candidate = config.benign_input.parent / filename
        if candidate != config.benign_input:
            benign_alternatives.append(candidate)
    if not config.benign_input.exists():
        for candidate in benign_alternatives:
            if candidate.exists():
                LOGGER.info(
                    "stage=input_resolve message=Using compatible benign input path requested=%s resolved=%s",
                    config.benign_input,
                    candidate,
                )
                config.benign_input = candidate
                break

    for input_path in (config.benign_input, config.dga_input):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")


def _output_names(config: BuildConfig) -> List[str]:
    names = [config.base_name]
    if config.split_datasets:
        names = [f"{config.base_name}_train", f"{config.base_name}_val", f"{config.base_name}_test"]
    return names


def ensure_output_paths(config: BuildConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    if config.temp_dir is not None:
        config.temp_dir.mkdir(parents=True, exist_ok=True)

    existing: List[Path] = []
    for name in _output_names(config):
        existing.extend(config.output_dir.glob(f"{name}*.parquet"))

    manifest_path = config.metadata_dir / f"{config.base_name}_manifest.json"
    stats_path = config.metadata_dir / config.stats_json_name
    integrity_path = config.metadata_dir / config.integrity_json_name
    sample_path = config.output_dir / SAMPLE_FILE_NAME
    schema_desc_path = config.metadata_dir / config.schema_description_name
    audit_paths = [
        config.metadata_dir / "benign_label_audit_sample.csv",
        config.metadata_dir / "dga_label_audit_sample.csv",
    ]

    if (
        existing
        or manifest_path.exists()
        or stats_path.exists()
        or integrity_path.exists()
        or sample_path.exists()
        or schema_desc_path.exists()
    ) and not config.overwrite:
        raise FileExistsError(
            f"Output exists in {config.output_dir} for base '{config.base_name}'. "
            "Use --overwrite to replace previous output."
        )

    if config.overwrite:
        for path in existing:
            path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)
        integrity_path.unlink(missing_ok=True)
        sample_path.unlink(missing_ok=True)
        schema_desc_path.unlink(missing_ok=True)
        for item in audit_paths:
            item.unlink(missing_ok=True)


def _ratio_ok(train: float, val: float, test: float) -> bool:
    total = train + val + test
    return train > 0 and val >= 0 and test > 0 and abs(total - 1.0) < 1e-9


def _assign_split(domain: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{domain}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _coerce_feature_values(
    features: Dict[str, object],
    feature_names: List[str],
    stats: BuildStats,
) -> List[float]:
    values: List[float] = []
    for name in feature_names:
        if name not in features:
            stats.feature_missing += 1
            stats.feature_errors += 1
            values.append(0.0)
            continue
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            stats.feature_invalid_numeric += 1
            stats.feature_errors += 1
            values.append(0.0)
            continue
        if not math.isfinite(value):
            stats.feature_invalid_numeric += 1
            stats.feature_errors += 1
            value = 0.0
        values.append(value)
    return values


def _write_audit_samples(output_dir: Path, benign_samples: List[Tuple[str, int, str]], dga_samples: List[Tuple[str, int, str]]) -> None:
    benign_path = output_dir / "benign_label_audit_sample.csv"
    dga_path = output_dir / "dga_label_audit_sample.csv"

    with open(benign_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["domain", "label", "split"])
        for domain, label, split_name in sorted(benign_samples):
            writer.writerow([domain, label, split_name])

    with open(dga_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["domain", "label", "split"])
        for domain, label, split_name in sorted(dga_samples):
            writer.writerow([domain, label, split_name])


def _write_final_sample_arff(
    config: BuildConfig,
    feature_names: List[str],
    sample_records: List[SampleRecord],
) -> Tuple[Path, int]:
    sample_path = config.output_dir / SAMPLE_FILE_NAME
    writer = SampleArffWriter(
        output_path=sample_path,
        relation_name=config.relation_name,
        feature_names=feature_names,
        class_attribute_name=CLASS_COLUMN,
        label_attribute_name=LABEL_COLUMN,
        sample_size=config.sample_size,
        encoding=config.encoding,
        output_errors=config.encoding_errors,
        buffer_size=config.buffer_size,
    )
    try:
        for item in sample_records:
            writer.write_row(format_arff_row(item.domain, item.feature_values, item.class_value, item.label))
    finally:
        writer.close()
    return sample_path, writer.rows_written


def _file_output_records(output_files: List[OutputFileStats], path_anchor: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for part in output_files:
        size = part.path.stat().st_size if part.path.exists() else part.bytes_written
        items.append(
            {
                "path": _portable_path(part.path, path_anchor),
                "sha256": _sha256_file(part.path) if part.path.exists() else None,
                "size_bytes": size,
                "format": "parquet",
                "rows": part.data_rows,
                "class_distribution": dict(part.class_distribution),
            }
        )
    return items


def build_manifest(
    config: BuildConfig,
    stats: BuildStats,
    feature_names: List[str],
    sample_path: Path,
    sample_rows_written: int,
    input_hashes: Dict[str, str],
    generated_files: List[Dict[str, object]],
    schema_description_path: Path,
    started: float,
    finished: float,
) -> Dict[str, object]:
    path_anchor = config.output_dir.parent
    transformations = [
        "dga_first_selection",
        "benign_selection_excluding_dga_overlap",
        "balanced_finalization",
        "selected_feature_extraction_only",
        "parquet_serialization",
        "split_parquet_by_target_size",
    ]
    if config.split_datasets:
        transformations.append("deterministic_train_val_test_split")
    transformations.append("post_build_random_sample_arff")

    return {
        "dataset": {
            "name": config.dataset_name,
            "version": config.dataset_version,
            "base_name": config.base_name,
            "relation_name": config.relation_name,
            "main_export_format": "parquet",
        },
        "generated_at": _utc_now(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "runtime": {
            "started_unix": started,
            "finished_unix": finished,
            "duration_seconds": finished - started,
        },
        "inputs": [
            {"path": _portable_path(config.benign_input, path_anchor), "sha256": input_hashes["benign"]},
            {"path": _portable_path(config.dga_input, path_anchor), "sha256": input_hashes["dga"]},
        ],
        "transformations": transformations,
        "schema": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "columns": full_dataset_columns(feature_names),
            "feature_schema_checksum": feature_schema_checksum(feature_names),
            "reference_schema_path": "selected_feature_subset_default",
            "description_json_path": _portable_path(schema_description_path, path_anchor),
        },
        "counts": {
            "benign_rows_read": stats.benign_rows_read,
            "dga_rows_read": stats.dga_rows_read,
            "invalid_benign_rows": stats.invalid_benign_rows,
            "invalid_dga_rows": stats.invalid_dga_rows,
            "empty_or_invalid_rows": stats.empty_or_invalid_rows,
            "benign_overlap_excluded": stats.benign_overlap_excluded,
            "benign_duplicate_rows_dropped": stats.benign_duplicate_rows_dropped,
            "dga_duplicate_rows_dropped": stats.dga_duplicate_rows_dropped,
            "benign_unique_domains_selected": stats.benign_unique_domains_selected,
            "dga_unique_domains_selected": stats.dga_unique_domains_selected,
            "dga_rows_reduced_for_balance": stats.dga_rows_reduced_for_balance,
            "final_balanced_rows_per_class": stats.final_balanced_rows_per_class,
            "rows_written": stats.rows_written,
            "benign_rows_written": stats.benign_rows_written,
            "dga_rows_written": stats.dga_rows_written,
            "feature_errors": stats.feature_errors,
            "feature_missing": stats.feature_missing,
            "feature_computation_errors": stats.feature_computation_errors,
            "feature_invalid_numeric": stats.feature_invalid_numeric,
        },
        "sample": {
            "path": _portable_path(sample_path, path_anchor),
            "rows_written": sample_rows_written,
            "requested_rows": config.sample_size,
            "strategy": "reservoir_random_without_replacement",
            "selection_universe": "final_eligible_population_before_split_assignment",
            "seed": config.random_seed,
            "format": "arff",
        },
        "split": {
            "enabled": config.split_datasets,
            "ratios": {
                "train": config.train_ratio,
                "val": config.val_ratio,
                "test": config.test_ratio,
            },
            "assignment": "sha256(seed:domain)",
            "random_seed": config.random_seed,
        },
        "outputs": {
            "output_files": generated_files,
            "sample_file": {
                "path": _portable_path(sample_path, path_anchor),
                "rows_written": sample_rows_written,
                "requested_rows": config.sample_size,
                "format": "arff",
            },
            "schema_description": {
                "path": _portable_path(schema_description_path, path_anchor),
                "format": "json",
                "sha256": _sha256_file(schema_description_path) if schema_description_path.exists() else None,
            },
        },
        "execution": {
            "arguments": config.execution_args,
            "platform": sys.platform,
            "random_seed": config.random_seed,
            "rotation_threshold_bytes": config.split_threshold_bytes,
            "parquet_row_group_size": config.parquet_row_group_size,
        },
    }


def build_stats_report(
    config: BuildConfig,
    stats: BuildStats,
    feature_names: List[str],
    distributions: Dict[str, DistributionStats],
    split_counts: Dict[str, int],
    output_files: List[OutputFileStats],
    sample_rows_written: int,
    started: float,
    finished: float,
    status: str = "success",
    failure_reason: Optional[str] = None,
) -> Dict[str, object]:
    parts: List[Dict[str, object]] = []
    total_output_size_bytes = 0
    for part in output_files:
        size_bytes = part.path.stat().st_size if part.path.exists() else part.bytes_written
        total_output_size_bytes += int(size_bytes)
        parts.append(
            {
                "part_name": part.path.stem,
                "rows": part.data_rows,
                "features": len(feature_names),
                "class_distribution": dict(part.class_distribution),
                "output_path": str(part.path),
                "format": "parquet",
                "size_bytes": size_bytes,
                "sha256": _sha256_file(part.path) if part.path.exists() else None,
            }
        )

    report = {
        "status": status,
        "failure_reason": failure_reason,
        "generated_at": _utc_now(),
        "pipeline_version": __version__,
        "global_counts": {
            "benign_rows_read": stats.benign_rows_read,
            "dga_rows_read": stats.dga_rows_read,
            "invalid_benign_rows": stats.invalid_benign_rows,
            "invalid_dga_rows": stats.invalid_dga_rows,
            "benign_overlap_excluded": stats.benign_overlap_excluded,
            "benign_duplicate_rows_dropped": stats.benign_duplicate_rows_dropped,
            "dga_duplicate_rows_dropped": stats.dga_duplicate_rows_dropped,
            "benign_unique_domains_selected": stats.benign_unique_domains_selected,
            "dga_unique_domains_selected": stats.dga_unique_domains_selected,
            "dga_rows_reduced_for_balance": stats.dga_rows_reduced_for_balance,
            "final_balanced_rows_per_class": stats.final_balanced_rows_per_class,
            "rows_written": stats.rows_written,
            "empty_or_invalid_rows": stats.empty_or_invalid_rows,
            "feature_errors": stats.feature_errors,
            "feature_missing": stats.feature_missing,
            "feature_computation_errors": stats.feature_computation_errors,
            "feature_invalid_numeric": stats.feature_invalid_numeric,
        },
        "representativity": {
            "global": _dist_to_json(distributions["global"]),
            "class_0": _dist_to_json(distributions["0"]),
            "class_1": _dist_to_json(distributions["1"]),
        },
        "split": {
            "enabled": config.split_datasets,
            "counts": split_counts,
            "ratios": {
                "train": config.train_ratio,
                "val": config.val_ratio,
                "test": config.test_ratio,
            },
            "random_seed": config.random_seed,
        },
        "sampling": {
            "strategy": "reservoir_random_without_replacement",
            "requested_rows": config.sample_size,
            "actual_rows": sample_rows_written,
            "selection_universe": "final_eligible_population_before_split_assignment",
            "seed": config.random_seed,
        },
        "label_audit": {
            "seed": config.random_seed,
            "sample_size": config.audit_sample_size,
            "benign_sample_file": "benign_label_audit_sample.csv",
            "dga_sample_file": "dga_label_audit_sample.csv",
        },
        "dataset_parameters": {
            "base_name": config.base_name,
            "relation_name": config.relation_name,
            "split_threshold_bytes": config.split_threshold_bytes,
            "schema_reference": "selected_feature_subset",
            "feature_count": len(feature_names),
            "main_export_format": "parquet",
            "parquet_row_group_size": config.parquet_row_group_size,
            "deduplicate": config.deduplicate,
            "deduplicate_global": config.deduplicate_global,
        },
        "runtime": {
            "duration_seconds": finished - started,
            "started_unix": started,
            "finished_unix": finished,
        },
    }
    report["global_stats"] = {
        "total_parts": len(parts),
        "total_rows": stats.rows_written,
        "total_attributes": len(feature_names) + 3,
        "total_features": len(feature_names),
        "class_distribution": {"0": stats.benign_rows_written, "1": stats.dga_rows_written},
        "total_output_size_bytes": total_output_size_bytes,
        "main_export_format": "parquet",
    }
    report["parts"] = parts
    return report


def _parquet_schema(feature_names: List[str]) -> pa.Schema:
    fields = [pa.field(DOMAIN_COLUMN, pa.string(), nullable=False)]
    fields.extend(pa.field(name, pa.float64(), nullable=False) for name in feature_names)
    fields.append(pa.field(CLASS_COLUMN, pa.string(), nullable=False))
    fields.append(pa.field(LABEL_COLUMN, pa.int8(), nullable=False))
    return pa.schema(fields)


def build_arff_dataset(config: BuildConfig) -> Dict[str, object]:
    if config.split_datasets and not _ratio_ok(config.train_ratio, config.val_ratio, config.test_ratio):
        raise ValueError("Invalid split ratios: train/val/test must be >=0 and sum exactly 1.0")

    if config.deduplicate or config.deduplicate_global:
        LOGGER.warning(
            "stage=config message=deduplicate flags are ignored in the staged balanced workflow "
            "because overlap exclusion and uniqueness are always enforced; "
            "--deduplicate and --deduplicate-global are deprecated"
        )

    ensure_inputs(config)
    ensure_output_paths(config)

    stats = BuildStats()
    started = time.time()
    feature_names = selected_feature_names()
    parquet_schema = _parquet_schema(feature_names)

    LOGGER.info("stage=args message=Execution arguments %s", json.dumps(config.execution_args, sort_keys=True))
    input_hashes = {
        "benign": _sha256_file(config.benign_input),
        "dga": _sha256_file(config.dga_input),
    }

    distributions = {
        "global": DistributionStats(),
        "0": DistributionStats(),
        "1": DistributionStats(),
    }
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "full": 0}

    benign_sampler = ReservoirSampler(config.audit_sample_size, config.random_seed)
    dga_sampler = ReservoirSampler(config.audit_sample_size, config.random_seed + 1)
    final_sample_sampler = SampleRecordSampler(config.sample_size, config.random_seed)

    output_files: List[OutputFileStats] = []

    integrity_writer: IntegrityJsonWriter | None = None
    if config.write_integrity:
        integrity_writer = IntegrityJsonWriter(
            path=config.metadata_dir / config.integrity_json_name,
            metadata={
                "dataset_name": config.dataset_name,
                "dataset_version": config.dataset_version,
                "generated_at": _utc_now(),
                "algorithm": "sha256",
                "inputs": [
                    {"path": _portable_path(config.benign_input, config.output_dir.parent), "sha256": input_hashes["benign"]},
                    {"path": _portable_path(config.dga_input, config.output_dir.parent), "sha256": input_hashes["dga"]},
                ],
                "main_export_format": "parquet",
            },
        )

    def make_writer(base_name: str) -> SplitParquetWriter:
        return SplitParquetWriter(
            output_dir=config.output_dir,
            base_name=base_name,
            split_threshold_bytes=config.split_threshold_bytes,
            parquet_schema=parquet_schema,
            row_group_size=config.parquet_row_group_size,
            on_part_closed=None,
        )

    if config.split_datasets:
        writers = {
            "train": make_writer(f"{config.base_name}_train"),
            "val": make_writer(f"{config.base_name}_val"),
            "test": make_writer(f"{config.base_name}_test"),
        }
    else:
        writers = {"full": make_writer(config.base_name)}

    schema_description_path = config.metadata_dir / config.schema_description_name
    sample_rows_written = 0
    sample_path = config.output_dir / SAMPLE_FILE_NAME

    try:
        LOGGER.info("stage=selection message=Selecting DGA rows first")
        final_rows = _select_balanced_records(config, stats)

        stats.cross_label_conflicts = stats.benign_overlap_excluded
        stats.conflict_rows_excluded = stats.benign_overlap_excluded
        stats.duplicate_rows_dropped = stats.benign_duplicate_rows_dropped + stats.dga_duplicate_rows_dropped

        LOGGER.info(
            "stage=selection_summary dga_rows_read=%d dga_unique_selected=%d benign_rows_read=%d benign_overlap_excluded=%d benign_unique_selected=%d dga_rows_reduced_for_balance=%d final_rows_per_class=%d",
            stats.dga_rows_read,
            stats.dga_unique_domains_selected,
            stats.benign_rows_read,
            stats.benign_overlap_excluded,
            stats.benign_unique_domains_selected,
            stats.dga_rows_reduced_for_balance,
            stats.final_balanced_rows_per_class,
        )
        if stats.dga_rows_reduced_for_balance > 0:
            LOGGER.info(
                "stage=balance_adjustment message=Reduced DGA pool to maintain balance reduced_rows=%d",
                stats.dga_rows_reduced_for_balance,
            )

        LOGGER.info(
            "stage=build message=Starting selected-feature extraction and Parquet generation rows=%d features=%d",
            len(final_rows),
            len(feature_names),
        )
        selected_feature_set = set(feature_names)

        benign_written_domains: set[str] = set()
        dga_written_domains: set[str] = set()

        for item in final_rows:
            if config.split_datasets:
                split_name = _assign_split(item.domain, config.random_seed, config.train_ratio, config.val_ratio)
            else:
                split_name = "full"

            try:
                features = compute_domain_features_subset(item.domain, selected_feature_names=selected_feature_set)
                feature_values = _coerce_feature_values(features, feature_names, stats)
            except Exception as exc:
                stats.feature_errors += 1
                stats.feature_computation_errors += 1
                LOGGER.warning("stage=row_error domain=%s error=%s", item.domain, exc)
                continue

            row_dict: Dict[str, object] = {
                DOMAIN_COLUMN: item.domain,
                CLASS_COLUMN: item.class_value,
                LABEL_COLUMN: int(item.label),
            }
            for idx, fname in enumerate(feature_names):
                row_dict[fname] = feature_values[idx]

            writers[split_name].write_row(row_dict, item.label)
            stats.rows_written += 1
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
            if item.label == 0:
                stats.benign_rows_written += 1
                benign_sampler.consider(item.domain, item.label, split_name)
                benign_written_domains.add(item.domain)
            else:
                stats.dga_rows_written += 1
                dga_sampler.consider(item.domain, item.label, split_name)
                dga_written_domains.add(item.domain)

            final_sample_sampler.consider(item.domain, item.class_value, item.label, feature_values)

            _update_distribution(distributions["global"], item.domain)
            _update_distribution(distributions[str(item.label)], item.domain)

            if integrity_writer is not None:
                entry_hash = _sha256_text(f"{item.domain},{item.label},{split_name}")
                integrity_writer.write_entry(item.domain, item.label, split_name, entry_hash)

            if config.progress_every > 0 and stats.rows_written % config.progress_every == 0:
                LOGGER.info(
                    "stage=progress rows_written=%d benign_rows_written=%d dga_rows_written=%d feature_errors=%d",
                    stats.rows_written,
                    stats.benign_rows_written,
                    stats.dga_rows_written,
                    stats.feature_errors,
                )

        for writer in writers.values():
            output_files.extend(writer.finalize())

        if stats.benign_rows_written != stats.dga_rows_written:
            raise RuntimeError(
                f"Final dataset is not balanced: benign={stats.benign_rows_written} dga={stats.dga_rows_written}"
            )
        if benign_written_domains & dga_written_domains:
            raise RuntimeError("Final dataset has DOMAIN overlap between benign and DGA classes")

        LOGGER.info(
            "stage=sample message=Generating final ARFF sample at end requested_rows=%d selection_universe=final_eligible_population_before_split_assignment seed=%d",
            config.sample_size,
            config.random_seed,
        )
        sample_path, sample_rows_written = _write_final_sample_arff(config, feature_names, final_sample_sampler.samples)

        if config.write_schema_description:
            write_field_descriptions(schema_description_path, feature_names)
            LOGGER.info("stage=schema_description message=Wrote dataset field description json path=%s", schema_description_path)

        if config.validate_output:
            base_names = [config.base_name]
            if config.split_datasets:
                base_names = [f"{config.base_name}_train", f"{config.base_name}_val", f"{config.base_name}_test"]
            for base in base_names:
                validation = validate_outputs(
                    output_dir=config.output_dir,
                    metadata_dir=config.metadata_dir,
                    base_name=base,
                    split_threshold_bytes=config.split_threshold_bytes,
                    encoding=config.encoding,
                    sample_file_name=SAMPLE_FILE_NAME,
                    require_sample=False,
                    expected_format="parquet",
                    expected_columns=full_dataset_columns(feature_names),
                    require_balanced=not config.split_datasets,
                )
                if not validation.valid:
                    raise RuntimeError(f"Output validation failed for {base}: " + "; ".join(validation.errors))

            sample_validation_base = config.base_name
            if config.split_datasets:
                sample_validation_base = f"{config.base_name}_train"
            sample_validation = validate_outputs(
                output_dir=config.output_dir,
                metadata_dir=config.metadata_dir,
                base_name=sample_validation_base,
                split_threshold_bytes=config.split_threshold_bytes,
                encoding=config.encoding,
                sample_file_name=SAMPLE_FILE_NAME,
                require_sample=True,
                expected_format="parquet",
                expected_columns=full_dataset_columns(feature_names),
                require_balanced=not config.split_datasets,
            )
            if not sample_validation.valid:
                raise RuntimeError("Sample validation failed: " + "; ".join(sample_validation.errors))

        _write_audit_samples(config.metadata_dir, benign_sampler.samples, dga_sampler.samples)
        generated_files = _file_output_records(output_files, config.output_dir.parent)

        if integrity_writer is not None:
            extra_files = list(generated_files)
            if sample_path.exists():
                extra_files.append(
                    {
                        "path": _portable_path(sample_path, config.output_dir.parent),
                        "sha256": _sha256_file(sample_path),
                        "size_bytes": sample_path.stat().st_size,
                        "format": "arff",
                    }
                )
            if schema_description_path.exists():
                extra_files.append(
                    {
                        "path": _portable_path(schema_description_path, config.output_dir.parent),
                        "sha256": _sha256_file(schema_description_path),
                        "size_bytes": schema_description_path.stat().st_size,
                        "format": "json",
                    }
                )
            integrity_writer.close(extra_files)

        finished = time.time()

        manifest = build_manifest(
            config=config,
            stats=stats,
            feature_names=feature_names,
            sample_path=sample_path,
            sample_rows_written=sample_rows_written,
            input_hashes=input_hashes,
            generated_files=generated_files,
            schema_description_path=schema_description_path,
            started=started,
            finished=finished,
        )

        stats_report = build_stats_report(
            config=config,
            stats=stats,
            feature_names=feature_names,
            distributions=distributions,
            split_counts=split_counts,
            output_files=output_files,
            sample_rows_written=sample_rows_written,
            started=started,
            finished=finished,
        )

        if config.write_stats:
            stats_path = config.metadata_dir / config.stats_json_name
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(stats_report, fh, indent=2, sort_keys=True)
            LOGGER.info("stage=stats message=Wrote dataset stats json path=%s", stats_path)

        if config.write_manifest or config.manifest:
            manifest_path = config.metadata_dir / f"{config.base_name}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            LOGGER.info("stage=manifest message=Wrote dataset manifest path=%s", manifest_path)

        LOGGER.info(
            "stage=complete rows_written=%d output_files=%d runtime_seconds=%.2f",
            stats.rows_written,
            len(output_files),
            finished - started,
        )
        LOGGER.info(
            "stage=counts benign_rows_read=%d dga_rows_read=%d benign_rows_written=%d dga_rows_written=%d benign_overlap_excluded=%d dga_reduced_for_balance=%d invalid_benign_rows=%d invalid_dga_rows=%d sample_rows_written=%d balanced=true",
            stats.benign_rows_read,
            stats.dga_rows_read,
            stats.benign_rows_written,
            stats.dga_rows_written,
            stats.benign_overlap_excluded,
            stats.dga_rows_reduced_for_balance,
            stats.invalid_benign_rows,
            stats.invalid_dga_rows,
            sample_rows_written,
        )
        return manifest
    except Exception as exc:
        finished = time.time()
        failed_report = build_stats_report(
            config=config,
            stats=stats,
            feature_names=feature_names,
            distributions=distributions,
            split_counts=split_counts,
            output_files=output_files,
            sample_rows_written=sample_rows_written,
            started=started,
            finished=finished,
            status="failed",
            failure_reason=str(exc),
        )
        if config.write_stats:
            stats_path = config.metadata_dir / config.stats_json_name
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(failed_report, fh, indent=2, sort_keys=True)
        if integrity_writer is not None:
            integrity_writer.close([])
        raise
