from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .logging_utils import configure_logging
from .paths import (
    DEFAULT_BENIGN_INPUT,
    DEFAULT_DGA_INPUT,
    DEFAULT_LOG_DIR,
    DEFAULT_METADATA_DIR,
    DEFAULT_OUTPUT_DIR,
)
from .pipeline import BuildConfig, build_arff_dataset
from .schema import FIELD_DESCRIPTION_FILE_NAME, SAMPLE_FILE_NAME, full_dataset_columns, selected_feature_names
from .validator import validate_outputs

DEFAULT_BASE_NAME = "udcdga_dataset"
DEFAULT_SPLIT_THRESHOLD = 1_900_000_000
DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_PROGRESS_EVERY = 100_000
DEFAULT_SCHEMA_REFERENCE = Path("domains_output.arff")
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_AUDIT_SAMPLE_SIZE = 100
DEFAULT_STATS_JSON_NAME = "udcdga_dataset_stats.json"
DEFAULT_INTEGRITY_JSON_NAME = "udcdga_dataset_integrity.json"
DEFAULT_ROW_GROUP_SIZE = 10_000


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer")
    return parsed


def ratio_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("Ratio must be between 0.0 and 1.0")
    return parsed


def _add_shared_build_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--benign-input", type=Path, default=DEFAULT_BENIGN_INPUT)
    cmd.add_argument("--dga-input", type=Path, default=DEFAULT_DGA_INPUT)
    cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    cmd.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Directory for metadata artifacts (manifest, integrity, stats, schema description).",
    )
    cmd.add_argument("--base-name", default=DEFAULT_BASE_NAME)
    cmd.add_argument("--split-threshold-bytes", type=positive_int, default=DEFAULT_SPLIT_THRESHOLD)
    cmd.add_argument("--encoding", default="utf-8")
    cmd.add_argument("--encoding-errors", choices=["strict", "replace", "ignore"], default="replace")
    cmd.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    cmd.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    cmd.add_argument("--buffer-size", type=positive_int, default=DEFAULT_BUFFER_SIZE)
    cmd.add_argument("--threads", type=positive_int, default=1)
    cmd.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    cmd.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deprecated flag kept for CLI compatibility; staged balanced workflow already enforces uniqueness.",
    )
    cmd.add_argument(
        "--deduplicate-global",
        action="store_true",
        help="Deprecated flag kept for CLI compatibility; staged balanced workflow already excludes cross-label overlap.",
    )
    cmd.add_argument("--overwrite", action="store_true")
    cmd.add_argument("--manifest", action="store_true", help="Backward-compatible alias to write manifest.")
    cmd.add_argument("--write-manifest", action="store_true")
    cmd.add_argument("--write-integrity", action="store_true")
    cmd.add_argument("--write-stats", action=argparse.BooleanOptionalAction, default=True)
    cmd.add_argument("--write-schema-description", action=argparse.BooleanOptionalAction, default=True)
    cmd.add_argument("--progress-every", type=non_negative_int, default=DEFAULT_PROGRESS_EVERY)
    cmd.add_argument("--validate-output", action="store_true")
    cmd.add_argument("--temp-dir", type=Path, default=None)
    cmd.add_argument("--relation-name", default="UDCDGA")
    cmd.add_argument("--schema-reference", type=Path, default=DEFAULT_SCHEMA_REFERENCE)
    cmd.add_argument("--sample-size", type=positive_int, default=DEFAULT_SAMPLE_SIZE)
    cmd.add_argument("--audit-sample-size", type=positive_int, default=DEFAULT_AUDIT_SAMPLE_SIZE)
    cmd.add_argument("--stats-json-name", default=DEFAULT_STATS_JSON_NAME)
    cmd.add_argument("--integrity-json-name", default=DEFAULT_INTEGRITY_JSON_NAME)
    cmd.add_argument("--schema-description-name", default=FIELD_DESCRIPTION_FILE_NAME)
    cmd.add_argument("--split-datasets", action="store_true")
    cmd.add_argument("--train-ratio", type=ratio_float, default=0.8)
    cmd.add_argument("--val-ratio", type=ratio_float, default=0.1)
    cmd.add_argument("--test-ratio", type=ratio_float, default=0.1)
    cmd.add_argument("--random-seed", type=int, default=42)
    cmd.add_argument("--dataset-name", default="udcdga")
    cmd.add_argument("--dataset-version", default="1.0.0")
    cmd.add_argument("--parquet-row-group-size", type=positive_int, default=DEFAULT_ROW_GROUP_SIZE)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dga-features-extractor",
        description="Construct a balanced DGA dataset with feature extraction and Parquet export.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser(
        "build-dataset",
        help="Build Parquet dataset from benign and DGA domain files (recommended).",
    )
    _add_shared_build_args(build_cmd)

    legacy_build_cmd = subparsers.add_parser(
        "build-arff",
        help="Legacy alias for build-dataset. Main outputs are now Parquet; ARFF is generated only for the final sample.",
    )
    _add_shared_build_args(legacy_build_cmd)

    validate_cmd = subparsers.add_parser(
        "validate-dataset",
        help="Validate generated Parquet outputs and final ARFF sample.",
    )
    validate_cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate_cmd.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    validate_cmd.add_argument("--base-name", default=DEFAULT_BASE_NAME)
    validate_cmd.add_argument("--split-threshold-bytes", type=positive_int, default=DEFAULT_SPLIT_THRESHOLD)
    validate_cmd.add_argument("--encoding", default="utf-8")
    validate_cmd.add_argument("--sample-file-name", default=SAMPLE_FILE_NAME)
    validate_cmd.add_argument("--schema-reference", type=Path, default=DEFAULT_SCHEMA_REFERENCE)

    legacy_validate_cmd = subparsers.add_parser(
        "validate-arff",
        help="Legacy alias for validate-dataset.",
    )
    legacy_validate_cmd.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    legacy_validate_cmd.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    legacy_validate_cmd.add_argument("--base-name", default=DEFAULT_BASE_NAME)
    legacy_validate_cmd.add_argument("--split-threshold-bytes", type=positive_int, default=DEFAULT_SPLIT_THRESHOLD)
    legacy_validate_cmd.add_argument("--encoding", default="utf-8")
    legacy_validate_cmd.add_argument("--sample-file-name", default=SAMPLE_FILE_NAME)
    legacy_validate_cmd.add_argument("--schema-reference", type=Path, default=DEFAULT_SCHEMA_REFERENCE)

    return parser


def run_build(args: argparse.Namespace) -> int:
    configure_logging(args.log_dir, args.log_level, args.base_name)
    config = BuildConfig(
        benign_input=args.benign_input,
        dga_input=args.dga_input,
        output_dir=args.output_dir,
        base_name=args.base_name,
        metadata_dir=args.metadata_dir,
        split_threshold_bytes=args.split_threshold_bytes,
        encoding=args.encoding,
        encoding_errors=args.encoding_errors,
        log_dir=args.log_dir,
        log_level=args.log_level,
        buffer_size=args.buffer_size,
        deduplicate=args.deduplicate,
        threads=args.threads,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
        overwrite=args.overwrite,
        relation_name=args.relation_name,
        manifest=args.manifest,
        progress_every=args.progress_every,
        validate_output=args.validate_output,
        schema_reference=args.schema_reference,
        compression_enabled=False,
        compression_format="none",
        keep_uncompressed_full_parts=True,
        sample_size=args.sample_size,
        stats_json_name=args.stats_json_name,
        write_manifest=args.write_manifest,
        write_integrity=args.write_integrity,
        write_stats=args.write_stats,
        deduplicate_global=args.deduplicate_global,
        split_datasets=args.split_datasets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        integrity_json_name=args.integrity_json_name,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        audit_sample_size=args.audit_sample_size,
        execution_args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        parquet_row_group_size=args.parquet_row_group_size,
        write_schema_description=args.write_schema_description,
        schema_description_name=args.schema_description_name,
        output_format="parquet",
    )

    try:
        manifest = build_arff_dataset(config)
    except Exception as exc:
        logging.getLogger(__name__).error("stage=failed error=%s", exc)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_validate(args: argparse.Namespace) -> int:
    feature_names = selected_feature_names()
    result = validate_outputs(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        base_name=args.base_name,
        split_threshold_bytes=args.split_threshold_bytes,
        encoding=args.encoding,
        sample_file_name=args.sample_file_name,
        require_sample=True,
        expected_format="parquet",
        expected_columns=full_dataset_columns(feature_names),
    )
    print(json.dumps(result.summary, indent=2))
    if not result.valid:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"build-dataset", "build-arff"}:
        return run_build(args)
    if args.command in {"validate-dataset", "validate-arff"}:
        return run_validate(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
