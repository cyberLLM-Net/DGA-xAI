from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import BuildError, build_dataset
from .logging_utils import setup_logging
from .models import BuildConfig, ParseConfig

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_FILENAME = Path("udcdga_benigns_domains.csv")
DEFAULT_STATS_FILENAME = Path("udcdga_benigns_domains_stats.json")
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / DEFAULT_OUTPUT_FILENAME
DEFAULT_FINAL_BENIGN_DOMAINS_STATS = DEFAULT_RESULTS_DIR / DEFAULT_STATS_FILENAME
DEFAULT_TARGET_SIZE = 15_000_000
DEFAULT_COUNT_WORKERS = 8
DEFAULT_COUNT_LOG_EVERY = 500_000
DEFAULT_COUNT_CACHE_FILENAME = "udcdga_benigns_count_cache.json"


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse controls entry here.
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer greater than 0")
    return parsed


def _target_size_was_provided(raw_args: list[str]) -> bool:
    return any(arg == "--target-size" or arg.startswith("--target-size=") for arg in raw_args)


def resolve_output_paths(output: Path, report: Path, results_dir: Path) -> tuple[Path, Path]:
    resolved_output = output if output.is_absolute() else (results_dir / output)
    resolved_report = report if report.is_absolute() else (results_dir / report)
    return resolved_output, resolved_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="domain_aggregator",
        description="Proportionally aggregate benign domain files into a single CSV.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build proportional final dataset")

    build.add_argument("--input-dir", type=Path, required=True, help="Input directory containing .txt/.csv files")
    build.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output CSV path (relative paths are resolved inside --results-dir)",
    )
    build.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_STATS_FILENAME,
        help="JSON stats output path (relative paths are resolved inside --results-dir)",
    )
    build.add_argument(
        "--target-size",
        type=_parse_positive_int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Final output row count target (positive integer, default: {DEFAULT_TARGET_SIZE})",
    )
    build.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    build.add_argument(
        "--count-workers",
        "--threads",
        dest="count_workers",
        type=_parse_positive_int,
        default=DEFAULT_COUNT_WORKERS,
        help=f"Worker count for per-file counting (default: {DEFAULT_COUNT_WORKERS})",
    )
    build.add_argument(
        "--count-log-every",
        type=_parse_positive_int,
        default=DEFAULT_COUNT_LOG_EVERY,
        help=(
            f"Deprecated compatibility option (default: {DEFAULT_COUNT_LOG_EVERY}); "
            "fine-grained intra-file progress logs are disabled"
        ),
    )
    build.add_argument(
        "--use-count-cache",
        action="store_true",
        help="Reuse cached count results when file size+mtime and counting settings match",
    )

    build.add_argument("--delimiter", default=None, help="Force CSV delimiter (default: auto-detect)")
    build.add_argument("--domain-column", default=None, help="Domain column name in CSV files")
    build.add_argument(
        "--header-auto-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable header auto-detection",
    )
    build.add_argument(
        "--validation-strictness",
        choices=["lenient", "balanced", "strict"],
        default="balanced",
        help="Domain validator strictness",
    )

    build.add_argument("--deduplicate-per-file", action="store_true", help="Deduplicate each file before sampling")
    build.add_argument("--deduplicate-final", action="store_true", help="Deduplicate final output globally")
    build.add_argument("--no-replacement", action="store_true", help="Disallow sampling with replacement")
    build.add_argument(
        "--strict-no-refill",
        action="store_true",
        help="Allow final row shortfall instead of refill when dedup/no-replacement prevent reaching target",
    )

    build.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Base directory for relative output/report paths (default: {DEFAULT_RESULTS_DIR})",
    )
    build.add_argument("--logs-dir", type=Path, default=Path("logs"), help="Directory for log file")

    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    default_target_size_used = not _target_size_was_provided(raw_args)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.command != "build":
        parser.error("Unknown command")

    output_path, report_path = resolve_output_paths(args.output, args.report, args.results_dir)
    log_path = args.logs_dir / "domain_aggregation.log"
    count_cache_path = report_path.parent / DEFAULT_COUNT_CACHE_FILENAME if args.use_count_cache else None

    parse_cfg = ParseConfig(
        delimiter=args.delimiter,
        domain_column=args.domain_column,
        header_auto_detect=args.header_auto_detect,
        validation_strictness=args.validation_strictness,
    )
    config = BuildConfig(
        input_dir=args.input_dir,
        output_csv=output_path,
        report_path=report_path,
        target_size=args.target_size,
        default_target_size_used=default_target_size_used,
        seed=args.seed,
        threads=args.count_workers,
        count_workers=args.count_workers,
        count_log_every=args.count_log_every,
        use_count_cache=args.use_count_cache,
        count_cache_path=count_cache_path,
        parse=parse_cfg,
        deduplicate_per_file=args.deduplicate_per_file,
        deduplicate_final=args.deduplicate_final,
        allow_replacement=not args.no_replacement,
        strict_no_refill=args.strict_no_refill,
        log_path=log_path,
    )

    logger = setup_logging(config.log_path)
    logger.info("Starting build with target size %d", config.target_size)
    try:
        build_dataset(config, logger)
    except (BuildError, FileNotFoundError, ValueError) as exc:
        logger.error("Build failed: %s", exc)
        return 1
    return 0
