from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    AppConfig,
    DEFAULT_GENERATION_MODE,
    ORDERED_CAPPED_GENERATION_MODE,
    ensure_dirs,
)
from .generator import run_pipeline
from .logging_utils import setup_logging
from .utils import read_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dga_fraudulents_dataset",
        description="Discover and execute DGA implementations and build a deduplicated domain dataset.",
    )
    p.add_argument("--algorithms-root", type=Path, default=Path("dga_algorithms"), help="Root directory containing one subdirectory per DGA algorithm.")
    p.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory for CSV, JSON stats, plan and state files.")
    p.add_argument("--target-count", type=int, default=15_000_000, help="Target number of unique domains to generate.")
    p.add_argument(
        "--generation-mode",
        default=DEFAULT_GENERATION_MODE,
        choices=[DEFAULT_GENERATION_MODE, ORDERED_CAPPED_GENERATION_MODE],
        help="Generation policy. ordered_capped runs a strict fixed algorithm order with per-algorithm unique caps.",
    )
    p.add_argument(
        "--per-algorithm-cap",
        type=int,
        default=200_000,
        help="Hard per-algorithm cap for ordered_capped mode.",
    )
    p.add_argument(
        "--ordered-algorithms",
        default=None,
        help="Optional comma-separated ordered list used by ordered_capped mode.",
    )
    p.add_argument("--threads", type=int, default=1, help="Worker threads/processes hint (current implementation uses single-process adapters).")
    p.add_argument("--batch-size", type=int, default=5_000, help="Domains requested per algorithm generation call.")
    p.add_argument("--checkpoint-every", type=int, default=100_000, help="Commit and save checkpoint state every N inserted unique domains.")
    p.add_argument("--plan-file", type=Path, default=None, help="Optional custom path for the generation plan JSON.")
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config file; CLI flags override file values.")
    p.add_argument("--resume", action="store_true", help="Resume from state JSON + SQLite DB in output directory.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity level.")
    p.add_argument("--dedup-backend", default="sqlite", choices=["sqlite"], help="Dedup backend implementation.")

    p.add_argument("--seed-strategy", default="sequential", choices=["sequential", "hashed_round_robin", "fixed"], help="Seed exploration strategy.")
    p.add_argument("--date-strategy", default="daily_forward", choices=["daily_forward", "daily_window", "monthly_window", "fixed"], help="Date exploration strategy.")

    p.add_argument("--min-unique-yield-ratio", type=float, default=0.01, help="Minimum per-batch unique yield ratio before counting as low-yield.")
    p.add_argument("--discard-after-consecutive-empty", type=int, default=2, help="Discard algorithm after this many consecutive empty batches.")
    p.add_argument("--discard-after-consecutive-low-yield", type=int, default=5, help="Discard algorithm after this many consecutive low-yield rounds.")
    p.add_argument("--low-yield-grace-rounds", type=int, default=2, help="Grace rounds before degrading/discarding low-yield algorithms.")
    p.add_argument("--max-algorithm-errors", type=int, default=5, help="Discard algorithm after this many generation errors.")
    p.add_argument("--algorithm-timeout-seconds", type=int, default=10, help="Timeout per CLI/subprocess invocation.")
    p.add_argument("--algorithm-batch-timeout-seconds", type=int, default=30, help="Wall-clock timeout per algorithm batch execution.")
    p.add_argument("--max-cli-invocations-per-batch", type=int, default=40, help="Maximum subprocess invocations for scalar CLI algorithms per batch.")
    p.add_argument("--heartbeat-seconds", type=int, default=15, help="Heartbeat interval in seconds for progress logs.")
    p.add_argument("--saturation-window", type=int, default=8, help="Rolling window size used for saturation detection.")
    p.add_argument("--saturation-min-yield", type=float, default=0.03, help="Minimum rolling yield to avoid saturation.")
    p.add_argument("--exhausted-after-zero-unique-batches", type=int, default=4, help="Mark algorithm exhausted after N consecutive zero-unique batches.")
    p.add_argument("--near-quota-exhaustion-margin", type=int, default=200, help="Remaining quota threshold to enable near-quota probe logic.")
    p.add_argument("--near-quota-max-retries", type=int, default=4, help="Maximum near-quota retries before exhaustion.")
    p.add_argument("--date-start", default="2018-01-01", help="Lower date bound for date exploration (YYYY-MM-DD).")
    p.add_argument("--date-end", default="2030-12-31", help="Upper date bound for date exploration (YYYY-MM-DD).")
    p.add_argument("--date-max-years-forward", type=int, default=8, help="Max years forward from base date before wrapping/clamping.")
    p.add_argument("--date-max-years-backward", type=int, default=8, help="Max years backward from base date before wrapping/clamping.")
    p.add_argument("--date-wrap-policy", choices=["clamp", "wrap", "reset-cycle"], default="clamp", help="Policy applied when requested dates exceed configured bounds.")
    p.add_argument("--max-effective-quota-multiplier", type=float, default=4.0, help="Cap for runtime effective_quota as multiplier of planned_quota.")
    p.add_argument("--redistribution-capacity-threshold", type=float, default=0.2, help="Minimum capacity_score required to absorb redistributed quota.")

    p.add_argument("--dry-run", action="store_true", help="Run discovery+inspection+planning and a small generation sample only.")
    p.add_argument("--validate-only", action="store_true", help="Validate an existing dedup DB count against --target-count without generating new data.")
    return p


def _parse_ordered_algorithms(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    out: list[str] = []
    for item in value.split(","):
        code = item.strip().lower()
        if code:
            out.append(code)
    return out


def _explicit_cli_destinations(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_destinations = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    return {
        option_destinations[token.split("=", 1)[0]]
        for token in argv
        if token.startswith("-") and token.split("=", 1)[0] in option_destinations
    }


def _merge_config(args: argparse.Namespace, explicit_cli: set[str] | None = None) -> argparse.Namespace:
    if not args.config:
        return args
    explicit_cli = explicit_cli or set()
    payload = read_json(args.config)
    for k, v in payload.items():
        if hasattr(args, k) and k not in explicit_cli:
            setattr(args, k, v)
    return args


def args_to_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        algorithms_root=Path(args.algorithms_root),
        output_dir=Path(args.output_dir),
        target_count=args.target_count,
        generation_mode=args.generation_mode,
        per_algorithm_cap=args.per_algorithm_cap,
        ordered_algorithms=_parse_ordered_algorithms(args.ordered_algorithms),
        threads=args.threads,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        plan_file=Path(args.plan_file) if args.plan_file is not None else None,
        config_file=Path(args.config) if args.config is not None else None,
        resume=args.resume,
        log_level=args.log_level,
        dedup_backend=args.dedup_backend,
        seed_strategy=args.seed_strategy,
        date_strategy=args.date_strategy,
        min_unique_yield_ratio=args.min_unique_yield_ratio,
        discard_after_consecutive_empty=args.discard_after_consecutive_empty,
        discard_after_consecutive_low_yield=args.discard_after_consecutive_low_yield,
        low_yield_grace_rounds=args.low_yield_grace_rounds,
        max_algorithm_errors=args.max_algorithm_errors,
        algorithm_timeout_seconds=args.algorithm_timeout_seconds,
        algorithm_batch_timeout_seconds=args.algorithm_batch_timeout_seconds,
        max_cli_invocations_per_batch=args.max_cli_invocations_per_batch,
        heartbeat_seconds=args.heartbeat_seconds,
        saturation_window=args.saturation_window,
        saturation_min_yield=args.saturation_min_yield,
        exhausted_after_zero_unique_batches=args.exhausted_after_zero_unique_batches,
        near_quota_exhaustion_margin=args.near_quota_exhaustion_margin,
        near_quota_max_retries=args.near_quota_max_retries,
        date_start=args.date_start,
        date_end=args.date_end,
        date_max_years_forward=args.date_max_years_forward,
        date_max_years_backward=args.date_max_years_backward,
        date_wrap_policy=args.date_wrap_policy,
        max_effective_quota_multiplier=args.max_effective_quota_multiplier,
        redistribution_capacity_threshold=args.redistribution_capacity_threshold,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    args = _merge_config(args, _explicit_cli_destinations(parser, raw_args))
    cfg = args_to_config(args)
    ensure_dirs(cfg)

    setup_logging(cfg.log_level, cfg.output_dir / "logs" / "dga_fraudulents_dataset.log")
    logger = logging.getLogger(__name__)
    logger.info("Starting dga_fraudulents_dataset pipeline")

    payload = run_pipeline(cfg)
    logger.info(
        "Finished. unique_domains=%s target_domains=%s",
        payload.get("unique_domains"),
        payload.get("target_domains"),
    )
    return 0
