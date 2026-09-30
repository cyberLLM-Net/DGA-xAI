from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .counting import COUNTING_MODE_RAW_ROWS, count_all_files
from .discovery import discover_input_files
from .domain import is_valid_domain
from .models import BuildConfig, FileStats, RunReport
from .quota import allocate_quotas_from_raw_rows
from .reporting import write_report
from .sampling import iter_randomized_domain_candidates, refill_candidate_pools, stable_file_seed
from .writer import open_csv_writer


class BuildError(RuntimeError):
    """Raised when dataset generation cannot satisfy constraints."""


@dataclass
class FileGenerationState:
    stats: FileStats
    iterator: Iterator[str]
    remaining_quota: int
    seen_per_file: set[str] | None
    exhausted: bool = False
    start_logged: bool = False


DEFAULT_COUNT_CACHE_FILENAME = "udcdga_benigns_count_cache.json"


def _compute_acceptance_rate(accepted: int, rejected: int) -> float:
    total = accepted + rejected
    if total <= 0:
        return 0.0
    return (accepted / total) * 100.0


def generate_from_file_until_quota(
    state: FileGenerationState,
    config: BuildConfig,
    csv_writer: object,
    final_seen: set[str] | None,
    quota_to_fill: int,
) -> tuple[int, int, int]:
    if quota_to_fill <= 0 or state.exhausted:
        return 0, 0, 0

    accepted = 0
    rejected = 0
    duplicates_removed_final = 0
    started_at = time.perf_counter()

    while accepted < quota_to_fill and not state.exhausted:
        try:
            candidate = next(state.iterator)
        except StopIteration:
            state.exhausted = True
            break

        state.stats.generation_attempted_rows += 1

        if not is_valid_domain(candidate, strictness=config.parse.validation_strictness):
            rejected += 1
            continue

        if state.seen_per_file is not None:
            if candidate in state.seen_per_file:
                rejected += 1
                state.stats.duplicates_removed_per_file += 1
                continue
            state.seen_per_file.add(candidate)

        if final_seen is not None and candidate in final_seen:
            rejected += 1
            duplicates_removed_final += 1
            continue

        if final_seen is not None:
            final_seen.add(candidate)
        csv_writer.writerow([candidate])
        accepted += 1

    state.stats.accepted_domains += accepted
    state.stats.rejected_domains += rejected
    state.stats.valid_rows = state.stats.accepted_domains
    state.stats.invalid_rows = state.stats.rejected_domains
    state.stats.emitted_rows = state.stats.accepted_domains
    state.stats.sampled_rows_before_final_dedup = state.stats.generation_attempted_rows
    state.stats.acceptance_rate = _compute_acceptance_rate(state.stats.accepted_domains, state.stats.rejected_domains)
    state.stats.generation_elapsed_seconds += time.perf_counter() - started_at
    state.stats.duplicates_removed_final += duplicates_removed_final

    return accepted, rejected, duplicates_removed_final


def redistribute_remaining_deficit(deficit: int, receivers: list[FileStats]) -> list[tuple[Path, int]]:
    if deficit <= 0 or not receivers:
        return []

    weights = [max(0, receiver.raw_rows_seen) for receiver in receivers]
    if sum(weights) <= 0:
        weights = [1] * len(receivers)

    allocation = allocate_quotas_from_raw_rows(weights, deficit)
    return [
        (receiver.path, quota)
        for receiver, quota in zip(receivers, allocation.quotas)
        if quota > 0
    ]


def _log_generation_file_start(logger: logging.Logger, state: FileGenerationState) -> None:
    logger.info(
        "stage=generate file=%s status=start assigned_quota=%d",
        state.stats.path.name,
        state.stats.effective_quota,
    )


def _log_generation_file_done(logger: logging.Logger, stat: FileStats) -> None:
    deficit = max(0, stat.effective_quota - stat.accepted_domains)
    logger.info(
        (
            "stage=generate file=%s status=done assigned_quota=%d accepted=%d rejected=%d "
            "exhausted=%s deficit=%d elapsed_sec=%.2f acceptance_rate=%.2f"
        ),
        stat.path.name,
        stat.effective_quota,
        stat.accepted_domains,
        stat.rejected_domains,
        str(stat.exhausted_before_quota).lower(),
        deficit,
        stat.generation_elapsed_seconds,
        stat.acceptance_rate,
    )


def _finalize_file_stats(stats: list[FileStats]) -> None:
    for stat in stats:
        stat.acceptance_rate = _compute_acceptance_rate(stat.accepted_domains, stat.rejected_domains)
        stat.valid_rows = stat.accepted_domains
        stat.invalid_rows = stat.rejected_domains
        stat.emitted_rows = stat.accepted_domains
        stat.sampled_rows_before_final_dedup = stat.generation_attempted_rows


def build_dataset(config: BuildConfig, logger: logging.Logger) -> RunReport:
    if config.target_size <= 0:
        raise BuildError("target_size must be greater than zero")

    build_started_at = time.perf_counter()
    files = discover_input_files(config.input_dir)
    effective_count_workers = config.count_workers if config.count_workers is not None else config.threads
    count_cache_path = (
        config.count_cache_path
        if config.count_cache_path is not None
        else (config.report_path.parent / DEFAULT_COUNT_CACHE_FILENAME if config.use_count_cache else None)
    )

    stats, count_summary = count_all_files(
        files=files,
        parse_config=config.parse,
        deduplicate_per_file=config.deduplicate_per_file,
        count_workers=effective_count_workers,
        count_log_every=config.count_log_every,
        logger=logger,
        use_count_cache=config.use_count_cache,
        cache_path=count_cache_path,
        validation_mode=COUNTING_MODE_RAW_ROWS,
    )

    total_raw_rows = sum(s.raw_rows_seen for s in stats)
    if total_raw_rows <= 0:
        raise BuildError("No input rows were found for allocation")

    allocation = allocate_quotas_from_raw_rows([s.raw_rows_seen for s in stats], config.target_size)
    for st, detail in zip(stats, allocation.details):
        st.usable_rows_for_quota = st.raw_rows_seen
        st.quota = detail.assigned_quota
        st.initial_quota = detail.assigned_quota
        st.effective_quota = detail.assigned_quota
        st.raw_quota = detail.raw_quota
        st.quota_floor = detail.floor_quota
        st.quota_fractional_remainder = detail.fractional_remainder
        st.quota_rounding_adjustment = detail.rounding_adjustment

    logger.info(
        (
            "stage=quota status=done allocation_basis=raw_rows total_raw_rows=%d "
            "rounding_method=%s assigned_total=%d"
        ),
        total_raw_rows,
        allocation.rounding_strategy,
        sum(st.initial_quota for st in stats),
    )
    logger.info(
        "stage=quota details=%s",
        ",".join(f"{st.path.name}:{st.initial_quota}" for st in stats),
    )

    output_handle, csv_writer = open_csv_writer(config.output_csv)
    total_written = 0
    duplicates_removed_final = 0
    final_seen: set[str] | None = set() if config.deduplicate_final else None

    redistributions_performed = 0
    total_redistributed_deficit = 0
    warnings: list[str] = []
    generation_started_at = time.perf_counter()

    states = [
        FileGenerationState(
            stats=st,
            iterator=iter_randomized_domain_candidates(
                st.path,
                config.parse,
                stable_file_seed(st.path, config.seed),
            ),
            remaining_quota=st.initial_quota,
            seen_per_file=(set() if config.deduplicate_per_file else None),
        )
        for st in stats
    ]

    try:
        while total_written < config.target_size and any(state.remaining_quota > 0 for state in states):
            progressed = False

            for state in states:
                if total_written >= config.target_size:
                    break
                if state.remaining_quota <= 0:
                    continue

                if not state.start_logged:
                    _log_generation_file_start(logger, state)
                    state.start_logged = True

                requested = min(state.remaining_quota, config.target_size - total_written)
                accepted, rejected, final_dedup_rejections = generate_from_file_until_quota(
                    state=state,
                    config=config,
                    csv_writer=csv_writer,
                    final_seen=final_seen,
                    quota_to_fill=requested,
                )
                duplicates_removed_final += final_dedup_rejections
                state.remaining_quota -= accepted
                total_written += accepted
                progressed = progressed or accepted > 0 or rejected > 0 or state.exhausted

                if state.remaining_quota > 0 and state.exhausted:
                    deficit = state.remaining_quota
                    state.remaining_quota = 0
                    state.stats.exhausted_before_quota = True
                    state.stats.redistributed_deficit_out += deficit

                    receivers = [candidate for candidate in states if candidate is not state and not candidate.exhausted]
                    redistribution = redistribute_remaining_deficit(deficit, [receiver.stats for receiver in receivers])

                    if redistribution:
                        redistribution_map = {path: share for path, share in redistribution}
                        redistribution_desc: list[str] = []
                        for receiver in receivers:
                            share = redistribution_map.get(receiver.stats.path, 0)
                            if share <= 0:
                                continue
                            receiver.remaining_quota += share
                            receiver.stats.redistributed_deficit_in += share
                            receiver.stats.effective_quota += share
                            redistribution_desc.append(f"{receiver.stats.path.name}:{share}")

                        redistributions_performed += 1
                        total_redistributed_deficit += deficit
                        logger.warning(
                            (
                                "stage=generate file=%s status=quota_unfilled initial_quota=%d "
                                "real_contribution=%d deficit=%d redistribution=%s"
                            ),
                            state.stats.path.name,
                            state.stats.initial_quota,
                            state.stats.accepted_domains,
                            deficit,
                            ",".join(redistribution_desc),
                        )
                    else:
                        warning = (
                            f"File {state.stats.path.name} exhausted before quota; "
                            f"deficit={deficit} could not be redistributed"
                        )
                        warnings.append(warning)
                        logger.warning(
                            (
                                "stage=generate file=%s status=quota_unfilled initial_quota=%d "
                                "real_contribution=%d deficit=%d redistribution=none"
                            ),
                            state.stats.path.name,
                            state.stats.initial_quota,
                            state.stats.accepted_domains,
                            deficit,
                        )

            if not progressed:
                break

        if total_written < config.target_size and not config.strict_no_refill and config.allow_replacement:
            missing = config.target_size - total_written
            logger.info("stage=refill status=start missing=%d", missing)

            pools = refill_candidate_pools(
                [s.path for s in stats],
                parse_config=config.parse,
                deduplicate_per_file=config.deduplicate_per_file,
            )
            weighted_paths = [s.path for s in stats if s.path in pools]
            weights = [max(1, s.raw_rows_seen) for s in stats if s.path in pools]
            if not weighted_paths:
                raise BuildError("Refill requested but no usable candidate pools were available")

            rng = random.Random(config.seed + 10_000_019)
            attempts = 0
            max_attempts = max(200_000, missing * 50)

            while total_written < config.target_size and attempts < max_attempts:
                attempts += 1
                file_path = rng.choices(weighted_paths, weights=weights, k=1)[0]
                pool = pools.get(file_path)
                if not pool:
                    continue
                candidate = rng.choice(pool)

                if final_seen is not None and candidate in final_seen:
                    duplicates_removed_final += 1
                    continue

                if final_seen is not None:
                    final_seen.add(candidate)
                csv_writer.writerow([candidate])
                total_written += 1

                for st in stats:
                    if st.path == file_path:
                        st.accepted_domains += 1
                        st.valid_rows = st.accepted_domains
                        st.emitted_rows = st.accepted_domains
                        st.effective_quota += 1
                        st.used_replacement = True
                        break

            logger.info(
                "stage=refill status=done generated=%d target=%d attempts=%d",
                total_written,
                config.target_size,
                attempts,
            )

            if total_written < config.target_size:
                raise BuildError(
                    "Could not refill to target size under current constraints. "
                    "Disable final deduplication, enable replacement, or run with --strict-no-refill to allow shortfall."
                )

        if total_written < config.target_size and config.strict_no_refill:
            shortfall = config.target_size - total_written
            warning = f"Output rows ({total_written}) are below target ({config.target_size}); shortfall={shortfall}"
            warnings.append(warning)
            logger.warning(
                "stage=generate status=shortfall_allowed generated=%d target=%d shortfall=%d",
                total_written,
                config.target_size,
                shortfall,
            )
        elif total_written < config.target_size and not config.allow_replacement:
            raise BuildError(
                "Output rows are below target and replacement is disabled. "
                "Either enable replacement (default) or run with --strict-no-refill to allow shortfall."
            )

    finally:
        for state in states:
            close = getattr(state.iterator, "close", None)
            if callable(close):
                close()
        output_handle.close()

    generation_elapsed = time.perf_counter() - generation_started_at
    _finalize_file_stats(stats)
    for stat in stats:
        _log_generation_file_done(logger, stat)

    logger.info(
        (
            "stage=summary status=done total_generated=%d target_size=%d total_rejected=%d "
            "total_redistributions=%d total_redistributed_deficit=%d"
        ),
        total_written,
        config.target_size,
        sum(s.rejected_domains for s in stats),
        redistributions_performed,
        total_redistributed_deficit,
    )
    logger.info(
        "stage=summary contributions=%s",
        ",".join(
            f"{st.path.name}:initial={st.initial_quota},effective={st.effective_quota},accepted={st.accepted_domains}"
            for st in stats
        ),
    )

    report = RunReport(
        config=config,
        input_files=files,
        files=stats,
        total_valid_rows=sum(s.accepted_domains for s in stats),
        total_usable_rows_for_quota=total_raw_rows,
        total_invalid_rows=sum(s.rejected_domains for s in stats),
        total_duplicates_removed_per_file=sum(s.duplicates_removed_per_file for s in stats),
        total_duplicates_removed_final=duplicates_removed_final,
        final_output_rows=total_written,
        status="success",
        allocation_basis="raw_rows",
        total_raw_rows=total_raw_rows,
        total_rejected_domains=sum(s.rejected_domains for s in stats),
        generation_phase_elapsed_seconds=generation_elapsed,
        total_elapsed_seconds=time.perf_counter() - build_started_at,
        redistributions_performed=redistributions_performed,
        total_redistributed_deficit=total_redistributed_deficit,
        count_phase_elapsed_seconds=count_summary.elapsed_seconds,
        count_phase_workers=count_summary.workers_used,
        count_phase_files_total=count_summary.files_total,
        count_phase_files_from_cache=count_summary.files_from_cache,
        count_phase_cache_enabled=count_summary.cache_enabled,
        count_phase_cache_path=count_summary.cache_path,
        count_phase_validation_mode=count_summary.counting_mode,
        count_phase_log_every=count_summary.count_log_every,
        rounding_strategy=allocation.rounding_strategy,
        quota_floor_total=allocation.floor_total,
        rounding_adjustment_total=allocation.rounding_adjustment_total,
        total_assigned_quota=sum(st.initial_quota for st in stats),
        warnings=warnings,
    )

    write_report(config.report_path, report)
    logger.info("Final benign domains stats written to %s", config.report_path)
    logger.info("Build complete. Final rows written: %d", total_written)
    return report
