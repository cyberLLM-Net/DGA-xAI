from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from .models import FileStats, ParseConfig

COUNTING_MODE_RAW_ROWS = "raw_rows_only"
_CACHE_VERSION = 2
_DELIMITER_CANDIDATES = [",", "\t", ";", "|"]


@dataclass(frozen=True)
class CountPhaseSummary:
    files_total: int
    files_counted: int
    files_from_cache: int
    total_raw_rows: int
    elapsed_seconds: float
    workers_used: int
    count_log_every: int
    counting_mode: str
    cache_enabled: bool
    cache_path: str | None


def _detect_delimiter_in_sample(lines: list[str]) -> str | None:
    best_delim: str | None = None
    best_score = 0
    for delimiter in _DELIMITER_CANDIDATES:
        score = sum(line.count(delimiter) for line in lines if line.strip())
        if score > best_score:
            best_score = score
            best_delim = delimiter
    return best_delim if best_score > 0 else None


def _looks_like_header(row: list[str]) -> bool:
    lowered = [cell.strip().lower() for cell in row]
    header_tokens = {"domain", "hostname", "fqdn", "host"}
    return any(token in header_tokens for token in lowered)


def _estimate_header_rows(path: Path, parse_config: ParseConfig) -> int:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample_lines: list[str] = []
        for _ in range(30):
            line = handle.readline()
            if not line:
                break
            sample_lines.append(line)

        delimiter = parse_config.delimiter if parse_config.delimiter is not None else _detect_delimiter_in_sample(sample_lines)
        if delimiter is None:
            return 0

        reader = csv.reader(chain(sample_lines, handle), delimiter=delimiter)
        first_row: list[str] | None = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                first_row = row
                break

    if first_row is None:
        return 0
    if parse_config.domain_column is not None:
        return 1
    if parse_config.header_auto_detect and _looks_like_header(first_row):
        return 1
    return 0


def _count_physical_rows_fast(path: Path, stats: FileStats) -> None:
    with path.open("rb", buffering=1024 * 1024) as handle:
        for _ in handle:
            stats.raw_rows_seen += 1


def count_raw_rows_in_file(
    path: Path,
    parse_config: ParseConfig,
    logger: logging.Logger | None = None,
    *,
    count_log_every: int = 500_000,
    file_index: int | None = None,
    total_files: int | None = None,
) -> FileStats:
    stats = FileStats(path=path)
    stats.size_bytes = path.stat().st_size
    started_at = time.perf_counter()

    if logger is not None:
        ordinal = "?"
        if file_index is not None and total_files is not None:
            ordinal = f"{file_index}/{total_files}"
        logger.info(
            "stage=count_raw_rows file=%s status=start index=%s size_bytes=%d",
            path.name,
            ordinal,
            stats.size_bytes,
        )

    _count_physical_rows_fast(
        path=path,
        stats=stats,
    )

    header_rows = 0
    try:
        header_rows = _estimate_header_rows(path, parse_config)
    except Exception as exc:  # pragma: no cover - defensive branch for malformed csv edge cases.
        if logger is not None:
            logger.warning(
                "stage=count_raw_rows file=%s status=header_estimate_failed error=%s",
                path.name,
                exc,
            )

    if header_rows > 0 and stats.raw_rows_seen > 0:
        stats.raw_rows_seen = max(0, stats.raw_rows_seen - header_rows)

    elapsed = time.perf_counter() - started_at
    stats.count_elapsed_seconds = elapsed
    stats.count_avg_rate_lps = (stats.raw_rows_seen / elapsed) if elapsed > 0 else 0.0
    stats.usable_rows_for_quota = stats.raw_rows_seen

    if logger is not None:
        logger.info(
            (
                "stage=count_raw_rows file=%s status=done raw_rows=%d "
                "header_rows_removed=%d elapsed_sec=%.2f avg_rate_lps=%.2f reused_from_cache=%s"
            ),
            path.name,
            stats.raw_rows_seen,
            header_rows,
            stats.count_elapsed_seconds,
            stats.count_avg_rate_lps,
            stats.reused_from_cache,
        )

    return stats


def _cache_key(path: Path) -> str:
    return str(path.resolve())


def _cache_signature(parse_config: ParseConfig, counting_mode: str) -> dict[str, Any]:
    return {
        "delimiter": parse_config.delimiter,
        "domain_column": parse_config.domain_column,
        "header_auto_detect": parse_config.header_auto_detect,
        "counting_mode": counting_mode,
    }


def _load_cache_entries(path: Path, logger: logging.Logger) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("stage=count_cache status=load_failed path=%s error=%s", path, exc)
        return {}

    if not isinstance(payload, dict):
        return {}
    if payload.get("version") != _CACHE_VERSION:
        return {}

    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def _write_cache_entries(path: Path, entries: dict[str, dict[str, Any]], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _CACHE_VERSION, "entries": entries}
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        logger.warning("stage=count_cache status=write_failed path=%s error=%s", path, exc)
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def _cache_hit(
    entry: dict[str, Any] | None,
    path: Path,
    signature: dict[str, Any],
) -> bool:
    if entry is None:
        return False
    try:
        stat = path.stat()
    except OSError:
        return False

    if entry.get("size_bytes") != stat.st_size:
        return False
    if entry.get("mtime_ns") != stat.st_mtime_ns:
        return False
    return entry.get("signature") == signature


def _stats_from_cache(path: Path, entry: dict[str, Any]) -> FileStats:
    raw_rows = int(entry.get("raw_rows_seen", 0))
    return FileStats(
        path=path,
        size_bytes=int(entry.get("size_bytes", 0)),
        raw_rows_seen=raw_rows,
        usable_rows_for_quota=raw_rows,
        reused_from_cache=True,
        count_elapsed_seconds=0.0,
        count_avg_rate_lps=0.0,
    )


def _stats_to_cache(path: Path, stats: FileStats, signature: dict[str, Any]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "signature": signature,
        "raw_rows_seen": stats.raw_rows_seen,
    }


def count_raw_rows_all_files(
    files: list[Path],
    parse_config: ParseConfig,
    count_workers: int,
    count_log_every: int,
    logger: logging.Logger,
    *,
    use_count_cache: bool = False,
    cache_path: Path | None = None,
    counting_mode: str = COUNTING_MODE_RAW_ROWS,
) -> tuple[list[FileStats], CountPhaseSummary]:
    if count_workers <= 0:
        raise ValueError("count_workers must be greater than zero")
    if count_log_every <= 0:
        raise ValueError("count_log_every must be greater than zero")

    phase_started_at = time.perf_counter()
    files_total = len(files)
    max_workers = min(max(1, count_workers), max(1, files_total))

    logger.info(
        (
            "stage=count_raw_rows status=start files=%d max_workers=%d "
            "counting_mode=%s count_log_every=%d count_log_every_effective=false cache_enabled=%s"
        ),
        files_total,
        max_workers,
        counting_mode,
        count_log_every,
        use_count_cache,
    )

    signature = _cache_signature(parse_config, counting_mode)
    cache_entries: dict[str, dict[str, Any]] = {}
    if use_count_cache and cache_path is not None:
        cache_entries = _load_cache_entries(cache_path, logger)

    cache_hits = 0
    collected_stats: list[FileStats] = []
    pending: list[tuple[int, Path]] = []

    for index, file_path in enumerate(files, start=1):
        cache_entry = cache_entries.get(_cache_key(file_path))
        if use_count_cache and cache_path is not None and _cache_hit(cache_entry, file_path, signature):
            cached_stats = _stats_from_cache(file_path, cache_entry)
            collected_stats.append(cached_stats)
            cache_hits += 1
            logger.info(
                (
                    "stage=count_raw_rows file=%s status=cache_hit index=%d/%d "
                    "size_bytes=%d raw_rows=%d"
                ),
                file_path.name,
                index,
                files_total,
                cached_stats.size_bytes,
                cached_stats.raw_rows_seen,
            )
            continue
        pending.append((index, file_path))

    workers_for_pending = min(max_workers, max(1, len(pending)))
    if pending:
        with ThreadPoolExecutor(max_workers=workers_for_pending) as executor:
            futures = {
                executor.submit(
                    count_raw_rows_in_file,
                    file_path,
                    parse_config,
                    logger,
                    count_log_every=count_log_every,
                    file_index=index,
                    total_files=files_total,
                ): (index, file_path)
                for index, file_path in pending
            }

            for future in as_completed(futures):
                _, file_path = futures[future]
                try:
                    file_stats = future.result()
                except Exception as exc:
                    logger.error("stage=count_raw_rows file=%s status=failed error=%s", file_path.name, exc)
                    raise

                collected_stats.append(file_stats)
                if use_count_cache and cache_path is not None:
                    cache_entries[_cache_key(file_path)] = _stats_to_cache(file_path, file_stats, signature)

    if use_count_cache and cache_path is not None:
        _write_cache_entries(cache_path, cache_entries, logger)

    collected_stats.sort(key=lambda stat: str(stat.path))

    total_raw_rows = sum(stat.raw_rows_seen for stat in collected_stats)
    elapsed = time.perf_counter() - phase_started_at

    logger.info(
        (
            "stage=count_raw_rows status=done files=%d counted=%d cache_hits=%d "
            "total_raw_rows=%d elapsed_sec=%.2f"
        ),
        files_total,
        len(pending),
        cache_hits,
        total_raw_rows,
        elapsed,
    )

    summary = CountPhaseSummary(
        files_total=files_total,
        files_counted=len(pending),
        files_from_cache=cache_hits,
        total_raw_rows=total_raw_rows,
        elapsed_seconds=elapsed,
        workers_used=workers_for_pending if pending else max_workers,
        count_log_every=count_log_every,
        counting_mode=counting_mode,
        cache_enabled=use_count_cache,
        cache_path=str(cache_path) if cache_path is not None else None,
    )
    return collected_stats, summary


def count_all_files(
    files: list[Path],
    parse_config: ParseConfig,
    deduplicate_per_file: bool,
    count_workers: int,
    count_log_every: int,
    logger: logging.Logger,
    *,
    use_count_cache: bool = False,
    cache_path: Path | None = None,
    validation_mode: str = COUNTING_MODE_RAW_ROWS,
) -> tuple[list[FileStats], CountPhaseSummary]:
    del deduplicate_per_file
    return count_raw_rows_all_files(
        files=files,
        parse_config=parse_config,
        count_workers=count_workers,
        count_log_every=count_log_every,
        logger=logger,
        use_count_cache=use_count_cache,
        cache_path=cache_path,
        counting_mode=validation_mode,
    )


def count_file(path: Path, parse_config: ParseConfig, deduplicate_per_file: bool) -> FileStats:
    del deduplicate_per_file
    return count_raw_rows_in_file(path, parse_config, logger=None)


def count_files(
    files: list[Path],
    parse_config: ParseConfig,
    deduplicate_per_file: bool,
    threads: int,
    logger: logging.Logger,
    *,
    count_log_every: int = 500_000,
    use_count_cache: bool = False,
    cache_path: Path | None = None,
    validation_mode: str = COUNTING_MODE_RAW_ROWS,
) -> list[FileStats]:
    stats, _ = count_all_files(
        files=files,
        parse_config=parse_config,
        deduplicate_per_file=deduplicate_per_file,
        count_workers=threads,
        count_log_every=count_log_every,
        logger=logger,
        use_count_cache=use_count_cache,
        cache_path=cache_path,
        validation_mode=validation_mode,
    )
    return stats
