from __future__ import annotations

import logging
from pathlib import Path

import pytest

from domain_aggregator.counting import count_all_files, count_file
from domain_aggregator.models import ParseConfig
from domain_aggregator.quota import allocate_quotas, allocate_quotas_from_raw_rows, allocate_quotas_with_details


def test_counting_uses_raw_rows_and_keeps_csv_header_support(tmp_path: Path) -> None:
    line_file = tmp_path / "domains.txt"
    csv_file = tmp_path / "domains.csv"
    line_file.write_text("example.com\nINVALID\n\nexample.org\n", encoding="utf-8")
    csv_file.write_text("domain,extra\na.com,1\nb.com,2\n", encoding="utf-8")

    line_stats = count_file(line_file, ParseConfig(), deduplicate_per_file=False)
    csv_stats = count_file(csv_file, ParseConfig(), deduplicate_per_file=False)

    assert line_stats.raw_rows_seen == 4
    assert line_stats.usable_rows_for_quota == 4
    assert csv_stats.raw_rows_seen == 2
    assert csv_stats.usable_rows_for_quota == 2


def test_allocate_quotas_from_raw_rows_is_exact_and_deterministic() -> None:
    raw_rows = [3, 2, 1]
    allocation = allocate_quotas_from_raw_rows(raw_rows, 10)
    assert sum(allocation.quotas) == 10
    assert allocation.quotas == [5, 3, 2]
    assert allocate_quotas(raw_rows, 10) == [5, 3, 2]


def test_allocate_quotas_with_details_reports_rounding_deterministically() -> None:
    allocation = allocate_quotas_with_details([1, 1, 1], 2)

    assert allocation.quotas == [1, 1, 0]
    assert allocation.rounding_strategy == "largest_remainder"
    assert allocation.floor_total == 0
    assert allocation.rounding_adjustment_total == 2
    assert [detail.rounding_adjustment for detail in allocation.details] == [1, 1, 0]


def test_count_all_files_parallel_logs_per_file_completion_without_progress_noise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("a.com\nb.com\nINVALID\n", encoding="utf-8")
    file_b.write_text("x.org\ny.org\nz.org\n", encoding="utf-8")

    logger = logging.getLogger("domain_aggregator")
    caplog.set_level(logging.INFO, logger="domain_aggregator")

    stats, summary = count_all_files(
        files=[file_a, file_b],
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        count_workers=8,
        count_log_every=1,
        logger=logger,
        use_count_cache=False,
    )

    assert summary.files_total == 2
    assert summary.workers_used == 2
    assert summary.total_raw_rows == 6
    assert sum(s.raw_rows_seen for s in stats) == 6
    assert not any("stage=count_raw_rows" in record.message and "status=progress" in record.message for record in caplog.records)
    per_file_done_logs = [
        record.message
        for record in caplog.records
        if "stage=count_raw_rows" in record.message and "file=" in record.message and "status=done" in record.message
    ]
    assert len(per_file_done_logs) == 2


def test_count_cache_reused_when_file_metadata_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "domains.txt"
    file_path.write_text("a.com\nb.com\nINVALID\n", encoding="utf-8")
    cache_path = tmp_path / "metadata" / "count_cache.json"
    logger = logging.getLogger("domain_aggregator")

    first_stats, first_summary = count_all_files(
        files=[file_path],
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        count_workers=4,
        count_log_every=1000,
        logger=logger,
        use_count_cache=True,
        cache_path=cache_path,
    )
    second_stats, second_summary = count_all_files(
        files=[file_path],
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        count_workers=4,
        count_log_every=1000,
        logger=logger,
        use_count_cache=True,
        cache_path=cache_path,
    )

    assert first_summary.files_from_cache == 0
    assert second_summary.files_from_cache == 1
    assert first_stats[0].raw_rows_seen == second_stats[0].raw_rows_seen
    assert second_stats[0].reused_from_cache is True


def test_count_cache_invalidates_when_file_changes(tmp_path: Path) -> None:
    file_path = tmp_path / "domains.txt"
    file_path.write_text("a.com\nb.com\n", encoding="utf-8")
    cache_path = tmp_path / "metadata" / "count_cache.json"
    logger = logging.getLogger("domain_aggregator")

    _, _ = count_all_files(
        files=[file_path],
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        count_workers=2,
        count_log_every=1000,
        logger=logger,
        use_count_cache=True,
        cache_path=cache_path,
    )

    file_path.write_text("a.com\nb.com\nc.com\n", encoding="utf-8")

    stats, summary = count_all_files(
        files=[file_path],
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        count_workers=2,
        count_log_every=1000,
        logger=logger,
        use_count_cache=True,
        cache_path=cache_path,
    )

    assert summary.files_from_cache == 0
    assert stats[0].reused_from_cache is False
    assert stats[0].raw_rows_seen == 3
