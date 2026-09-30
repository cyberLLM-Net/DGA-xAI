from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import domain_aggregator.builder as builder_module
import domain_aggregator.cli as cli_module
from domain_aggregator.builder import build_dataset
from domain_aggregator.cli import run
from domain_aggregator.logging_utils import setup_logging
from domain_aggregator.models import BuildConfig, ParseConfig


def _make_input(tmp_path: Path) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "f1.txt").write_text("a.com\nb.com\nc.com\n", encoding="utf-8")
    (input_dir / "f2.txt").write_text("x.org\ny.org\n", encoding="utf-8")
    return input_dir


def test_build_stream_writes_target_and_report(tmp_path: Path) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "results" / "final.csv"
    report = tmp_path / "results" / "report.json"
    log_path = tmp_path / "logs" / "run.log"

    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=10,
        seed=42,
        threads=2,
        parse=ParseConfig(),
        log_path=log_path,
    )

    logger = setup_logging(log_path)
    run_report = build_dataset(cfg, logger)

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "domain"
    assert len(lines) == 11
    assert run_report.final_output_rows == 10

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["target_size"] == 10
    assert payload["totals"]["final_output_rows"] == 10
    assert payload["allocation_basis"] == "raw_rows"
    assert payload["rounding_method"] == "largest_remainder"
    assert payload["selection_strategy"] == "random_per_file"
    assert len(payload["files"]) == 2
    assert payload["counting"]["count_workers"] == 2
    assert payload["counting"]["count_log_every"] == 500000
    assert payload["counting"]["count_validation_mode"] == "raw_rows_only"
    assert "elapsed_seconds" in payload["counting"]
    assert "generation_seconds" in payload["timings"]
    assert "raw_rows" in payload["quota_allocation"]["per_file"][0]


def test_generation_validates_on_demand_without_full_prevalidation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    domains_a = "\n".join(f"a{i}.com" for i in range(200))
    domains_b = "\n".join(f"b{i}.org" for i in range(200))
    (input_dir / "a.txt").write_text(domains_a + "\n", encoding="utf-8")
    (input_dir / "b.txt").write_text(domains_b + "\n", encoding="utf-8")

    output = tmp_path / "result.csv"
    report = tmp_path / "stats.json"
    log_path = tmp_path / "logs" / "run.log"

    validation_calls = {"count": 0}
    original_validator = builder_module.is_valid_domain

    def _tracked_validator(domain: str, strictness: str = "balanced") -> bool:
        validation_calls["count"] += 1
        return original_validator(domain, strictness=strictness)

    monkeypatch.setattr(builder_module, "is_valid_domain", _tracked_validator)

    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=1,
        seed=13,
        threads=2,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=True,
        log_path=log_path,
    )

    logger = setup_logging(log_path)
    build_dataset(cfg, logger)

    assert validation_calls["count"] < 20


def test_generation_keeps_reading_same_file_after_discards_until_quota(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "single.txt").write_text(
        "INVALID\nalso_bad\nok.com\nbad domain\nyes.org\n",
        encoding="utf-8",
    )

    output = tmp_path / "result.csv"
    report = tmp_path / "stats.json"
    log_path = tmp_path / "logs" / "run.log"
    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=2,
        seed=99,
        threads=1,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=False,
        log_path=log_path,
    )

    logger = setup_logging(log_path)
    run_report = build_dataset(cfg, logger)

    assert run_report.final_output_rows == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    file_stats = payload["files"][0]
    assert file_stats["accepted_domains"] == 2
    assert file_stats["generation_attempted_rows"] >= 2
    assert file_stats["exhausted_before_quota"] is False


def test_exhausted_file_redistributes_deficit_deterministically(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a_invalid.txt").write_text("INVALID\nBAD\nWRONG\nNOPE\n", encoding="utf-8")
    (input_dir / "b_valid.txt").write_text("a.com\nb.com\nc.com\nd.com\ne.com\nf.com\n", encoding="utf-8")

    output = tmp_path / "result.csv"
    report = tmp_path / "stats.json"
    log_path = tmp_path / "logs" / "run.log"
    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=6,
        seed=5,
        threads=2,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=False,
        log_path=log_path,
    )

    logger = setup_logging(log_path)
    run_report = build_dataset(cfg, logger)
    assert run_report.final_output_rows == 6

    payload = json.loads(report.read_text(encoding="utf-8"))
    by_name = {Path(item["path"]).name: item for item in payload["files"]}
    invalid_stats = by_name["a_invalid.txt"]
    valid_stats = by_name["b_valid.txt"]

    assert invalid_stats["initial_quota"] == 2
    assert invalid_stats["accepted_domains"] == 0
    assert invalid_stats["exhausted_before_quota"] is True
    assert invalid_stats["redistributed_deficit_out"] == 2
    assert valid_stats["redistributed_deficit_in"] == 2
    assert valid_stats["accepted_domains"] == 6
    assert payload["totals"]["redistributions_performed"] >= 1


def test_total_generated_matches_target_when_capacity_is_sufficient(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "f1.txt").write_text("a.com\nb.com\nc.com\nd.com\ne.com\n", encoding="utf-8")
    (input_dir / "f2.txt").write_text("x.org\ny.org\nz.org\nw.org\n", encoding="utf-8")

    output = tmp_path / "result.csv"
    report = tmp_path / "stats.json"
    log_path = tmp_path / "logs" / "run.log"
    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=8,
        seed=11,
        threads=2,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=False,
        log_path=log_path,
    )

    logger = setup_logging(log_path)
    run_report = build_dataset(cfg, logger)

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 9
    assert run_report.final_output_rows == 8


def test_quota_allocation_is_deterministic_across_runs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a.com\nb.com\nc.com\n", encoding="utf-8")
    (input_dir / "b.txt").write_text("x.org\ny.org\n", encoding="utf-8")
    (input_dir / "c.txt").write_text("m.net\nn.net\n", encoding="utf-8")

    logger = setup_logging(tmp_path / "logs" / "run.log")

    output_1 = tmp_path / "out1.csv"
    report_1 = tmp_path / "report1.json"
    cfg_1 = BuildConfig(
        input_dir=input_dir,
        output_csv=output_1,
        report_path=report_1,
        target_size=13,
        parse=ParseConfig(),
        allow_replacement=True,
        strict_no_refill=False,
        log_path=tmp_path / "logs" / "run.log",
    )
    build_dataset(cfg_1, logger)

    output_2 = tmp_path / "out2.csv"
    report_2 = tmp_path / "report2.json"
    cfg_2 = BuildConfig(
        input_dir=input_dir,
        output_csv=output_2,
        report_path=report_2,
        target_size=13,
        parse=ParseConfig(),
        allow_replacement=True,
        strict_no_refill=False,
        log_path=tmp_path / "logs" / "run.log",
    )
    build_dataset(cfg_2, logger)

    payload_1 = json.loads(report_1.read_text(encoding="utf-8"))
    payload_2 = json.loads(report_2.read_text(encoding="utf-8"))
    q1 = {Path(item["path"]).name: item["initial_quota"] for item in payload_1["files"]}
    q2 = {Path(item["path"]).name: item["initial_quota"] for item in payload_2["files"]}
    assert q1 == q2
    assert sum(q1.values()) == 13


def test_cli_smoke(tmp_path: Path) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "out.csv"
    report = tmp_path / "report.json"

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--target-size",
            "8",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    assert output.exists()
    assert report.exists()


def test_cli_default_outputs_are_written_inside_results_and_results_is_auto_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = _make_input(tmp_path)

    results_dir = tmp_path / "results"
    assert not results_dir.exists()

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--target-size",
            "8",
            "--seed",
            "1",
        ]
    )

    assert rc == 0
    assert (results_dir / "udcdga_benigns_domains.csv").exists()
    assert (results_dir / "udcdga_benigns_domains_stats.json").exists()


def test_cli_defaults_write_udcdga_outputs(tmp_path: Path) -> None:
    input_dir = _make_input(tmp_path)
    logs_dir = tmp_path / "logs"

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--results-dir",
            str(tmp_path),
            "--logs-dir",
            str(logs_dir),
            "--target-size",
            "8",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    assert (tmp_path / "udcdga_benigns_domains.csv").exists()
    assert (tmp_path / "udcdga_benigns_domains_stats.json").exists()


def test_generation_randomizes_within_single_file_instead_of_always_using_prefix(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    domains = "\n".join(f"d{i}.com" for i in range(100))
    (input_dir / "single.txt").write_text(domains + "\n", encoding="utf-8")

    output = tmp_path / "results" / "out.csv"
    report = tmp_path / "results" / "report.json"
    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=10,
        seed=123,
        threads=1,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=True,
        log_path=tmp_path / "logs" / "run.log",
    )

    logger = setup_logging(cfg.log_path)
    build_dataset(cfg, logger)

    sampled = output.read_text(encoding="utf-8").strip().splitlines()[1:]
    prefix = [f"d{i}.com" for i in range(10)]
    assert sampled != prefix
    sampled_indices = [int(value.split(".")[0][1:]) for value in sampled]
    assert max(sampled_indices) >= 20


def test_generation_randomization_is_reproducible_with_fixed_seed(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    domains = "\n".join(f"d{i}.com" for i in range(120))
    (input_dir / "single.txt").write_text(domains + "\n", encoding="utf-8")

    logger = setup_logging(tmp_path / "logs" / "run.log")

    cfg_1 = BuildConfig(
        input_dir=input_dir,
        output_csv=tmp_path / "results" / "out1.csv",
        report_path=tmp_path / "results" / "report1.json",
        target_size=30,
        seed=2026,
        threads=1,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=True,
        log_path=tmp_path / "logs" / "run.log",
    )
    cfg_2 = BuildConfig(
        input_dir=input_dir,
        output_csv=tmp_path / "results" / "out2.csv",
        report_path=tmp_path / "results" / "report2.json",
        target_size=30,
        seed=2026,
        threads=1,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=True,
        log_path=tmp_path / "logs" / "run.log",
    )

    build_dataset(cfg_1, logger)
    build_dataset(cfg_2, logger)

    rows_1 = (tmp_path / "results" / "out1.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
    rows_2 = (tmp_path / "results" / "out2.csv").read_text(encoding="utf-8").strip().splitlines()[1:]
    assert rows_1 == rows_2


def test_generation_logs_only_file_completion_events(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "results" / "out.csv"
    report = tmp_path / "results" / "report.json"

    logger = logging.getLogger("domain_aggregator")
    caplog.set_level(logging.INFO, logger="domain_aggregator")

    cfg = BuildConfig(
        input_dir=input_dir,
        output_csv=output,
        report_path=report,
        target_size=5,
        seed=17,
        threads=2,
        parse=ParseConfig(),
        allow_replacement=False,
        strict_no_refill=False,
        log_path=tmp_path / "logs" / "run.log",
    )

    build_dataset(cfg, logger)

    assert not any("progress_pct=" in record.message for record in caplog.records)
    done_logs = [
        record.message
        for record in caplog.records
        if "stage=generate" in record.message and "status=done" in record.message and "file=" in record.message
    ]
    assert len(done_logs) >= 2
    assert all("assigned_quota=" in message and "accepted=" in message and "elapsed_sec=" in message for message in done_logs)


def test_cli_default_target_size_remains_backward_compatible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "default.csv"
    report = tmp_path / "default.json"

    monkeypatch.setattr(cli_module, "DEFAULT_TARGET_SIZE", 10)

    rc = cli_module.run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--seed",
            "7",
        ]
    )
    assert rc == 0

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 11

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["target_size"] == 10
    assert payload["generated_total"] == 10
    assert payload["default_target_size_used"] is True


def test_cli_target_size_1000_is_exact_and_reported(tmp_path: Path) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "target1000.csv"
    report = tmp_path / "target1000.json"

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--target-size",
            "1000",
            "--seed",
            "3",
        ]
    )
    assert rc == 0

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "domain"
    assert len(rows) == 1001

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["target_size"] == 1000
    assert payload["generated_total"] == 1000
    assert payload["default_target_size_used"] is False
    assert payload["totals"]["final_output_rows"] == 1000
    assert sum(item["initial_quota"] for item in payload["files"]) == 1000
    assert payload["quota_allocation"]["rounding_strategy"] == "largest_remainder"
    assert sum(item["assigned_quota"] for item in payload["quota_allocation"]["per_file"]) == 1000
    assert sum(item["emitted_rows"] for item in payload["quota_allocation"]["per_file"]) == 1000
    assert payload["counting"]["count_workers"] == 2
    assert payload["counting"]["count_validation_mode"] == "raw_rows_only"
    assert payload["counting"]["cache_enabled"] is False


@pytest.mark.parametrize("invalid_value", ["0", "-3"])
def test_cli_invalid_target_size_returns_parse_error(tmp_path: Path, capsys: pytest.CaptureFixture[str], invalid_value: str) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "bad.csv"
    report = tmp_path / "bad.json"

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--target-size",
            invalid_value,
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "must be a positive integer greater than 0" in captured.err


@pytest.mark.parametrize(
    ("arg_name", "arg_value"),
    [
        ("--count-workers", "0"),
        ("--count-workers", "-2"),
        ("--count-log-every", "0"),
        ("--count-log-every", "-10"),
    ],
)
def test_cli_invalid_counting_parameters_return_parse_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    arg_name: str,
    arg_value: str,
) -> None:
    input_dir = _make_input(tmp_path)

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            arg_name,
            arg_value,
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "must be a positive integer greater than 0" in captured.err


def test_cli_parses_new_counting_flags(tmp_path: Path) -> None:
    input_dir = _make_input(tmp_path)
    output = tmp_path / "counting-flags.csv"
    report = tmp_path / "counting-flags.json"

    rc = run(
        [
            "build",
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--target-size",
            "10",
            "--count-workers",
            "2",
            "--count-log-every",
            "1",
            "--use-count-cache",
        ]
    )
    assert rc == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["counting"]["count_workers"] == 2
    assert payload["counting"]["count_log_every"] == 1
    assert payload["counting"]["cache_enabled"] is True
