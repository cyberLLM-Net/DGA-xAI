from __future__ import annotations

from pathlib import Path

from domain_aggregator.discovery import discover_input_files
from domain_aggregator.domain import is_valid_domain, normalize_domain
from domain_aggregator.models import ParseConfig
from domain_aggregator.parser import iter_domain_candidates


def test_discover_input_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("example.com\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("domain\nexample.org\n", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("x", encoding="utf-8")

    files = discover_input_files(tmp_path)
    assert [p.name for p in files] == ["a.txt", "b.csv"]


def test_parser_line_and_csv_and_header_detection(tmp_path: Path) -> None:
    line_file = tmp_path / "line.txt"
    csv_file = tmp_path / "table.txt"
    line_file.write_text("\ufeffExample.COM\n\nsub.example.org\n", encoding="utf-8")
    csv_file.write_text("domain,extra\nExample.ORG,1\nExample.NET,2\n", encoding="utf-8")

    line_values = list(iter_domain_candidates(line_file, ParseConfig()))
    csv_values = list(iter_domain_candidates(csv_file, ParseConfig()))

    assert line_values == ["example.com", "sub.example.org"]
    assert csv_values == ["example.org", "example.net"]


def test_domain_normalization_does_not_apply_idna_conversion() -> None:
    assert normalize_domain("\ufeff B\u00dcCHER.Example. \n") == "b\u00fccher.example"


def test_validation_modes_apply_documented_tld_rules() -> None:
    assert is_valid_domain("example.a1", "lenient")
    assert not is_valid_domain("example.a1", "balanced")
    assert is_valid_domain(f"example.{'a' * 63}", "balanced")
    assert not is_valid_domain(f"example.{'a' * 25}", "strict")
    assert is_valid_domain(f"example.{'a' * 24}", "strict")
