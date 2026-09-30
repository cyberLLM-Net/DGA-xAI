from __future__ import annotations

from pathlib import Path

from domain_aggregator.models import ParseConfig
from domain_aggregator.sampling import iter_sampled_domains


def test_sampling_reproducible_without_replacement(tmp_path: Path) -> None:
    file_path = tmp_path / "domains.txt"
    file_path.write_text("a.com\nb.com\nc.com\nd.com\n", encoding="utf-8")

    iter1, rep1 = iter_sampled_domains(
        path=file_path,
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        quota=2,
        available_rows=4,
        allow_replacement=True,
        seed=123,
    )
    iter2, rep2 = iter_sampled_domains(
        path=file_path,
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        quota=2,
        available_rows=4,
        allow_replacement=True,
        seed=123,
    )

    assert rep1 is False and rep2 is False
    assert list(iter1) == list(iter2)


def test_sampling_with_replacement_when_quota_exceeds_available(tmp_path: Path) -> None:
    file_path = tmp_path / "domains.txt"
    file_path.write_text("a.com\nb.com\n", encoding="utf-8")

    sampled_iter, used_replacement = iter_sampled_domains(
        path=file_path,
        parse_config=ParseConfig(),
        deduplicate_per_file=False,
        quota=5,
        available_rows=2,
        allow_replacement=True,
        seed=7,
    )

    rows = list(sampled_iter)
    assert used_replacement is True
    assert len(rows) == 5
    assert set(rows).issubset({"a.com", "b.com"})
