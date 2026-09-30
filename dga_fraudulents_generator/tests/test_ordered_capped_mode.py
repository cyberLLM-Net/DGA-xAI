from __future__ import annotations

import csv
from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import run_pipeline
from dga_fraudulents_dataset.utils import read_json


UNIQUE_TEMPLATE = """
def dga(seed, nr):
    return f"{prefix}-{{nr}}.example.com"
"""


def _write_unique_algo(root: Path, code: str, prefix: str) -> None:
    p = root / code
    p.mkdir(parents=True, exist_ok=True)
    (p / "dga.py").write_text(UNIQUE_TEMPLATE.format(prefix=prefix), encoding="utf-8")


def _csv_rows(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            out.append((row[0], row[1]))
    return out


def test_ordered_capped_enforces_strict_order(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "first", "f")
    _write_unique_algo(root, "second", "s")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["second", "first"],
            per_algorithm_cap=3,
            batch_size=5,
            checkpoint_every=5,
        )
    )
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    rows = _csv_rows(out / "udcdga_dga_domains.csv")

    assert payload["generation_mode"] == "ordered_capped"
    assert stats["ordered_algorithms"] == ["second", "first"]
    assert stats["distribution_by_algorithm"]["second"]["order_index"] == 0
    assert stats["distribution_by_algorithm"]["first"]["order_index"] == 1
    assert [algo for _, algo in rows[:3]] == ["second", "second", "second"]


def test_ordered_capped_hard_cap_and_no_redistribution(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "a1", "a1")
    _write_unique_algo(root, "a2", "a2")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["a1", "a2"],
            per_algorithm_cap=7,
            batch_size=9,
            checkpoint_every=7,
        )
    )
    stats = read_json(out / "udcdga_dga_domains_stats.json")
    a1 = stats["distribution_by_algorithm"]["a1"]
    a2 = stats["distribution_by_algorithm"]["a2"]

    assert a1["delivered_unique"] == 7
    assert a2["delivered_unique"] == 7
    assert a1["delivered_unique"] <= 7
    assert a2["delivered_unique"] <= 7
    assert a1["stopped_reason"] == "cap_reached"
    assert a2["stopped_reason"] == "cap_reached"
    assert a1["redistributed_quota"] == 0
    assert a2["redistributed_quota"] == 0
    assert payload["redistribution_enabled"] is False


def test_ordered_capped_final_size_equals_sum_of_contributions(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "x1", "x1")
    _write_unique_algo(root, "x2", "x2")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["x1", "x2"],
            per_algorithm_cap=5,
            batch_size=5,
            checkpoint_every=5,
        )
    )
    by_algo = payload["distribution_by_algorithm"]
    delivered_sum = sum(v["delivered_unique"] for v in by_algo.values())
    assert payload["final_unique_domains"] == delivered_sum
    assert payload["effective_target_count"] == payload["final_unique_domains"]


def test_ordered_capped_missing_algorithms_continue(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "present", "p")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["present", "missing_algo"],
            per_algorithm_cap=4,
            batch_size=4,
            checkpoint_every=4,
        )
    )
    missing = payload["distribution_by_algorithm"]["missing_algo"]
    assert missing["status"] == "missing"
    assert missing["stopped_reason"] == "missing_algorithm"
    assert missing["delivered_unique"] == 0


def test_ordered_capped_resume_preserves_order_and_cap(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "r1", "r1")
    _write_unique_algo(root, "r2", "r2")

    run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["r1", "r2"],
            per_algorithm_cap=10,
            batch_size=10,
            checkpoint_every=5,
            dry_run=True,
            target_count=5,
        )
    )

    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["r1", "r2"],
            per_algorithm_cap=10,
            batch_size=10,
            checkpoint_every=5,
            resume=True,
        )
    )
    state = read_json(out / "udcdga_generation_state.json")
    assert payload["ordered_algorithms"] == ["r1", "r2"]
    assert payload["distribution_by_algorithm"]["r1"]["delivered_unique"] == 10
    assert payload["distribution_by_algorithm"]["r2"]["delivered_unique"] == 10
    assert state["ordered_algorithms"] == ["r1", "r2"]
    assert int(state["per_algorithm_cap"]) == 10
    assert int(state["current_algorithm_index"]) == 2


def test_ordered_capped_no_15m_requirement(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "tiny", "tiny")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["tiny"],
            per_algorithm_cap=3,
            batch_size=3,
            checkpoint_every=3,
        )
    )
    assert payload["final_unique_domains"] == 3
    assert payload["global_target_count"] is None


def test_ordered_capped_csv_and_json_never_exceed_cap(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "out"
    _write_unique_algo(root, "c1", "c1")
    _write_unique_algo(root, "c2", "c2")
    payload = run_pipeline(
        AppConfig(
            algorithms_root=root,
            output_dir=out,
            generation_mode="ordered_capped",
            ordered_algorithms=["c1", "c2"],
            per_algorithm_cap=4,
            batch_size=4,
            checkpoint_every=4,
        )
    )
    rows = _csv_rows(out / "udcdga_dga_domains.csv")
    counts: dict[str, int] = {}
    for _, code in rows:
        counts[code] = counts.get(code, 0) + 1
    for code, rec in payload["distribution_by_algorithm"].items():
        assert rec["delivered_unique"] <= 4
        assert counts.get(code, 0) <= 4
