from __future__ import annotations

import json
from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import GenerationError, run_pipeline
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.orchard_adapter import OrchardAdapter
from dga_fraudulents_dataset.utils import read_json

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _write_orchard_algo(root: Path, balances: list[int]) -> Path:
    orchard = root / "orchard"
    orchard.mkdir(parents=True, exist_ok=True)
    (orchard / "dga.py").write_text(
        "def dga(when, blockchain=False):\n    return []\n",
        encoding="utf-8",
    )
    db = {}
    for i, bal in enumerate(balances):
        db[f"tx_{i:03d}"] = {"hash": f"tx_{i:03d}", "balance": bal, "time": 1598166461 + i}
    (orchard / "db.json").write_text(json.dumps(db), encoding="utf-8")
    return orchard


def _orchard_inspection(orchard_path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="orchard",
        path=str(orchard_path),
        strategy="python_function",
        entrypoint=str(orchard_path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_orchard_adapter_loads_db_from_algorithm_relative_path(tmp_path: Path):
    orchard = _write_orchard_algo(tmp_path / "algos", [1000, 900, 800, 700])
    adapter = OrchardAdapter(_orchard_inspection(orchard), seed_strategy="sequential")

    profile = adapter.profile()
    assert adapter.db_path == orchard / "db.json"
    assert profile["source_records"] == 4
    assert profile["estimated_max_unique_capacity"] > 0
    assert profile["generation_mode"] == "dataset_backed"


def test_orchard_batch_generation_avoids_replay(tmp_path: Path):
    orchard = _write_orchard_algo(tmp_path / "algos", [1000, 900, 800, 700, 600])
    adapter = OrchardAdapter(_orchard_inspection(orchard), seed_strategy="sequential")

    b1 = adapter.generate(12)
    b2 = adapter.generate(12)

    assert len(b1.domains) == 12
    assert len(b2.domains) == 12
    assert set(b1.domains).isdisjoint(set(b2.domains))


def test_orchard_capacity_is_finite_and_clipped(tmp_path: Path):
    orchard = _write_orchard_algo(tmp_path / "algos", [1000, 900])
    adapter = OrchardAdapter(_orchard_inspection(orchard), seed_strategy="sequential")

    cap = adapter.estimated_max_unique_capacity
    r1 = adapter.generate(cap + 10)
    r2 = adapter.generate(10)

    assert cap > 0
    assert r1.generated == cap
    assert r2.generated == 0


def test_orchard_resume_uses_saved_domain_index(tmp_path: Path):
    orchard = _write_orchard_algo(tmp_path / "algos", [1000, 900, 800, 700])
    first = OrchardAdapter(_orchard_inspection(orchard), seed_strategy="sequential")
    first_batch = first.generate(15)

    resume_idx = first_batch.last_effective_params["next_domain_index"]
    second = OrchardAdapter(
        _orchard_inspection(orchard, strategy={"resume_next_domain_index": resume_idx}),
        seed_strategy="sequential",
    )
    second_batch = second.generate(15)

    assert set(first_batch.domains).isdisjoint(set(second_batch.domains))


def test_orchard_exhausts_when_capacity_reached(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_orchard_algo(root, [2000, 1900, 1800, 1700])

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=500,
        batch_size=50,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    orchard_stats = stats["distribution_by_algorithm"]["orchard"]

    assert orchard_stats["status"] == "exhausted"
    assert orchard_stats["exhaustion_reason"] == "estimated_capacity_reached"
    assert orchard_stats["maximum_effective_quota"] == orchard_stats["profiling"]["estimated_max_unique_capacity"]


def test_orchard_resume_continues_from_saved_index(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_orchard_algo(root, [2000, 1900, 1800, 1700])

    cfg1 = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=12,
        batch_size=6,
        checkpoint_every=6,
    )
    p1 = run_pipeline(cfg1)
    state1 = read_json(out / "udcdga_generation_state.json")
    params1 = state1["plans"]["orchard"]["last_effective_params"]
    idx1 = params1["next_domain_index"]
    rec1 = params1["next_record_index"]
    assert p1["unique_domains"] == 12

    cfg2 = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=40,
        batch_size=50,
        checkpoint_every=20,
        resume=True,
    )
    p2 = run_pipeline(cfg2)
    state2 = read_json(out / "udcdga_generation_state.json")

    params2 = state2["plans"]["orchard"]["last_effective_params"]
    idx2 = params2["next_domain_index"]
    rec2 = params2["next_record_index"]

    assert p2["unique_domains"] == 40
    assert idx2 >= idx1
    assert rec2 >= rec1


def test_orchard_registry_routes_to_dedicated_adapter(tmp_path: Path):
    orchard = _write_orchard_algo(tmp_path / "algos", [1000, 900, 800])
    ins = _orchard_inspection(orchard)

    adapter = get_adapter(
        ins,
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, OrchardAdapter)
