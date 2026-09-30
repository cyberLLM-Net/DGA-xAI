from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import (
    GenerationError,
    _adaptive_batch_request,
    _evaluate_algorithm_health,
    run_pipeline,
)
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.nymaim_adapter import NymaimAdapter
from dga_fraudulents_dataset.utils import read_json


def _real_nymaim_inspection(*, strategy: dict | None = None) -> AlgorithmInspection:
    npath = Path(__file__).resolve().parents[1] / "dga_algorithms" / "nymaim"
    return AlgorithmInspection(
        algorithm_code="nymaim",
        path=str(npath),
        strategy="python_function",
        entrypoint=str(npath / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def _write_collision_nymaim(root: Path, *, pool: int = 6) -> Path:
    n = root / "nymaim"
    n.mkdir(parents=True, exist_ok=True)
    (n / "dga.py").write_text(
        "def dga(date, nr):\n"
        "    d = getattr(date, 'toordinal', lambda: 0)()\n"
        "    for i in range(nr):\n"
        f"        print(f'n{{(d + i) % {pool}}}.com')\n",
        encoding="utf-8",
    )
    return n


def test_nymaim_date_counter_variation_changes_output():
    adapter = NymaimAdapter(
        _real_nymaim_inspection(strategy={"domains_per_date": 16, "date_mode": "daily_forward"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r1 = adapter.generate(32)
    r2 = adapter.generate(32)
    assert r1.domains != r2.domains
    assert r2.last_effective_params["next_counter"] > r1.last_effective_params["next_counter"]


def test_nymaim_empirical_capacity_estimation_exists():
    adapter = NymaimAdapter(_real_nymaim_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert p["nymaim_medium_capacity"] is True
    assert p["estimated_max_unique_capacity"] >= p["sample_unique"]
    assert len(p["collision_curve"]) >= 3


def test_nymaim_batch_size_shrinks_under_declining_yield():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="nymaim",
        path="/tmp/nymaim",
        category="seed_based",
        strategy="date_counter_medium_capacity",
        entrypoint="/tmp/nymaim/dga.py",
        callable_name="dga",
        required_params=["date", "nr"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
        profiling={"nymaim_medium_capacity": True, "collision_prone": False},
    )
    plan.last_batch_unique_yield_ratio = 0.02

    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 50
    assert plan.adaptive_batch_mode == "nymaim_probe"


def test_nymaim_saturates_or_exhausts_under_duplicate_dominance(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_collision_nymaim(root, pool=4)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=250,
        batch_size=100,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    n = stats["distribution_by_algorithm"]["nymaim"]
    assert n["status"] in {"saturated", "exhausted"}
    if n["status"] == "exhausted":
        assert n["exhaustion_reason"] in {
            "nymaim_saturation_plateau",
            "collision_plateau",
            "zero_unique_batches",
            "observed_capacity_capped",
        }


def test_nymaim_effective_quota_capped_by_observed_capacity(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_collision_nymaim(root, pool=8)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=10000,
        batch_size=500,
        checkpoint_every=100,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    n = stats["distribution_by_algorithm"]["nymaim"]
    assert n["maximum_effective_quota"] <= n["profiling"]["estimated_max_unique_capacity"]


def test_nymaim_redistribution_reduced_when_yield_falls():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."))
    plan = AlgorithmPlan(
        algorithm_code="nymaim",
        path="/tmp/nymaim",
        category="seed_based",
        strategy="date_counter_medium_capacity",
        entrypoint="/tmp/nymaim/dga.py",
        callable_name="dga",
        required_params=["date", "nr"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=1000,
        planned_quota=1000,
        effective_quota=1000,
        profiling={"nymaim_medium_capacity": True},
        expected_diversity_score=0.1,
    )
    _evaluate_algorithm_health(cfg, plan, inserted=0, generated=1000, timed_out=False)
    assert plan.redistribution_eligibility is False
