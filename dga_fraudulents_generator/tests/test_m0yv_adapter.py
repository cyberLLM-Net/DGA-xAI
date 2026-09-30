from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import (
    GenerationError,
    _adaptive_batch_request,
    _redistribute_from_algorithm,
    run_pipeline,
)
from dga_fraudulents_dataset.m0yv_adapter import M0yvAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.utils import read_json


def _write_m0yv_algo(root: Path, *, collision_pool: int = 8) -> Path:
    m0yv = root / "m0yv"
    m0yv.mkdir(parents=True, exist_ok=True)

    # Heavy-collision variant: many seeds collapse into a tiny .biz space.
    (m0yv / "dga.py").write_text(
        "def dga(seed):\n"
        "    for i in range(128):\n"
        f"        yield f'd{{(seed + i) % {collision_pool}}}.biz'\n",
        encoding="utf-8",
    )
    # td variant differs deterministically by date ordinal.
    (m0yv / "dga-td.py").write_text(
        "def dga(seed, date):\n"
        "    d = getattr(date, 'toordinal', lambda: 0)()\n"
        "    for i in range(128):\n"
        f"        yield f'td{{(seed + d + i) % {collision_pool}}}.biz'\n",
        encoding="utf-8",
    )
    return m0yv


def _inspection(m0yv_path: Path, *, strategy: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="m0yv",
        path=str(m0yv_path),
        strategy="python_function",
        entrypoint=str(m0yv_path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
    )


def test_m0yv_variant_selection_is_explicit_and_deterministic(tmp_path: Path):
    m0yv_path = _write_m0yv_algo(tmp_path / "algos")

    a_default = M0yvAdapter(_inspection(m0yv_path), seed_strategy="sequential", date_strategy="daily_forward")
    p_default = a_default.profile()
    assert p_default["selected_variant"] == "dga"
    assert p_default["implementation_file"].endswith("dga.py")

    a_td = M0yvAdapter(
        _inspection(m0yv_path, strategy={"m0yv_variant": "td", "td_date_mode": "daily_window"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p_td = a_td.profile()
    assert p_td["selected_variant"] == "td"
    assert p_td["implementation_file"].endswith("dga-td.py")
    assert p_td["available_variants"] == {"dga": "dga.py", "td": "dga-td.py"}


def test_m0yv_capacity_estimation_detects_high_duplication(tmp_path: Path):
    m0yv_path = _write_m0yv_algo(tmp_path / "algos", collision_pool=6)
    adapter = M0yvAdapter(_inspection(m0yv_path), seed_strategy="sequential", date_strategy="daily_forward")
    profile = adapter.profile()

    assert profile["collision_prone"] is True
    assert profile["sample_unique_yield"] < 0.1
    assert profile["estimated_max_unique_capacity"] <= int(profile["sample_unique"] * 1.25)
    assert len(profile["collision_curve"]) >= 3


def test_m0yv_batch_size_shrinks_aggressively_when_low_yield():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="m0yv",
        path="/tmp/m0yv",
        category="seed_based",
        strategy="seed_collision_prone",
        entrypoint="/tmp/m0yv/dga.py",
        callable_name="dga",
        required_params=["seed"],
        default_params={},
        requires_seed=True,
        requires_date=False,
        target_count=20000,
        planned_quota=20000,
        effective_quota=20000,
        profiling={"collision_prone": True},
    )
    plan.last_batch_unique_yield_ratio = 0.0

    ask = _adaptive_batch_request(cfg, plan, remaining_global=20000)
    assert ask <= 25
    assert plan.adaptive_batch_mode == "collision_probe"


def test_m0yv_exhausts_under_repeated_duplication(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_m0yv_algo(root, collision_pool=4)

    cfg = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=200,
        batch_size=100,
        checkpoint_every=20,
        exhausted_after_zero_unique_batches=3,
    )
    try:
        run_pipeline(cfg)
    except GenerationError:
        pass

    stats = read_json(out / "udcdga_dga_domains_stats.json")
    m = stats["distribution_by_algorithm"]["m0yv"]

    assert m["status"] in {"saturated", "exhausted"}
    if m["status"] == "exhausted":
        assert m["exhaustion_reason"] in {"collision_plateau", "observed_capacity_capped", "zero_unique_batches"}
    assert m["profiling"]["collision_prone"] is True
    assert m["profiling"]["implementation_file"].endswith("dga.py")
    assert len(m["profiling"]["collision_curve"]) >= 3


def test_redistribution_not_assigned_into_saturated_m0yv():
    donor = AlgorithmPlan(
        algorithm_code="donor",
        path="/d",
        category="x",
        strategy="python_function",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        unique_valid_count=0,
        status="discarded",
    )
    m0yv = AlgorithmPlan(
        algorithm_code="m0yv",
        path="/m",
        category="x",
        strategy="seed_collision_prone",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=True,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        maximum_effective_quota=300,
        unique_valid_count=10,
        status="saturated",
        redistribution_eligibility=False,
        capacity_score=0.01,
    )
    good = AlgorithmPlan(
        algorithm_code="good",
        path="/g",
        category="x",
        strategy="python_function",
        entrypoint=None,
        callable_name=None,
        required_params=[],
        default_params={},
        requires_seed=False,
        requires_date=False,
        target_count=100,
        planned_quota=100,
        effective_quota=100,
        maximum_effective_quota=300,
        unique_valid_count=50,
        status="usable",
        redistribution_eligibility=True,
        capacity_score=2.0,
    )

    moved = _redistribute_from_algorithm({"donor": donor, "m0yv": m0yv, "good": good}, "donor", "donor_down")
    assert moved > 0
    assert good.effective_quota > 100
    assert m0yv.effective_quota == 100


def test_m0yv_resume_continues_seed_offset(tmp_path: Path):
    root = tmp_path / "algos"
    out = tmp_path / "results"
    _write_m0yv_algo(root, collision_pool=10)

    cfg1 = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=50,
        batch_size=40,
        checkpoint_every=20,
    )
    try:
        run_pipeline(cfg1)
    except GenerationError:
        pass
    s1 = read_json(out / "udcdga_generation_state.json")
    off1 = s1["plans"]["m0yv"]["last_effective_params"]["next_seed_offset"]

    cfg2 = AppConfig(
        algorithms_root=root,
        output_dir=out,
        target_count=80,
        batch_size=40,
        checkpoint_every=20,
        resume=True,
    )
    try:
        run_pipeline(cfg2)
    except GenerationError:
        pass
    s2 = read_json(out / "udcdga_generation_state.json")
    off2 = s2["plans"]["m0yv"]["last_effective_params"]["next_seed_offset"]

    assert off2 >= off1
