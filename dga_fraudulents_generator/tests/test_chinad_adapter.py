from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import _adaptive_batch_request
from dga_fraudulents_dataset.chinad_adapter import ChinadAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_chinad_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "chinad"
    return AlgorithmInspection(
        algorithm_code="chinad",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_chinad_high_yield_regression_guard():
    adapter = ChinadAdapter(
        _real_chinad_inspection(defaults={"profile_slots": 4096, "day_window": 3650}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(3000)
    ratio = len(set(r.domains)) / max(len(r.domains), 1)
    assert ratio >= 0.98


def test_chinad_multiple_tlds_are_produced():
    adapter = ChinadAdapter(_real_chinad_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(2000)
    tlds = {("." + d.rsplit(".", 1)[-1]) for d in r.domains}
    assert len(tlds) >= 5
    assert {".com", ".org", ".net"}.issubset(tlds)


def test_chinad_profile_contains_collision_and_capacity_diagnostics():
    adapter = ChinadAdapter(_real_chinad_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert p["adapter_type"] == "chinad_dedicated"
    assert p["estimated_capacity"] >= p["sample_unique"]
    assert p["estimated_saturation_onset"] >= 1
    assert "collision_growth_summary" in p
    assert "tld_distribution" in p
    assert "redistribution_absorption_reason" in p


def test_chinad_traversal_is_deterministic_and_not_collapsed():
    a1 = ChinadAdapter(_real_chinad_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    a2 = ChinadAdapter(_real_chinad_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r1 = a1.generate(600)
    r2 = a2.generate(600)
    assert r1.domains == r2.domains
    assert r1.last_effective_params["nr_explored_count"] >= 128
    assert r1.last_effective_params["date_explored_count"] >= 2


def test_chinad_late_stage_yield_decline_triggers_soft_reduction():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="chinad",
        path="/tmp/chinad",
        category="date_based",
        strategy="date_nr_schedule",
        entrypoint="/tmp/chinad/dga.py",
        callable_name="dga",
        required_params=["date"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
    )
    plan.last_batch_unique_yield_ratio = 0.8
    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 5000 // 18
    assert plan.adaptive_batch_mode == "chinad_small"


def test_chinad_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_chinad_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, ChinadAdapter)
