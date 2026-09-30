from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import _adaptive_batch_request
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.mydoom_adapter import MydoomAdapter

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_mydoom_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "mydoom"
    return AlgorithmInspection(
        algorithm_code="mydoom",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_mydoom_explores_date_magic_and_number_axes():
    adapter = MydoomAdapter(
        _real_mydoom_inspection(defaults={"number_span": 128, "day_window": 120, "profile_slots": 512}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(300)
    params = r.last_effective_params
    assert params["date_explored_count"] > 1
    assert params["number_explored_count"] >= 40
    assert len(params["magic_explored"]) >= 1
    assert r.supported_parameter_axes == ["date", "magic", "number"]


def test_mydoom_high_yield_regression_guard():
    adapter = MydoomAdapter(
        _real_mydoom_inspection(defaults={"number_span": 512, "day_window": 365, "profile_slots": 2048}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(1600)
    ratio = len(set(r.domains)) / max(len(r.domains), 1)
    assert ratio >= 0.95


def test_mydoom_profile_and_generate_share_dedicated_behavior():
    adapter = MydoomAdapter(_real_mydoom_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(80)
    assert p["adapter_type"] == "mydoom_dedicated"
    assert r.adapter_type == "mydoom_dedicated"
    assert r.last_effective_params["generation_mode"] == "date_magic_number_schedule"
    assert "collision_growth_summary" in p
    assert "estimated_saturation_onset" in p
    assert "redistribution_absorption_reason" in p


def test_mydoom_late_stage_duplication_triggers_softer_batch_reduction():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="mydoom",
        path="/tmp/mydoom",
        category="seed_based",
        strategy="date_magic_number_schedule",
        entrypoint="/tmp/mydoom/dga.py",
        callable_name="dga",
        required_params=["date", "magic", "number"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
    )
    plan.last_batch_unique_yield_ratio = 0.72
    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 5000 // 20
    assert plan.adaptive_batch_mode == "mydoom_small"


def test_mydoom_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_mydoom_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, MydoomAdapter)
