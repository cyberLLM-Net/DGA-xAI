from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.config import AppConfig
from dga_fraudulents_dataset.generator import _adaptive_batch_request
from dga_fraudulents_dataset.models import AlgorithmInspection, AlgorithmPlan
from dga_fraudulents_dataset.newgoz_adapter import NewgozAdapter

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_newgoz_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "newgoz"
    return AlgorithmInspection(
        algorithm_code="newgoz",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="create_domain",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_newgoz_generates_multi_segment_alnum_domains():
    adapter = NewgozAdapter(_real_newgoz_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(200)
    assert len(r.domains) == 200
    for d in r.domains[:20]:
        label = d.split(".", 1)[0]
        assert len(label) >= 12
        assert any(c.isalpha() for c in label)
        assert any(c.isdigit() for c in label)


def test_newgoz_seed_date_traversal_changes_sequence():
    adapter = NewgozAdapter(
        _real_newgoz_inspection(defaults={"seq_span": 256, "day_window": 365}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r1 = adapter.generate(120)
    r2 = adapter.generate(120)
    assert r1.domains != r2.domains
    assert r2.last_effective_params["next_schedule_offset"] > r1.last_effective_params["next_schedule_offset"]
    assert r2.last_effective_params["date_explored_count"] >= 1
    assert r2.last_effective_params["seed_axis_explored_count"] >= 1


def test_newgoz_duplicates_below_threshold():
    adapter = NewgozAdapter(
        _real_newgoz_inspection(defaults={"profile_slots": 2048, "seq_span": 1000}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(2500)
    ratio = len(set(r.domains)) / max(len(r.domains), 1)
    assert ratio >= 0.9


def test_newgoz_profile_contains_capacity_and_dictionary_diagnostics():
    adapter = NewgozAdapter(_real_newgoz_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    assert p["adapter_type"] == "newgoz_dedicated"
    assert p["dictionary_size"] >= 36
    assert "word_count_distribution" in p
    assert p["estimated_capacity"] >= p["sample_unique"]
    assert p["estimated_saturation_onset"] >= 1


def test_newgoz_late_stage_yield_decline_triggers_slowdown():
    cfg = AppConfig(algorithms_root=Path("."), output_dir=Path("."), batch_size=5000)
    plan = AlgorithmPlan(
        algorithm_code="newgoz",
        path="/tmp/newgoz",
        category="date_based",
        strategy="date_seed_seq_schedule",
        entrypoint="/tmp/newgoz/dga.py",
        callable_name="create_domain",
        required_params=["seq_nr", "date"],
        default_params={},
        requires_seed=False,
        requires_date=True,
        target_count=10000,
        planned_quota=10000,
        effective_quota=10000,
    )
    plan.last_batch_unique_yield_ratio = 0.68
    ask = _adaptive_batch_request(cfg, plan, remaining_global=10000)
    assert ask <= 5000 // 24
    assert plan.adaptive_batch_mode == "newgoz_small"


def test_newgoz_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_newgoz_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, NewgozAdapter)
