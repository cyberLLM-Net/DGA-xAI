from __future__ import annotations

from pathlib import Path

import pytest

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.fobber_adapter import FobberAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.validation import validate_domain

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_fobber_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dga_algorithms" / "fobber"


def _fobber_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = _real_fobber_path()
    return AlgorithmInspection(
        algorithm_code="fobber",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_fobber_adapter_generates_without_uninitialized_nr():
    adapter = FobberAdapter(_fobber_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    r = adapter.generate(40)
    assert r.generated == 40
    assert not any("uninitialized_local_state" in e for e in r.errors)
    assert r.last_effective_params["nr_source"] == "adapter_initialized_counter"


def test_fobber_example_style_domains_and_tlds_are_valid():
    adapter_v1 = FobberAdapter(
        _fobber_inspection(defaults={"fobber_variant": "v1"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    adapter_v2 = FobberAdapter(
        _fobber_inspection(defaults={"fobber_variant": "v2"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r1 = adapter_v1.generate(1).domains[0]
    r2 = adapter_v2.generate(1).domains[0]

    assert r1 == "vhkintjtksyxgjrzz.net"
    assert r2 == "drohppbkxj.com"
    assert validate_domain(r1).is_valid is True
    assert validate_domain(r2).is_valid is True


def test_fobber_variant_selection_is_explicit_and_deterministic():
    adapter = FobberAdapter(
        _fobber_inspection(defaults={"fobber_variant": "v1"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    r = adapter.generate(30)
    assert [x["variant"] for x in p["per_variant"]] == ["v1"]
    assert all(d.endswith(".net") for d in r.domains)
    assert r.last_effective_params["selected_variants"] == ["v1"]


def test_fobber_profiling_and_generation_share_adapter_path():
    adapter = FobberAdapter(_fobber_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(50)
    assert p["adapter_type"] == "fobber_dedicated"
    assert r.adapter_type == "fobber_dedicated"
    assert r.last_effective_params["generation_mode"] == "variant_counter_structured"


def test_fobber_resume_counter_schedule_progresses_without_replay():
    first = FobberAdapter(
        _fobber_inspection(defaults={"fobber_variant": "v1"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    b1 = first.generate(20)
    resume = int(b1.last_effective_params["v1_next_counter"])

    second = FobberAdapter(
        _fobber_inspection(
            strategy={"resume_v1_next_counter": resume},
            defaults={"fobber_variant": "v1"},
        ),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    b2 = second.generate(20)
    assert set(b1.domains).isdisjoint(set(b2.domains))


def test_fobber_unsupported_variant_is_explicit_error():
    with pytest.raises(RuntimeError) as exc:
        FobberAdapter(
            _fobber_inspection(defaults={"fobber_variant": "v9"}),
            seed_strategy="sequential",
            date_strategy="daily_forward",
        )
    assert "unsupported_variant" in str(exc.value)


def test_fobber_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _fobber_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, FobberAdapter)
