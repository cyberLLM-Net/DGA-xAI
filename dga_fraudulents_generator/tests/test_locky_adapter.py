from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.locky_adapter import LockyAdapter
from dga_fraudulents_dataset.models import AlgorithmInspection

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_locky_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "locky"
    return AlgorithmInspection(
        algorithm_code="locky",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dgav2.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def _write_fake_locky(root: Path) -> Path:
    d = root / "locky"
    d.mkdir(parents=True, exist_ok=True)
    (d / "dgav2.py").write_text(
        "config={1:{'seed':1},2:{'seed':2}}\n"
        "def dga(date, config_nr, domain_nr):\n"
        "  return f'v2-{date.strftime(\"%Y%m%d\")}-{config_nr}-{domain_nr}.com'\n",
        encoding="utf-8",
    )
    (d / "dgav3.py").write_text(
        "config={1:{'seed':1},2:{'seed':2},3:{'seed':3}}\n"
        "def dga(date, config_nr, domain_nr):\n"
        "  return f'v3-fixed-{config_nr}.com'\n",
        encoding="utf-8",
    )
    return d


def _fake_locky_inspection(path: Path, *, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="locky",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dgav2.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_locky_variant_selection_between_v2_and_v3_is_explicit_and_deterministic(tmp_path: Path):
    path = _write_fake_locky(tmp_path / "algos")
    auto = LockyAdapter(_fake_locky_inspection(path), seed_strategy="sequential", date_strategy="daily_forward")
    p_auto = auto.profile()
    assert p_auto["best_variant"] == "v2"
    assert p_auto["selected_variants"] == ["v2"]

    force_v3 = LockyAdapter(
        _fake_locky_inspection(path, defaults={"locky_variant": "v3"}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p_v3 = force_v3.profile()
    assert p_v3["selected_variants"] == ["v3"]


def test_locky_explores_config_nr_and_domain_nr_axes():
    adapter = LockyAdapter(
        _real_locky_inspection(defaults={"locky_variant": "v2", "domain_nr_span": 64, "profile_slots": 256}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(200)
    params = r.last_effective_params
    assert params["domain_nr_explored_count"] > 10
    assert len(params["config_nr_explored"]) >= 2
    assert r.supported_parameter_axes == ["date", "config_nr", "domain_nr"]


def test_locky_near_perfect_uniqueness_regression_guard():
    adapter = LockyAdapter(
        _real_locky_inspection(defaults={"locky_variant": "v3", "domain_nr_span": 512}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(1200)
    ratio = len(set(r.domains)) / max(len(r.domains), 1)
    assert ratio >= 0.995


def test_locky_profile_and_generate_share_same_adapter_behavior():
    adapter = LockyAdapter(_real_locky_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(80)
    assert p["adapter_type"] == "locky_dedicated"
    assert r.adapter_type == "locky_dedicated"
    assert r.last_effective_params["generation_mode"] == "date_config_domain_schedule"
    assert "selected_implementation_files" in p
    assert "redistribution_absorption_reason" in p


def test_locky_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_locky_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, LockyAdapter)
