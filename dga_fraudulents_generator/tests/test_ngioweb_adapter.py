from __future__ import annotations

from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.ngioweb_adapter import NgiowebAdapter
from dga_fraudulents_dataset.validation import validate_domain

DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}


def _real_ngioweb_inspection(*, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    p = Path(__file__).resolve().parents[1] / "dga_algorithms" / "ngioweb"
    return AlgorithmInspection(
        algorithm_code="ngioweb",
        path=str(p),
        strategy="python_function",
        entrypoint=str(p / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def _write_fake_ngioweb(root: Path) -> Path:
    n = root / "ngioweb"
    n.mkdir(parents=True, exist_ok=True)
    (n / "dga.py").write_text(
        "class Rand:\n"
        "  def __init__(self, seed):\n"
        "    self.r = seed & 0xffffffff\n"
        "  def rand(self, mod):\n"
        "    self.r = (1664525*self.r + 1013904223) & 0xffffffff\n"
        "    return self.r % mod\n"
        "def dga(r):\n"
        "  return f\"n{r.rand(100000)}.com\"\n",
        encoding="utf-8",
    )
    return n


def _fake_ngioweb_inspection(path: Path, *, strategy: dict | None = None, defaults: dict | None = None) -> AlgorithmInspection:
    return AlgorithmInspection(
        algorithm_code="ngioweb",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        parameter_strategy=strategy or {},
        default_params=defaults or {},
    )


def test_ngioweb_int_seed_is_normalized_to_rng_object():
    adapter = NgiowebAdapter(
        _real_ngioweb_inspection(defaults={"ngioweb_seeds": [0x56EDC15], "profile_slots": 64}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    p = adapter.profile()
    sched = p["tested_parameter_schedule"]
    assert sched
    assert sched[0]["types_before"]["seed"] == "int"
    assert sched[0]["types_after"]["rng"] == "Rand"


def test_ngioweb_no_rand_attribute_errors_after_adapter_fix(tmp_path: Path):
    path = _write_fake_ngioweb(tmp_path / "algos")
    adapter = NgiowebAdapter(
        _fake_ngioweb_inspection(path, defaults={"ngioweb_seeds": [1, 2, 3]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(40)
    assert r.generated > 0
    assert not any("expected_random_object" in e for e in r.errors)
    assert all(validate_domain(d).is_valid for d in r.domains)


def test_ngioweb_example_style_outputs_pass_validation():
    adapter = NgiowebAdapter(
        _real_ngioweb_inspection(defaults={"ngioweb_seeds": [0x56EDC15]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    r = adapter.generate(1)
    assert r.domains[0] == "minihileth-subatudofy.org"
    assert validate_domain(r.domains[0]).is_valid is True


def test_ngioweb_profiling_and_generation_share_adapter_behavior():
    adapter = NgiowebAdapter(_real_ngioweb_inspection(), seed_strategy="sequential", date_strategy="daily_forward")
    p = adapter.profile()
    r = adapter.generate(25)
    assert p["adapter_type"] == "ngioweb_dedicated"
    assert r.adapter_type == "ngioweb_dedicated"
    assert r.last_effective_params["generation_mode"] == "rng_object_seed_normalized"


def test_ngioweb_registry_routes_to_dedicated_adapter():
    adapter = get_adapter(
        _real_ngioweb_inspection(),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    assert isinstance(adapter, NgiowebAdapter)
