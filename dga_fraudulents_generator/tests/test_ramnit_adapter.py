from __future__ import annotations

import json
from pathlib import Path

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.ramnit_adapter import RamnitAdapter
from dga_fraudulents_dataset.validation import validate_domain


DATE_ARGS = {
    "date_start": "2018-01-01",
    "date_end": "2030-12-31",
    "date_max_years_forward": 8,
    "date_max_years_backward": 8,
    "date_wrap_policy": "clamp",
}
REFERENCE_VECTOR = (
    Path(__file__).resolve().parents[2]
    / "validation"
    / "dga_reference_validation"
    / "vectors"
    / "ramnit_reference_raw.json"
)


def _inspection(*, defaults: dict | None = None, strategy: dict | None = None) -> AlgorithmInspection:
    path = Path(__file__).resolve().parents[1] / "dga_algorithms" / "ramnit"
    return AlgorithmInspection(
        algorithm_code="ramnit",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="get_domains",
        default_params=defaults or {},
        parameter_strategy=strategy or {},
    )


def _adapter(*, offset: int = 0, seeds: list[str] | None = None) -> RamnitAdapter:
    return RamnitAdapter(
        _inspection(
            defaults={
                "ramnit_seeds": seeds or ["16647BB4"],
                "ramnit_tlds": ["click", "com", "eu", "bid"],
                "ramnit_sld_min_length": 9,
                "ramnit_sld_max_length": 25,
                "ramnit_resume_next_offset": offset,
            }
        ),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )


def test_ramnit_registry_selects_dedicated_adapter_and_production_path_is_nonempty():
    adapter = get_adapter(
        _inspection(defaults={"ramnit_seeds": ["16647BB4"]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    result = adapter.generate(8)
    assert isinstance(adapter, RamnitAdapter)
    assert result.adapter_type == "ramnit_dedicated"
    assert len(result.domains) == 8
    assert all(validate_domain(domain).is_valid for domain in result.domains)


def test_ramnit_configuration_and_effective_parameters_are_explicit():
    adapter = _adapter(seeds=["16647BB4", "E7392D18"])
    result = adapter.generate(5)
    params = result.last_effective_params
    assert params["seeds"] == ["16647BB4", "E7392D18"]
    assert params["tlds"] == ["click", "com", "eu", "bid"]
    assert params["sld_min_length"] == 9
    assert params["sld_max_length"] == 25
    assert params["start_offset"] == 0
    assert params["next_offset"] == 5
    assert params["per_seed_next_offsets"] == {"16647BB4": 3, "E7392D18": 2}


def test_ramnit_batching_and_resume_preserve_one_ordered_sequence():
    complete = _adapter().generate(40).domains
    split_adapter = _adapter()
    split = split_adapter.generate(13).domains + split_adapter.generate(27).domains
    resumed = _adapter(offset=13).generate(27).domains
    assert split == complete
    assert resumed == complete[13:]


def test_ramnit_production_adapter_matches_independent_reference_vector():
    vector = json.loads(REFERENCE_VECTOR.read_text(encoding="utf-8"))
    expected = vector["expected_domains"]
    params = vector["parameters"]
    adapter = RamnitAdapter(
        _inspection(
            defaults={
                "ramnit_seeds": [params["seed"]],
                "ramnit_tlds": params["tlds"],
                "ramnit_sld_min_length": params["sld_min_length"],
                "ramnit_sld_max_length": params["sld_max_length"],
            }
        ),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )
    actual = adapter.generate(len(expected)).domains
    normalized_expected = [validate_domain(domain).normalized for domain in expected]
    normalized_actual = [validate_domain(domain).normalized for domain in actual]
    assert actual == expected
    assert normalized_actual == normalized_expected
