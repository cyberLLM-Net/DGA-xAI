from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from dga_fraudulents_dataset.adapter_registry import get_adapter
from dga_fraudulents_dataset.models import AlgorithmInspection
from dga_fraudulents_dataset.tinba_adapter import TINBA_CONFIGURATIONS, TinbaAdapter
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
    / "tinba_reference_raw.json"
)


def _inspection(*, defaults: dict | None = None, strategy: dict | None = None) -> AlgorithmInspection:
    path = Path(__file__).resolve().parents[1] / "dga_algorithms" / "tinba"
    return AlgorithmInspection(
        algorithm_code="tinba",
        path=str(path),
        strategy="python_function",
        entrypoint=str(path / "dga.py"),
        callable_name="dga",
        default_params=defaults or {},
        parameter_strategy=strategy or {},
    )


def _adapter(*, configuration: int = 3, offset: int = 0) -> TinbaAdapter:
    return TinbaAdapter(
        _inspection(
            defaults={
                "tinba_configurations": [configuration],
                "tinba_resume_next_offset": offset,
            }
        ),
        seed_strategy="sequential",
        date_strategy="daily_forward",
    )


def test_tinba_configurations_are_coherent_and_immutable():
    cfg = TINBA_CONFIGURATIONS[2]
    assert cfg.configuration_id == 3
    assert cfg.seed == "yqokqFC2TPBFfJcG"
    assert cfg.initial_domain == "watchthisnow.xyz"
    assert cfg.tlds == ("pw", "us", "xyz", "club")
    assert cfg.num_domains == 100
    with pytest.raises(FrozenInstanceError):
        cfg.seed = "different"  # type: ignore[misc]


def test_tinba_registry_selects_dedicated_adapter_and_production_path_is_nonempty():
    adapter = get_adapter(
        _inspection(defaults={"tinba_configurations": [3]}),
        seed_strategy="sequential",
        date_strategy="daily_forward",
        timeout_seconds=5,
        batch_timeout_seconds=5,
        max_cli_invocations_per_batch=5,
        **DATE_ARGS,
    )
    result = adapter.generate(8)
    assert isinstance(adapter, TinbaAdapter)
    assert result.adapter_type == "tinba_dedicated"
    assert len(result.domains) == 8
    assert all(validate_domain(domain).is_valid for domain in result.domains)


def test_tinba_batching_resume_exhaustion_and_effective_parameters():
    complete = _adapter().generate(401).domains
    split_adapter = _adapter()
    first = split_adapter.generate(137)
    second = split_adapter.generate(300)
    exhausted = split_adapter.generate(10)
    resumed = _adapter(offset=137).generate(300)

    assert first.domains + second.domains == complete
    assert len(second.domains) == 264
    assert resumed.domains == complete[137:]
    assert exhausted.domains == []
    assert exhausted.last_effective_params["exhausted"] is True
    assert exhausted.last_effective_params["remaining_unique_capacity"] == 0
    assert first.last_effective_params["configurations"][0] == {
        "configuration_id": 3,
        "seed": "yqokqFC2TPBFfJcG",
        "initial_domain": "watchthisnow.xyz",
        "tlds": ["pw", "us", "xyz", "club"],
        "num_domains": 100,
        "output_count": 401,
    }


def test_tinba_production_adapter_matches_independent_reference_vector():
    vector = json.loads(REFERENCE_VECTOR.read_text(encoding="utf-8"))
    expected = vector["expected_domains"]
    params = vector["parameters"]
    adapter = _adapter(configuration=params["configuration"])
    actual = adapter.generate(len(expected)).domains
    normalized_expected = [validate_domain(domain).normalized for domain in expected]
    normalized_actual = [validate_domain(domain).normalized for domain in actual]
    assert actual == expected
    assert normalized_actual == normalized_expected
