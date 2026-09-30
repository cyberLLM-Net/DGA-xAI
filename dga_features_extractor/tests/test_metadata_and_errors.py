from __future__ import annotations

import tomllib
from pathlib import Path

import dga_features_extractor
from dga_features_extractor.pipeline import BuildStats, _coerce_feature_values


def test_project_metadata_matches_package_version() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert metadata["version"] == dga_features_extractor.__version__ == "1.0.0"
    assert metadata["requires-python"] == ">=3.11"
    assert metadata["license"]["text"] == "Apache License 2.0"
    assert metadata["authors"] == [
        {"name": "Victor Carneiro", "email": "victor.carneiro@udc.es"}
    ]


def test_feature_fallbacks_are_counted_by_reason() -> None:
    stats = BuildStats()
    values = _coerce_feature_values(
        {"valid": 1.5, "invalid": "not-a-number", "infinite": float("inf")},
        ["valid", "missing", "invalid", "infinite"],
        stats,
    )

    assert values == [1.5, 0.0, 0.0, 0.0]
    assert stats.feature_missing == 1
    assert stats.feature_invalid_numeric == 2
    assert stats.feature_errors == 3
