from __future__ import annotations

import tomllib
from pathlib import Path

import domain_aggregator


def test_project_metadata_matches_package_version() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert metadata["version"] == domain_aggregator.__version__ == "1.0.0"
    assert metadata["requires-python"] == ">=3.11"
    assert metadata["license"]["text"] == "Apache License 2.0"
    assert metadata["authors"] == [
        {"name": "Victor Carneiro", "email": "victor.carneiro@udc.es"}
    ]
