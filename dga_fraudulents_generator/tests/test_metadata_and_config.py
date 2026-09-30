from __future__ import annotations

import json
import tomllib
from pathlib import Path

import dga_fraudulents_dataset
from dga_fraudulents_dataset.cli import _explicit_cli_destinations, _merge_config, build_parser


def test_project_metadata_matches_package_version() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert metadata["version"] == dga_fraudulents_dataset.__version__ == "1.0.0"
    assert metadata["requires-python"] == ">=3.11"
    assert metadata["license"]["text"] == "Apache License 2.0"
    assert metadata["authors"] == [
        {"name": "Victor Carneiro", "email": "victor.carneiro@udc.es"}
    ]


def test_explicit_cli_values_override_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"target_count": 200, "batch_size": 25}),
        encoding="utf-8",
    )
    argv = ["--config", str(config_path), "--target-count", "300"]
    parser = build_parser()
    args = parser.parse_args(argv)

    merged = _merge_config(args, _explicit_cli_destinations(parser, argv))

    assert merged.target_count == 300
    assert merged.batch_size == 25
