from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dga_features_extractor import FeatureDefinition, FeatureRegistry, default_feature_registry
from dga_features_extractor.feature_registry import (
    feature_definition_checksum,
    feature_name_order_checksum,
)
from dga_features_extractor.features import compute_domain_features_subset, compute_registered_features
from dga_features_extractor.pipeline import BuildConfig, build_arff_dataset
from dga_features_extractor.schema import selected_feature_names
from dga_features_extractor.validator import validate_outputs


BASELINE_NAME_CHECKSUM = "fef951ea89729a42135bd6ef2135205cec35e916070e7bc4c30bce1565f24f2b"


def squared_length_definition(**overrides: object) -> FeatureDefinition:
    values = {
        "name": "domain_length_squared",
        "extractor": lambda context: float(len(context.normalized_domain) ** 2),
        "dtype": "float64",
        "description": "Square of the normalized domain string length, including internal dots.",
        "definition_version": "1",
    }
    values.update(overrides)
    return FeatureDefinition(**values)  # type: ignore[arg-type]


def config_for(tmp_path: Path, registry: FeatureRegistry, suffix: str = "") -> BuildConfig:
    root = tmp_path / f"run{suffix}"
    root.mkdir()
    benign = root / "benign.csv"
    dga = root / "dga.csv"
    benign.write_text("good.com\n", encoding="utf-8")
    dga.write_text("domain,algorithm_code\nbad.xyz,testdga\n", encoding="utf-8")
    output = root / "output"
    return BuildConfig(
        benign_input=benign,
        dga_input=dga,
        output_dir=output,
        base_name="udcdga_dataset",
        metadata_dir=output / "metadata",
        split_threshold_bytes=200_000,
        encoding="utf-8",
        encoding_errors="replace",
        log_dir=root / "logs",
        log_level="INFO",
        buffer_size=1024,
        deduplicate=False,
        threads=1,
        batch_size=100,
        temp_dir=root / "tmp",
        overwrite=True,
        relation_name="UDCDGA",
        manifest=True,
        progress_every=0,
        validate_output=True,
        schema_reference=None,
        compression_enabled=False,
        compression_format="none",
        keep_uncompressed_full_parts=True,
        sample_size=10,
        stats_json_name="stats.json",
        write_manifest=True,
        write_integrity=True,
        write_stats=True,
        random_seed=42,
        parquet_row_group_size=100,
        write_schema_description=True,
        feature_registry=registry,
    )


def test_default_registry_is_exactly_backward_compatible() -> None:
    registry = default_feature_registry()
    assert len(registry.definitions) == 54
    assert list(registry.names) == selected_feature_names()
    assert feature_name_order_checksum(registry.names) == BASELINE_NAME_CHECKSUM
    domain = "a1-b2.example"
    legacy = compute_domain_features_subset(domain, selected_feature_names())
    registered = compute_registered_features(domain, registry)
    assert registered == legacy


def test_registration_validation_and_immutability() -> None:
    default = default_feature_registry()
    custom = default.with_feature(squared_length_definition())
    assert len(default.definitions) == 54
    assert len(custom.definitions) == 55
    assert custom.names[-1] == "domain_length_squared"
    with pytest.raises(ValueError, match="Duplicate"):
        custom.with_feature(squared_length_definition())
    for reserved in ("DOMAIN", "CLASS", "LABEL", "domain"):
        with pytest.raises(ValueError, match="Reserved"):
            squared_length_definition(name=reserved)
    with pytest.raises(TypeError, match="callable"):
        squared_length_definition(extractor=3)
    with pytest.raises(ValueError, match="dtype"):
        squared_length_definition(dtype="int64")
    with pytest.raises(ValueError, match="Invalid"):
        squared_length_definition(name="bad feature")
    assert compute_registered_features("abc.com", custom)["domain_length_squared"] == 49.0


def test_definition_checksum_covers_order_name_dtype_and_version() -> None:
    base = default_feature_registry()
    custom = base.with_feature(squared_length_definition())
    baseline = feature_definition_checksum(base.definitions)
    assert feature_definition_checksum(custom.definitions) != baseline
    assert feature_name_order_checksum(custom.names) != BASELINE_NAME_CHECKSUM
    assert feature_definition_checksum(custom.without("1G_75P").definitions) != feature_definition_checksum(custom.definitions)
    swapped = custom.reordered((custom.names[1], custom.names[0], *custom.names[2:]))
    assert feature_definition_checksum(swapped.definitions) != feature_definition_checksum(custom.definitions)
    renamed = FeatureRegistry((replace(custom.definitions[0], name="RENAMED"), *custom.definitions[1:]))
    assert feature_definition_checksum(renamed.definitions) != feature_definition_checksum(custom.definitions)
    dtype_payload = json.loads(json.dumps([item.metadata() for item in custom.definitions]))
    dtype_payload[-1]["dtype"] = "float32"
    canonical_dtype = [{k: item[k] for k in ("definition_version", "dtype", "name")} for item in dtype_payload]
    import hashlib
    dtype_digest = hashlib.sha256(json.dumps(canonical_dtype, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert dtype_digest != feature_definition_checksum(custom.definitions)
    versioned = FeatureRegistry((*base.definitions, replace(squared_length_definition(), definition_version="2")))
    assert feature_definition_checksum(versioned.definitions) != feature_definition_checksum(custom.definitions)


def test_custom_feature_propagates_and_validates_from_manifest(tmp_path: Path) -> None:
    default = default_feature_registry()
    custom = default.with_feature(squared_length_definition())
    cfg = config_for(tmp_path, custom)
    manifest = build_arff_dataset(cfg)

    table = pq.read_table(cfg.output_dir / "udcdga_dataset.parquet")
    assert len(custom.definitions) == 55 and len(default.definitions) == 54
    assert "domain_length_squared" in table.column_names
    for domain, value in zip(table.column("DOMAIN").to_pylist(), table.column("domain_length_squared").to_pylist()):
        assert value == float(len(domain) ** 2)

    arff = (cfg.output_dir / "udcdga_ARFF_dataset_samples.arff").read_text(encoding="utf-8")
    assert "@ATTRIBUTE domain_length_squared NUMERIC" in arff
    schema = json.loads((cfg.metadata_dir / "udcdga_ARFF_dataset_schema.json").read_text(encoding="utf-8"))
    custom_field = next(item for item in schema["fields"] if item["field_name"] == "domain_length_squared")
    assert custom_field["storage_type"] == "float64"
    assert custom_field["definition_version"] == "1"
    assert custom_field["description"] == squared_length_definition().description

    assert manifest["schema"]["feature_count"] == 55
    assert manifest["schema"]["feature_names"][-1] == "domain_length_squared"
    assert manifest["schema"]["feature_name_order_checksum"] != BASELINE_NAME_CHECKSUM
    integrity = json.loads((cfg.metadata_dir / "udcdga_dataset_integrity.json").read_text(encoding="utf-8"))
    assert integrity["metadata"]["feature_count"] == 55
    assert integrity["metadata"]["feature_names"][-1] == "domain_length_squared"
    assert integrity["metadata"]["feature_definition_checksum"] == manifest["schema"]["feature_definition_checksum"]

    result = validate_outputs(
        output_dir=cfg.output_dir,
        metadata_dir=cfg.metadata_dir,
        base_name=cfg.base_name,
        split_threshold_bytes=cfg.split_threshold_bytes,
        encoding=cfg.encoding,
        sample_file_name="udcdga_ARFF_dataset_samples.arff",
        require_sample=True,
        expected_columns=None,
    )
    assert result.valid, result.errors


def test_repeated_custom_builds_have_identical_logical_output(tmp_path: Path) -> None:
    registry = default_feature_registry().with_feature(squared_length_definition())
    first = config_for(tmp_path, registry, "1")
    second = config_for(tmp_path, registry, "2")
    first_manifest = build_arff_dataset(first)
    second_manifest = build_arff_dataset(second)
    assert pq.read_table(first.output_dir / "udcdga_dataset.parquet").to_pylist() == pq.read_table(
        second.output_dir / "udcdga_dataset.parquet"
    ).to_pylist()
    assert first_manifest["schema"] == second_manifest["schema"]
