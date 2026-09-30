import json
from pathlib import Path

import pyarrow.parquet as pq

from dga_features_extractor.pipeline import BuildConfig, _assign_split, build_arff_dataset
from dga_features_extractor.schema import SAMPLE_FILE_NAME


def _make_config(tmp_path: Path, benign: Path, dga: Path, **overrides) -> BuildConfig:
    cfg = BuildConfig(
        benign_input=benign,
        dga_input=dga,
        output_dir=tmp_path / "processed_data",
        base_name="udcdga_dataset",
        metadata_dir=tmp_path / "processed_data" / "metadata",
        split_threshold_bytes=200000,
        encoding="utf-8",
        encoding_errors="replace",
        log_dir=tmp_path / "logs",
        log_level="INFO",
        buffer_size=4096,
        deduplicate=False,
        threads=1,
        batch_size=1000,
        temp_dir=tmp_path / "tmp",
        overwrite=True,
        relation_name="UDCDGA",
        manifest=False,
        progress_every=0,
        validate_output=True,
        schema_reference=Path("domains_output.arff"),
        compression_enabled=False,
        compression_format="none",
        keep_uncompressed_full_parts=True,
        sample_size=20,
        stats_json_name="udcdga_dataset_stats.json",
        write_manifest=True,
        write_integrity=True,
        write_stats=True,
        deduplicate_global=True,
        split_datasets=False,
        random_seed=123,
        audit_sample_size=10,
        parquet_row_group_size=1000,
        write_schema_description=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_integrity_and_manifest_reference_new_outputs(tmp_path: Path):
    benign = tmp_path / "benign.csv"
    dga = tmp_path / "dga.csv"
    benign.write_text("good.com\n", encoding="utf-8")
    dga.write_text("domain,algorithm_code\nbad.xyz,kraken\n", encoding="utf-8")

    config = _make_config(tmp_path, benign, dga)
    build_arff_dataset(config)

    manifest = json.loads((config.metadata_dir / "udcdga_dataset_manifest.json").read_text(encoding="utf-8"))
    integrity = json.loads((config.metadata_dir / "udcdga_dataset_integrity.json").read_text(encoding="utf-8"))

    assert manifest["dataset"]["main_export_format"] == "parquet"
    assert manifest["package_version"] == "1.0.0"
    assert manifest["schema"]["feature_count"] == 54
    assert manifest["split"]["assignment"] == "sha256(seed:domain)"
    assert manifest["sample"]["path"].endswith(SAMPLE_FILE_NAME)
    assert manifest["outputs"]["schema_description"]["path"].endswith("udcdga_ARFF_dataset_schema.json")

    output_paths = {item["path"] for item in integrity["output_files"]}
    assert any(path.endswith(".parquet") for path in output_paths)
    assert any(path.endswith(SAMPLE_FILE_NAME) for path in output_paths)
    assert any(path.endswith("udcdga_ARFF_dataset_schema.json") for path in output_paths)


def test_overlap_exclusion_and_balance_are_reported(tmp_path: Path):
    benign = tmp_path / "benign.csv"
    dga = tmp_path / "dga.csv"
    benign.write_text("same.com\nonly-benign.com\n", encoding="utf-8")
    dga.write_text("domain,algorithm_code\nsame.com,kraken\nonly-dga.net,kraken\n", encoding="utf-8")

    config = _make_config(tmp_path, benign, dga, deduplicate_global=True)
    manifest = build_arff_dataset(config)

    assert manifest["counts"]["benign_overlap_excluded"] == 1
    assert manifest["counts"]["benign_rows_written"] == manifest["counts"]["dga_rows_written"]
    assert manifest["counts"]["rows_written"] == 2


def test_split_datasets_are_reproducible_and_non_overlapping(tmp_path: Path):
    benign = tmp_path / "benign.csv"
    dga = tmp_path / "dga.csv"
    benign.write_text("\n".join(f"good{i}.com" for i in range(30)) + "\n", encoding="utf-8")
    dga.write_text(
        "domain,algorithm_code\n" + "\n".join(f"evil{i}.xyz,kraken" for i in range(30)) + "\n",
        encoding="utf-8",
    )

    config = _make_config(
        tmp_path,
        benign,
        dga,
        split_datasets=True,
        deduplicate_global=True,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_seed=999,
    )
    build_arff_dataset(config)

    train = set(pq.read_table(config.output_dir / "udcdga_dataset_train.parquet").column("DOMAIN").to_pylist())
    val = set(pq.read_table(config.output_dir / "udcdga_dataset_val.parquet").column("DOMAIN").to_pylist())
    test = set(pq.read_table(config.output_dir / "udcdga_dataset_test.parquet").column("DOMAIN").to_pylist())

    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_hash_split_is_deterministic_and_has_approximate_ratios():
    domains = [f"domain-{index}.example" for index in range(10_000)]
    first = [_assign_split(domain, 42, 0.8, 0.1) for domain in domains]
    second = [_assign_split(domain, 42, 0.8, 0.1) for domain in reversed(domains)]

    assert dict(zip(domains, first)) == dict(zip(reversed(domains), second))
    assert abs(first.count("train") / len(first) - 0.8) < 0.03
    assert abs(first.count("val") / len(first) - 0.1) < 0.02
    assert abs(first.count("test") / len(first) - 0.1) < 0.02
