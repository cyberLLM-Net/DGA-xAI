import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

from dga_features_extractor.pipeline import BuildConfig, build_arff_dataset
from dga_features_extractor.schema import SAMPLE_FILE_NAME, full_dataset_columns, selected_feature_names


class TestPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="dga_test_"))
        self.dataset_dir = self.tmp_dir / "raw_data"
        self.output_dir = self.tmp_dir / "processed_data"
        self.metadata_dir = self.output_dir / "metadata"
        self.log_dir = self.tmp_dir / "logs"
        self.temp_dir = self.tmp_dir / "tmp"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.benign_file = self.dataset_dir / "udcdga_benigns_domains.csv"
        self.dga_file = self.dataset_dir / "udcdga_dga_domains.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_inputs(self, benign_rows: list[str], dga_rows: list[str]) -> None:
        self.benign_file.write_text("\n".join(benign_rows) + "\n", encoding="utf-8")
        self.dga_file.write_text("\n".join(dga_rows) + "\n", encoding="utf-8")

    def _config(self, **overrides) -> BuildConfig:
        cfg = BuildConfig(
            benign_input=self.benign_file,
            dga_input=self.dga_file,
            output_dir=self.output_dir,
            base_name="udcdga_dataset",
            metadata_dir=self.metadata_dir,
            split_threshold_bytes=200000,
            encoding="utf-8",
            encoding_errors="replace",
            log_dir=self.log_dir,
            log_level="INFO",
            buffer_size=1024,
            deduplicate=False,
            threads=1,
            batch_size=500,
            temp_dir=self.temp_dir,
            overwrite=True,
            relation_name="UDCDGA",
            manifest=True,
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
            random_seed=42,
            audit_sample_size=10,
            parquet_row_group_size=100,
            write_schema_description=True,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def _read_sample_rows(self, path: Path) -> list[list[str]]:
        rows = []
        in_data = False
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.upper() == "@DATA":
                    in_data = True
                    continue
                if not in_data:
                    continue
                rows.append(next(csv.reader([line], delimiter=",", quotechar="'", escapechar="\\")))
        return rows

    def test_default_output_schema_balanced_and_semantics(self) -> None:
        self._write_inputs(
            ["good.com", "safe.net"],
            [
                "domain,algorithm_code",
                "bad.xyz,kraken",
                "evil.org,banjori",
            ],
        )
        manifest = build_arff_dataset(self._config())

        parquet_out = self.output_dir / "udcdga_dataset.parquet"
        table = pq.read_table(parquet_out)

        expected_columns = full_dataset_columns(selected_feature_names())
        self.assertEqual(table.column_names, expected_columns)
        self.assertEqual(len(selected_feature_names()), 54)
        self.assertEqual(len(table.column_names), 57)
        self.assertTrue(pa.types.is_string(table.schema.field("DOMAIN").type))
        for feature_name in selected_feature_names():
            self.assertTrue(pa.types.is_float64(table.schema.field(feature_name).type))
        self.assertTrue(pa.types.is_string(table.schema.field("CLASS").type))
        self.assertTrue(pa.types.is_int8(table.schema.field("LABEL").type))
        self.assertNotIn("class", table.column_names)

        labels = table.column("LABEL").to_pylist()
        domains = table.column("DOMAIN").to_pylist()
        classes = table.column("CLASS").to_pylist()
        self.assertEqual(labels.count(0), labels.count(1))

        benign_domains = {d for d, l in zip(domains, labels) if l == 0}
        dga_domains = {d for d, l in zip(domains, labels) if l == 1}
        self.assertTrue(benign_domains.isdisjoint(dga_domains))

        for domain, class_value, label in zip(domains, classes, labels):
            if label == 0:
                self.assertEqual(class_value, domain.rsplit(".", 1)[-1] if "." in domain else "")
            else:
                self.assertIn(class_value, {"kraken", "banjori"})

        self.assertEqual(manifest["counts"]["benign_rows_written"], 2)
        self.assertEqual(manifest["counts"]["dga_rows_written"], 2)

    def test_fraudulent_rows_are_emitted_before_benign_rows_in_full_mode(self) -> None:
        self._write_inputs(
            ["same.com", "good.com", "safe.net"],
            [
                "domain,algorithm_code",
                "same.com,kraken",
                "bad.xyz,banjori",
            ],
        )
        manifest = build_arff_dataset(self._config())
        table = pq.read_table(self.output_dir / "udcdga_dataset.parquet")
        labels = table.column("LABEL").to_pylist()

        self.assertEqual(labels[:2], [1, 1])
        self.assertEqual(labels[2:], [0, 0])
        self.assertEqual(manifest["counts"]["benign_overlap_excluded"], 1)

    def test_reduces_fraudulent_rows_when_benign_supply_is_insufficient(self) -> None:
        self._write_inputs(
            ["overlap.com", "onlybenign.net"],
            [
                "domain,algorithm_code",
                "overlap.com,alg_a",
                "f1.com,alg_a",
                "f2.com,alg_b",
                "f3.com,alg_b",
            ],
        )
        manifest = build_arff_dataset(self._config())
        table = pq.read_table(self.output_dir / "udcdga_dataset.parquet")
        labels = table.column("LABEL").to_pylist()

        self.assertEqual(labels.count(0), 1)
        self.assertEqual(labels.count(1), 1)
        self.assertEqual(manifest["counts"]["dga_rows_reduced_for_balance"], 3)
        self.assertEqual(manifest["counts"]["final_balanced_rows_per_class"], 1)

    def test_extra_benign_rows_are_truncated_to_dga_count(self) -> None:
        self._write_inputs(
            ["good1.com", "good2.com", "good3.com"],
            ["domain,algorithm_code", "bad.xyz,alg_a"],
        )
        build_arff_dataset(self._config())
        table = pq.read_table(self.output_dir / "udcdga_dataset.parquet")
        self.assertEqual(table.column("LABEL").to_pylist().count(0), 1)
        self.assertEqual(table.column("LABEL").to_pylist().count(1), 1)

    def test_internal_duplicates_are_removed_from_each_class(self) -> None:
        self._write_inputs(
            ["good.com", "good.com", "safe.net"],
            ["domain,algorithm_code", "bad.xyz,alg_a", "bad.xyz,alg_b", "evil.org,alg_b"],
        )
        manifest = build_arff_dataset(self._config())
        self.assertEqual(manifest["counts"]["benign_duplicate_rows_dropped"], 1)
        self.assertEqual(manifest["counts"]["dga_duplicate_rows_dropped"], 1)

    def test_empty_or_missing_class_is_rejected(self) -> None:
        self._write_inputs([], ["domain,algorithm_code", "bad.xyz,alg_a"])
        with self.assertRaisesRegex(ValueError, "at least one valid domain"):
            build_arff_dataset(self._config(validate_output=False))

    def test_sample_file_generated_and_uses_new_schema(self) -> None:
        self._write_inputs(
            [f"good{i}.com" for i in range(40)],
            ["domain,algorithm_code", *[f"bad{i}.xyz,kraken" for i in range(40)]],
        )
        build_arff_dataset(self._config(sample_size=30))
        sample_rows = self._read_sample_rows(self.output_dir / SAMPLE_FILE_NAME)

        self.assertTrue(sample_rows)
        self.assertEqual(len(sample_rows[0]), len(full_dataset_columns(selected_feature_names())))

    def test_metadata_schema_description_uses_class_and_label_fields(self) -> None:
        self._write_inputs(
            ["good.com"],
            ["domain,algorithm_code", "bad.xyz,kraken"],
        )
        build_arff_dataset(self._config())

        schema_desc = json.loads((self.metadata_dir / "udcdga_ARFF_dataset_schema.json").read_text(encoding="utf-8"))
        field_names = [item["field_name"] for item in schema_desc["fields"]]
        self.assertIn("DOMAIN", field_names)
        self.assertIn("CLASS", field_names)
        self.assertIn("LABEL", field_names)
        self.assertNotIn("class", field_names)

    def test_legacy_benign_filename_is_used_as_fallback(self) -> None:
        legacy_benign_file = self.dataset_dir / "udcdga_benign_domains.csv"
        legacy_benign_file.write_text("good.com\n", encoding="utf-8")
        self.dga_file.write_text("domain,algorithm_code\nbad.xyz,kraken\n", encoding="utf-8")

        build_arff_dataset(self._config())
        table = pq.read_table(self.output_dir / "udcdga_dataset.parquet")
        self.assertEqual(table.num_rows, 2)


if __name__ == "__main__":
    unittest.main()
