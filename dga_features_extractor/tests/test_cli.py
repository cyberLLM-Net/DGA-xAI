import tempfile
import unittest
from pathlib import Path

from dga_features_extractor.cli import build_parser, main
from dga_features_extractor.schema import SAMPLE_FILE_NAME


class TestCli(unittest.TestCase):
    def test_default_paths_use_raw_and_processed_layout(self):
        parser = build_parser()
        args = parser.parse_args(["build-dataset"])
        self.assertEqual(str(args.benign_input), "raw_data/udcdga_benigns_domains.csv")
        self.assertEqual(str(args.dga_input), "raw_data/udcdga_dga_domains.csv")
        self.assertEqual(str(args.output_dir), "processed_data")
        self.assertEqual(str(args.metadata_dir), "processed_data/metadata")

    def test_parser_rejects_invalid_threshold(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["build-dataset", "--split-threshold-bytes", "0"])

    def test_build_command_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dga_cli_") as tmp:
            root = Path(tmp)
            dataset_dir = root / "raw_data"
            output_dir = root / "processed_data"
            metadata_dir = output_dir / "metadata"
            log_dir = root / "logs"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "udcdga_benigns_domains.csv").write_text("good.com\n", encoding="utf-8")
            (dataset_dir / "udcdga_dga_domains.csv").write_text(
                "domain,algorithm_code\nbad.xyz,kraken\n",
                encoding="utf-8",
            )

            code = main(
                [
                    "build-dataset",
                    "--benign-input",
                    str(dataset_dir / "udcdga_benigns_domains.csv"),
                    "--dga-input",
                    str(dataset_dir / "udcdga_dga_domains.csv"),
                    "--output-dir",
                    str(output_dir),
                    "--metadata-dir",
                    str(metadata_dir),
                    "--log-dir",
                    str(log_dir),
                    "--schema-reference",
                    str(Path("domains_output.arff")),
                    "--overwrite",
                    "--write-manifest",
                    "--validate-output",
                    "--sample-size",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "udcdga_dataset.parquet").exists())
            self.assertTrue((output_dir / SAMPLE_FILE_NAME).exists())
            self.assertTrue((metadata_dir / "udcdga_dataset_manifest.json").exists())
            self.assertTrue((metadata_dir / "udcdga_dataset_stats.json").exists())
            self.assertTrue((metadata_dir / "udcdga_ARFF_dataset_schema.json").exists())

    def test_validate_command_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dga_cli_validate_") as tmp:
            root = Path(tmp)
            dataset_dir = root / "raw_data"
            output_dir = root / "processed_data"
            metadata_dir = output_dir / "metadata"
            log_dir = root / "logs"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            (dataset_dir / "udcdga_benigns_domains.csv").write_text("good.com\n", encoding="utf-8")
            (dataset_dir / "udcdga_dga_domains.csv").write_text(
                "domain,algorithm_code\nbad.xyz,kraken\n",
                encoding="utf-8",
            )

            build_code = main(
                [
                    "build-dataset",
                    "--benign-input",
                    str(dataset_dir / "udcdga_benigns_domains.csv"),
                    "--dga-input",
                    str(dataset_dir / "udcdga_dga_domains.csv"),
                    "--output-dir",
                    str(output_dir),
                    "--metadata-dir",
                    str(metadata_dir),
                    "--log-dir",
                    str(log_dir),
                    "--overwrite",
                    "--sample-size",
                    "2",
                ]
            )
            self.assertEqual(build_code, 0)

            validate_code = main([
                "validate-dataset",
                "--output-dir",
                str(output_dir),
                "--metadata-dir",
                str(metadata_dir),
                "--base-name",
                "udcdga_dataset",
                "--split-threshold-bytes",
                "1000000",
            ])
            self.assertEqual(validate_code, 0)


if __name__ == "__main__":
    unittest.main()
