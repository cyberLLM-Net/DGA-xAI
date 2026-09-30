from __future__ import annotations

from pathlib import Path

DEFAULT_RAW_DATA_DIR = Path("raw_data")
DEFAULT_PROCESSED_DATA_DIR = Path("processed_data")
DEFAULT_METADATA_SUBDIR = "metadata"
DEFAULT_LOG_DIR = Path("logs")

DEFAULT_BENIGN_INPUT = DEFAULT_RAW_DATA_DIR / "udcdga_benigns_domains.csv"
DEFAULT_BENIGN_INPUT_ALT = DEFAULT_RAW_DATA_DIR / "udcdga_benign_domains.csv"
DEFAULT_DGA_INPUT = DEFAULT_RAW_DATA_DIR / "udcdga_dga_domains.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_DATA_DIR
DEFAULT_METADATA_DIR = DEFAULT_PROCESSED_DATA_DIR / DEFAULT_METADATA_SUBDIR
