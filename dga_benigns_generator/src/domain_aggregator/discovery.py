from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".csv"}


def discover_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No .txt or .csv files found in input directory: {input_dir}")
    return files
