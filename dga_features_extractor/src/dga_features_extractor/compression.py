from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class CompressedFileStats:
    source_path: Path
    compressed_path: Path
    source_size_bytes: int
    compressed_size_bytes: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_compress_and_verify(source_path: Path, keep_uncompressed: bool) -> CompressedFileStats:
    compressed_path = source_path.with_suffix(source_path.suffix + ".gz")
    source_size_bytes = source_path.stat().st_size

    try:
        with open(source_path, "rb") as src, gzip.open(compressed_path, "wb", compresslevel=6) as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)

        if not compressed_path.exists() or compressed_path.stat().st_size <= 0:
            raise RuntimeError(f"Compressed file is missing or empty: {compressed_path}")

        source_sha = _sha256_file(source_path)
        decompressed_sha = _sha256_gzip_decompressed(compressed_path)
        if source_sha != decompressed_sha:
            raise RuntimeError(f"Compression verification failed for {source_path} -> {compressed_path}")

        if not keep_uncompressed:
            source_path.unlink(missing_ok=False)
    except Exception:
        compressed_path.unlink(missing_ok=True)
        raise

    return CompressedFileStats(
        source_path=source_path,
        compressed_path=compressed_path,
        source_size_bytes=source_size_bytes,
        compressed_size_bytes=compressed_path.stat().st_size,
        sha256=source_sha,
    )


def compress_outputs(
    source_paths: Iterable[Path],
    compression_format: str,
    keep_uncompressed: bool,
) -> List[CompressedFileStats]:
    if compression_format != "gzip":
        raise ValueError(f"Unsupported compression format: {compression_format}")

    stats: List[CompressedFileStats] = []
    for path in source_paths:
        item = gzip_compress_and_verify(path, keep_uncompressed=keep_uncompressed)
        stats.append(item)
    return stats
