from __future__ import annotations

import csv
from itertools import chain
from pathlib import Path
from typing import Iterator

from .domain import normalize_domain
from .models import ParseConfig


def _detect_delimiter(lines: list[str]) -> str | None:
    candidates = [",", "\t", ";", "|"]
    best_delim: str | None = None
    best_score = 0
    for delim in candidates:
        score = 0
        for line in lines:
            if line.strip():
                score += line.count(delim)
        if score > best_score:
            best_score = score
            best_delim = delim
    return best_delim if best_score > 0 else None


def _looks_like_header(row: list[str]) -> bool:
    lowered = [c.strip().lower() for c in row]
    header_tokens = {"domain", "hostname", "fqdn", "host"}
    return any(token in header_tokens for token in lowered)


def _pick_domain_index(header_row: list[str], domain_column: str | None) -> int:
    if domain_column is None:
        return 0
    lowered = [c.strip().lower() for c in header_row]
    target = domain_column.strip().lower()
    if target not in lowered:
        raise ValueError(f"Domain column '{domain_column}' was not found in header: {header_row}")
    return lowered.index(target)


def iter_domain_candidates(path: Path, config: ParseConfig) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample_lines = []
        for _ in range(30):
            line = handle.readline()
            if not line:
                break
            sample_lines.append(line)

        delimiter = config.delimiter if config.delimiter is not None else _detect_delimiter(sample_lines)

        if delimiter is None:
            for raw_line in chain(sample_lines, handle):
                normalized = normalize_domain(raw_line)
                if normalized:
                    yield normalized
            return

        reader = csv.reader(chain(sample_lines, handle), delimiter=delimiter)
        first_row: list[str] | None = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                first_row = row
                break

        if first_row is None:
            return

        has_header = False
        if config.domain_column is not None:
            has_header = True
        elif config.header_auto_detect:
            has_header = _looks_like_header(first_row)

        if has_header:
            domain_index = _pick_domain_index(first_row, config.domain_column)
        else:
            domain_index = 0
            if domain_index < len(first_row):
                normalized = normalize_domain(first_row[domain_index])
                if normalized:
                    yield normalized

        for row in reader:
            if domain_index >= len(row):
                continue
            normalized = normalize_domain(row[domain_index])
            if normalized:
                yield normalized
