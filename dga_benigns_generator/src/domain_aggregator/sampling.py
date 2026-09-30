from __future__ import annotations

import hashlib
import random
from array import array
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterator

from .domain import is_valid_domain, normalize_domain
from .models import ParseConfig
from .parser import iter_domain_candidates

_TEXT_RANDOM_SUFFIXES = {".txt"}
_DELIMITER_CANDIDATES = [",", "\t", ";", "|"]
_RANDOM_ACCESS_BUFFER_SIZE = 1024 * 1024


@dataclass
class SampleEmissionStats:
    sampled_rows: int
    emitted_rows: int
    used_replacement: bool
    skipped_by_final_dedup: int


def stable_file_seed(path: Path, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{path.as_posix()}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _detect_delimiter(lines: list[str]) -> str | None:
    best_delim: str | None = None
    best_score = 0
    for delimiter in _DELIMITER_CANDIDATES:
        score = sum(line.count(delimiter) for line in lines if line.strip())
        if score > best_score:
            best_delim = delimiter
            best_score = score
    return best_delim if best_score > 0 else None


def _looks_like_tabular_input(path: Path, parse_config: ParseConfig) -> bool:
    if parse_config.domain_column is not None:
        return True
    if parse_config.delimiter is not None:
        return True

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            sample_lines = []
            for _ in range(30):
                line = handle.readline()
                if not line:
                    break
                sample_lines.append(line)
    except OSError:
        return False

    return _detect_delimiter(sample_lines) is not None


def build_random_access_strategy_for_file(path: Path) -> array:
    offsets = array("Q")
    with path.open("rb", buffering=_RANDOM_ACCESS_BUFFER_SIZE) as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            offsets.append(offset)
    return offsets


def _pick_coprime_step(total_lines: int, rng: random.Random) -> int:
    if total_lines <= 1:
        return 1

    step = rng.randrange(1, total_lines)
    while gcd(step, total_lines) != 1:
        step = rng.randrange(1, total_lines)
    return step


def sample_random_candidates_from_file(path: Path, offsets: array, seed: int) -> Iterator[str]:
    total_lines = len(offsets)
    if total_lines <= 0:
        return

    rng = random.Random(seed)
    start = rng.randrange(total_lines)
    step = _pick_coprime_step(total_lines, rng)

    with path.open("rb", buffering=_RANDOM_ACCESS_BUFFER_SIZE) as handle:
        for index in range(total_lines):
            randomized_index = (start + (index * step)) % total_lines
            handle.seek(int(offsets[randomized_index]))
            raw_line = handle.readline()
            if not raw_line:
                continue
            yield raw_line.decode("utf-8", errors="replace")


def iter_randomized_domain_candidates(path: Path, parse_config: ParseConfig, seed: int) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix not in _TEXT_RANDOM_SUFFIXES or _looks_like_tabular_input(path, parse_config):
        yield from iter_domain_candidates(path, parse_config)
        return

    offsets = build_random_access_strategy_for_file(path)
    for raw_line in sample_random_candidates_from_file(path, offsets, seed):
        normalized = normalize_domain(raw_line)
        if normalized:
            yield normalized


def iter_usable_domains(path: Path, parse_config: ParseConfig, deduplicate_per_file: bool) -> Iterator[str]:
    seen: set[str] | None = set() if deduplicate_per_file else None
    for candidate in iter_domain_candidates(path, parse_config):
        if not is_valid_domain(candidate, strictness=parse_config.validation_strictness):
            continue
        if seen is not None:
            if candidate in seen:
                continue
            seen.add(candidate)
        yield candidate


def _stream_sample_without_replacement(
    path: Path,
    parse_config: ParseConfig,
    deduplicate_per_file: bool,
    quota: int,
    available_rows: int,
    rng: random.Random,
) -> Iterator[str]:
    need = quota
    remaining = available_rows
    if need <= 0:
        return

    for domain in iter_usable_domains(path, parse_config, deduplicate_per_file):
        if need <= 0:
            break
        # Exactly-q streaming selection based on known remaining pool size.
        if rng.random() < (need / remaining):
            yield domain
            need -= 1
        remaining -= 1


def _load_pool(path: Path, parse_config: ParseConfig, deduplicate_per_file: bool) -> list[str]:
    return list(iter_usable_domains(path, parse_config, deduplicate_per_file))


def iter_sampled_domains(
    path: Path,
    parse_config: ParseConfig,
    deduplicate_per_file: bool,
    quota: int,
    available_rows: int,
    allow_replacement: bool,
    seed: int,
) -> tuple[Iterator[str], bool]:
    rng = random.Random(seed)

    if quota <= available_rows:
        return (
            _stream_sample_without_replacement(
                path,
                parse_config,
                deduplicate_per_file,
                quota,
                available_rows,
                rng,
            ),
            False,
        )

    pool = _load_pool(path, parse_config, deduplicate_per_file)
    if not pool:
        return iter(()), False

    if not allow_replacement:
        return iter(pool), False

    def _iter_with_replacement() -> Iterator[str]:
        for domain in pool:
            yield domain
        extra = quota - len(pool)
        for _ in range(extra):
            yield rng.choice(pool)

    return _iter_with_replacement(), True


def refill_candidate_pools(
    file_paths: list[Path],
    parse_config: ParseConfig,
    deduplicate_per_file: bool,
) -> dict[Path, list[str]]:
    pools: dict[Path, list[str]] = {}
    for path in file_paths:
        pool = _load_pool(path, parse_config, deduplicate_per_file)
        if pool:
            pools[path] = pool
    return pools
