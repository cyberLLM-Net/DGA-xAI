# Domain Aggregator

## Purpose

Domain Aggregator builds a reproducible benign-domain corpus from multiple
`.txt` and `.csv` sources. Each source contributes in proportion to its raw row
count, and largest-remainder apportionment makes the assigned quotas sum to the
requested target (15,000,000 rows by default). Generation is streamed, while a
JSON report records configuration, per-file contributions, validation results,
quota redistribution, timings, and warnings.

## Project names

- Repository module: `dga_benigns_generator`
- Python package: `domain_aggregator`
- Project invocation: `python -m domain_aggregator build ...`
- Installed command: `domain-aggregator build ...`

## Installation

Python 3.11 or newer is required.

```bash
cd dga_benigns_generator
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the test dependency with `python -m pip install -e '.[dev]'`.
The project uses `pyproject.toml` and the setuptools build backend. If an
editable installation fails in an older environment, upgrade `pip`,
`setuptools`, and `wheel`, or use `--no-build-isolation` when build dependencies
are already installed locally.

## Quick start

```bash
python -m domain_aggregator build \
  --input-dir benigns_domains \
  --target-size 15000000 \
  --seed 42 \
  --count-workers 8
```

Default artifacts are `results/udcdga_benigns_domains.csv`,
`results/udcdga_benigns_domains_stats.json`, and
`logs/domain_aggregation.log`. Use `python -m domain_aggregator --help` and
`python -m domain_aggregator build --help` for all options. Relative values
passed to `--output` and `--report` are resolved inside `--results-dir`.

## Input format

The input directory is searched for `.txt` and `.csv` files. Plain-text files
may contain one domain per line. Delimited files may contain a header and
additional columns; delimiters are detected from comma, tab, semicolon, and
pipe characters. By default, the first column is used. Use `--domain-column`
to select a named header column, `--delimiter` to force a delimiter, or
`--no-header-auto-detect` to disable automatic header recognition.

Before validation, each candidate is normalized by removing UTF-8 BOM
characters, trimming surrounding whitespace, converting to lowercase, and
removing trailing dots. The package does **not** convert Unicode domain labels
to IDNA/Punycode. Unicode labels therefore do not pass the current ASCII
validator unless the input has already been converted to its ASCII form.

## Output format

The output CSV has one column:

```csv
domain
example.com
example.org
```

The JSON report records the target and generated totals, input configuration,
raw-row counts, validation and deduplication statistics, initial and effective
quotas, redistribution, timings, cache use, status, and warnings. Quotas are
based on raw input rows; validation is performed while domains are generated.

If a source is exhausted before filling its quota, its deficit is redistributed
deterministically among remaining sources. Replacement is enabled by default
to refill a shortfall when possible. `--no-replacement` disables it, while
`--strict-no-refill` permits a final result smaller than the target. Optional
`--deduplicate-per-file` and `--deduplicate-final` checks trade additional
memory for duplicate removal.

## Reproducibility

`--seed` defaults to `42`. For unchanged inputs and equivalent options, it
controls randomized traversal of plain-text sources and refill selection.
Quota allocation and deficit redistribution are independently deterministic.
Input counting can run concurrently with `--count-workers`; the deprecated
`--threads` spelling remains an alias. `--use-count-cache` reuses counts only
when the stored file size, modification time, and counting settings match.

## Validation modes

All modes require at least one dot and a maximum total length of 253
characters. Labels before the TLD must start and end with an ASCII letter or
digit and may not exceed 63 characters. Their TLD policies differ:

- `lenient`: TLD length 2--63; ASCII lowercase letters, digits, and hyphens are
  accepted, including a hyphen at either end of the TLD.
- `balanced` (default): TLD length 2--63; only ASCII lowercase letters are
  accepted.
- `strict`: TLD length 2--24; only ASCII lowercase letters are accepted.

These modes use local syntactic checks; they do not determine whether a TLD or
domain is registered or reachable.

## Testing

Run `pytest`. The suite covers input discovery and parsing, normalization and
validation, quota allocation, deterministic sampling, replacement, streaming
output, report generation, metadata, and CLI behavior.

## Citation

If this software supports published research, cite the associated SoftwareX
article and identify the software version used. Complete bibliographic details
should be added here when the article citation is available.

## License

Copyright 2026 Victor Carneiro. Licensed under the Apache License, Version 2.0.
See `LICENSE` for the license text.
