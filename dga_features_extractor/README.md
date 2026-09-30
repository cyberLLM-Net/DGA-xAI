# DGA Feature Dataset Builder

## Purpose

`dga_features_extractor` constructs a balanced supervised-learning dataset from
benign domains and DGA-generated domains. It normalizes and validates inputs,
removes duplicates and cross-label overlap, computes a selected set of 54
numeric features, and writes the primary dataset as Parquet. An ARFF sample and
optional statistics, schema, manifest, and integrity metadata support audit and
interoperability.

## Requirements

- Python 3.11 or newer.
- `pyarrow>=16.0.0`, installed automatically with the package.
- Sufficient storage for input CSV files, Parquet output, temporary buffers,
  and optional row-level integrity metadata.

The software version is `1.0.0`. Dataset version (`--dataset-version`) and
schema version are separate metadata concepts; both currently default to
`1.0.0` but may evolve independently.

## Installation

```bash
cd dga_features_extractor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Input datasets

The benign input accepts a CSV header containing `domain`, or a headerless file
whose first column contains domains. The default path is
`raw_data/udcdga_benigns_domains.csv`; the historical singular filename
`udcdga_benign_domains.csv` remains a compatible fallback.

The DGA input must provide a domain and an implementation identifier. The
recommended schema is:

```csv
domain,algorithm_code
example-dga.net,example_algorithm
```

Several historical spellings of the algorithm column remain accepted for
compatibility. Documentation and new datasets should use `algorithm_code` and
the term “DGA-generated domain”, rather than “fraudulent domain”.

Domains are trimmed, lowercased, stripped of trailing dots, and converted to
ASCII with Python's IDNA codec. Values containing whitespace, commas, empty
labels, labels longer than 63 characters, leading/trailing label hyphens,
invalid ASCII characters, or total length greater than 253 are rejected.

## Dataset construction workflow

The recommended command is:

```bash
python -m dga_features_extractor build-dataset \
  --benign-input raw_data/udcdga_benigns_domains.csv \
  --dga-input raw_data/udcdga_dga_domains.csv \
  --output-dir processed_data \
  --metadata-dir processed_data/metadata \
  --base-name udcdga_dataset \
  --parquet-row-group-size 10000 \
  --sample-size 1000 \
  --write-manifest \
  --write-integrity \
  --write-stats \
  --write-schema-description \
  --validate-output \
  --overwrite
```

The workflow reads and deduplicates DGA rows first, then reads benign rows,
excluding domains already assigned to the DGA label. It balances the two
classes, computes only the selected features, streams rows to Parquet, selects
the ARFF and audit samples, and finally writes metadata and validates requested
outputs.

Parquet is the primary format. `build-arff` and `validate-arff`, together with
the installed `dga-arff` command, are legacy aliases retained for compatibility.
Use `build-dataset`, `validate-dataset`, or the `dga-dataset` entry point in new
workflows.

## Feature set

The output schema contains exactly 57 columns:

```text
DOMAIN | 54 selected numeric features | CLASS | LABEL
```

Feature names are fixed by `SELECTED_FEATURE_NAMES`. Their prefixes describe
the general measurement group:

- `N_*`: character-count and linguistic measurements for domain levels;
- `L_*`: lexical length, run, ratio, alternation, and entropy measurements;
- `1G_*`, `2G_*`, and `3G_*`: unigram, bigram, and trigram distribution or
  distance measurements.

Subset extraction calculates only requested groups and returns only requested
keys. Formulas and reference distributions are unchanged in version 1.0.0.

`DOMAIN` is the normalized string. All selected features are non-null
`float64`. `CLASS` is the TLD for a benign row and the source `algorithm_code`
for a DGA row. `LABEL` is `int8`, with benign=`0` and DGA=`1`. A valid domain
without a dot is currently accepted by normalization and receives an empty
benign `CLASS`; the validator checks that this matches the same extraction
rule.

If a selected feature key is absent, its value is written as `0.0`. Values
that cannot be converted to a finite float are also replaced by `0.0`.
`feature_missing`, `feature_invalid_numeric`, and the aggregate
`feature_errors` counters record these cases. An exception raised by the
feature extractor increments `feature_computation_errors` and skips that row;
the final balance check prevents silently publishing an unbalanced result.

## Balancing and deduplication

Selection is performed in this order:

1. Normalize DGA rows, reject invalid rows, and deduplicate by domain.
2. Normalize benign rows, deduplicate them, and exclude DGA overlap.
3. Select benign domains up to the number of unique DGA domains.
4. If benign supply is smaller, reduce DGA rows proportionally by
   `algorithm_code`, using largest remainders and deterministic hash ordering.
5. Require a non-empty 1:1 class balance.

Uniqueness and overlap exclusion are always enabled. `--deduplicate` and
`--deduplicate-global` are deprecated compatibility flags and are ignored; a
warning is emitted when either is supplied. They may be removed in a future
major version.

## Reproducibility

`--random-seed` controls proportional DGA reduction, reservoir selection for
the ARFF and audit samples, and deterministic train/validation/test assignment.
Splits use the first 64 bits of `SHA256(seed:domain)`, so assignment is
independent of input order and a domain can belong to only one split.

Equivalent normalized inputs, seed, selected feature schema, and configuration
produce the same logical records and split assignments. Byte-identical Parquet
files are not promised across different Python, PyArrow, or platform versions.
Size-based part rotation uses a deterministic estimated row footprint rather
than compressed on-disk size.

Input SHA256 values, the feature-schema checksum, output hashes, software and
dataset versions, Python version, seed, execution arguments, transformations,
row counts, split settings, and feature-error counts are recorded in metadata.
Paths inside the dataset workspace are stored relative to its root; external
paths remain as configured.

## Output artifacts

Depending on selected flags, the pipeline writes:

- `udcdga_dataset.parquet`, or numbered Parquet parts after size rotation;
- train/validation/test Parquet files with `--split-datasets`;
- `udcdga_ARFF_dataset_samples.arff`, a deterministic reservoir sample;
- `metadata/udcdga_dataset_stats.json`;
- `metadata/udcdga_dataset_manifest.json`;
- `metadata/udcdga_dataset_integrity.json`;
- `metadata/udcdga_ARFF_dataset_schema.json`;
- benign and DGA label-audit CSV samples.

The ARFF sample uses the same `DOMAIN + 54 features + CLASS + LABEL` schema as
Parquet and is not the primary dataset. `--sample-size` sets its maximum row
count. Invalid numeric conversions are represented by the already sanitized
numeric values passed from the main build.

The integrity JSON is streamed during generation and contains one SHA256 entry
per output row plus final artifact hashes. This provides row-level auditability
but adds linear I/O and storage cost and can be large for multi-million-row
datasets. Disable `--write-integrity` when row-level hashes are unnecessary.

## Validation

```bash
python -m dga_features_extractor validate-dataset \
  --output-dir processed_data \
  --metadata-dir processed_data/metadata \
  --base-name udcdga_dataset
```

Validation checks Parquet readability, exact column order, consistent schemas
between parts, binary labels, 1:1 balance when required, absence of cross-label
domain overlap, benign `CLASS=TLD`, non-empty DGA `CLASS`, ARFF field count and
schema, and manifest JSON syntax.

## Testing

```bash
pytest -q
```

Tests cover feature selection, normalization, balancing, duplicate and overlap
handling, split determinism, Parquet rotation, ARFF consistency, metadata,
integrity, CLI behavior, and validator failures.

## Citation

If this software supports published research, cite the associated SoftwareX
article and identify the software, dataset, and schema versions used. Complete
bibliographic details should be added here when available.

## License

Copyright 2026 Victor Carneiro. Licensed under the Apache License, Version 2.0.
See `LICENSE` for the license text.
