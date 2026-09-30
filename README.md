# UDCDGA_Generator

UDCDGA_Generator is a reproducible and extensible Python framework for constructing the balanced UDCDGA domain-name dataset. The repository separates corpus construction into three independent stages; it does not provide a master orchestration command.

## Core stages

1. [`dga_benigns_generator`](dga_benigns_generator/README.md) validates, normalizes, samples, and aggregates benign-domain sources.
2. [`dga_fraudulents_generator`](dga_fraudulents_generator/README.md) executes locally supplied DGA implementations and persistently deduplicates the generated domains.
3. [`dga_features_extractor`](dga_features_extractor/README.md) removes cross-label overlap, balances both classes, extracts 54 lexical features, and writes the final 57-field dataset.

```text
benign sources ──> dga_benigns_generator ──┐
                                           ├─> dga_features_extractor ──> UDCDGA
DGA implementations ─> dga_fraudulents_generator ─┘
```

Model training and evaluation are optional downstream activities, not a fourth dataset-generation stage. Any separately distributed classification component is not required to build UDCDGA.

## Requirements and installation

Use Python 3.11 and a shared virtual environment. The source corpora and third-party DGA implementations are not bundled. Benign sources may include data obtained from [Domains Project](https://domainsproject.org/); users remain responsible for the terms and provenance of every input.

```bash
git clone https://github.com/cyberLLM-Net/DGA-xAI.git
cd DGA-xAI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./dga_benigns_generator \
  -e ./dga_fraudulents_generator \
  -e ./dga_features_extractor
```

## Quick start

Run these commands from the repository root. Full-scale generation is computationally and storage intensive; use smaller targets for a trial run.

```bash
python -m domain_aggregator build \
  --input-dir dga_benigns_generator/benigns_domains \
  --results-dir dga_benigns_generator/results \
  --target-size 15000000 --seed 42

python -m dga_fraudulents_dataset \
  --algorithms-root dga_fraudulents_generator/dga_algorithms \
  --output-dir dga_fraudulents_generator/results \
  --target-count 15000000

dga-dataset build-dataset \
  --benign-input dga_benigns_generator/results/udcdga_benigns_domains.csv \
  --dga-input dga_fraudulents_generator/results/udcdga_dga_domains.csv \
  --output-dir dga_features_extractor/processed_data \
  --metadata-dir dga_features_extractor/processed_data/metadata \
  --write-manifest --write-integrity --validate-output --overwrite
```

The first two stages produce intermediate CSV corpora and execution statistics. The third produces the final balanced dataset as partitioned Parquet, with `DOMAIN`, 54 features, `CLASS`, and `LABEL` (57 fields total). `LABEL=0` denotes benign domains and `LABEL=1` DGA domains. The ARFF output is a small compatibility sample, not the primary dataset.

## Documentation

- [UDCDGA dataset card](docs/UDCDGA_DATASET.md)
- [Benign generator](dga_benigns_generator/README.md)
- [DGA generator](dga_fraudulents_generator/README.md)
- [Feature extractor](dga_features_extractor/README.md)


## License and support

The repository software is licensed under the [Apache License 2.0](LICENSE). External data and DGA implementations may have different terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Report reproducible defects through the repository issue tracker.
