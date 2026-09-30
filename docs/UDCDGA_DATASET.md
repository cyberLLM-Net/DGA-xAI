# UDCDGA dataset

## Release metadata

| Field | Value |
| --- | --- |
| Dataset | UDCDGA |
| Generating software | UDCDGA_Generator |
| Software release | 1.0.0 |
| Dataset release | 1.0.0 |
| Repository | <https://github.com/cyberLLM-Net/DGA-xAI> |
| Primary format | Apache Parquet |
| Software license | Apache License 2.0 |
| Dataset DOI | TODO: add the archival DOI before publication |

The software license does not override the licenses or terms of source lists, third-party DGA implementations, or independently deposited dataset files. Those terms must be verified for the publication deposit.

## Overview

UDCDGA is a balanced binary-classification corpus of 8,450,556 domain names: 4,225,278 benign samples and 4,225,278 algorithmically generated DGA samples. Each record contains the domain, 54 lexical features, a textual class, and a numeric label, for 57 fields in total. `LABEL=0` represents benign domains and `LABEL=1` represents DGA domains.

The Parquet parts constitute the primary dataset. The 1,000-row ARFF file is provided only as a small interoperability example.

## Construction and provenance

The benign stage aggregates supplied domain lists after parsing, normalization, validation, quota allocation, sampling, and deduplication. Sources may include lists obtained from [Domains Project](https://domainsproject.org/); neither that service nor its data is bundled by this repository. The preserved build statistics report that the final builder read 4,226,261 benign rows, removed 970 duplicates and excluded 13 domains that overlapped the DGA class, leaving 4,225,278 samples.

The malicious-domain stage executes DGA implementations supplied in a local algorithm repository. The inspected generation statistics report 43 valid implementations and 4,353,800 generated candidates, reduced by persistent global deduplication to 4,225,278 unique DGA domains. The implementations cover seed-, date-, counter-, and wordlist-driven mechanisms; the exact implementations and per-algorithm contributions are recorded by the generation plan and statistics rather than inferred as malware-family labels.

The feature stage normalizes both intermediate corpora, prevents cross-label overlap, selects equal class sizes with a recorded seed, computes the feature vector, and writes deterministic schema and statistics metadata. Generation plans, state databases, checkpoints, logs, and intermediate CSVs are operational artifacts; they are not parts of the final feature dataset unless explicitly included in an archival deposit.

## Feature representation

The 54 features describe lexical properties of a domain, including length and label structure, character composition and transitions, digit and vowel/consonant patterns, repetition, entropy and related distribution measures, n-gram statistics, and dictionary-oriented indicators. The authoritative names, types, nullability, label semantics, and order are recorded in `udcdga_dataset_schema.json` and implemented by the feature extractor.

## Released Parquet parts

| File | Rows | Fields | Size (bytes) |
| --- | ---: | ---: | ---: |
| `udcdga_dataset_part_0001.parquet` | 3,530,539 | 57 | 432,196,588 |
| `udcdga_dataset_part_0002.parquet` | 3,479,046 | 57 | 528,207,132 |
| `udcdga_dataset_part_0003.parquet` | 1,440,971 | 57 | 177,355,095 |
| **Total** | **8,450,556** | **57** | **1,137,758,815** |

The inspected metadata directory contains the dataset statistics, schema, and label-audit files. Version 1.0.0 can additionally emit a machine-readable manifest and integrity report with `--write-manifest --write-integrity`; these files were not present in the inspected historical output and must not be claimed as release artifacts unless regenerated or added to the deposit.

## Reproduction

Install the three packages as described in the root README, provide the source domain lists and licensed DGA implementations, run both generators, and pass their CSV outputs to `dga-dataset build-dataset`. Record the command line, random seeds, source snapshots, execution plan, statistics, schema, manifest, integrity report, and software commit or release tag.

The pipeline supports seeded selection, persistent deduplication, health monitoring, quota redistribution, resumable DGA generation, schema validation, and output integrity metadata. These controls improve repeatability, but they do not guarantee bit-identical output across changes in external inputs, DGA implementations, dependency versions, platform, concurrency, or Parquet encoding. The preserved dataset statistics identify the original build pipeline as version 0.5.0; the 1.0.0 designation refers to the stabilized publication release and does not imply that the archived rows were regenerated.

## Known limitations

- Inclusion in a benign source list is not proof that a domain remained benign at every point in time.
- Generated DGA domains model algorithmic output and are not evidence of registration, resolution, or observed malicious activity.
- Dataset composition depends on the supplied source snapshots and available DGA implementations.
- Lexical features omit network, DNS, temporal, and host context.
- Source-data and third-party-code licenses require independent verification before redistribution.

## Citation

For reproducibility, citations should also identify the repository release or commit used to generate the data.
