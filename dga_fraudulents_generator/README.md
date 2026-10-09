# DGA Domain Dataset Generator

## Purpose

`dga_fraudulents_dataset` provides the infrastructure used to discover,
inspect, plan, execute, validate, and deduplicate domains produced by Domain
Generation Algorithm (DGA) implementations. The implementations themselves
are not part of the framework: they are supplied through an external directory
selected with `--algorithms-root`. The framework is not limited to the
implementations used to produce UDCDGA; additional implementations can use the
generic discovery path or a dedicated adapter without changing the pipeline's
global workflow.

## Requirements

- Python 3.11 or newer.
- An external algorithm repository for a dataset generation run.
- Sufficient storage for the SQLite deduplication database, checkpoints, CSV,
  JSON artifacts, and logs. Runtime and storage grow with the target size and
  with the collision rate of the selected implementations.

The framework has no third-party runtime dependencies. Individual external
algorithm implementations may have their own requirements.

## Installation

```bash
cd dga_fraudulents_generator
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the test dependency with `python -m pip install pytest`.

## Quick start

```bash
python -m dga_fraudulents_dataset \
  --algorithms-root /path/to/dga_algorithms \
  --output-dir results \
  --target-count 15000000 \
  --batch-size 5000 \
  --checkpoint-every 100000 \
  --seed-strategy sequential \
  --date-strategy daily_forward \
  --log-level INFO
```

Run `python -m dga_fraudulents_dataset --help` for the complete CLI.

## Algorithm repository structure

`--algorithms-root` must contain one directory per algorithm implementation;
the directory name becomes its `algorithm_code`:

```text
dga_algorithms/
├── algorithm_a/
│   ├── dga.py
│   └── example_domains.txt
└── algorithm_b/
    └── generator.py
```

Discovery parses Python files with the AST, preferring conventional entrypoint
names and functions such as `dga`, `generate_domains`, or `get_domains`. It
records callable signatures, seed/date parameters, example suffixes, and
execution strategy. Empty modules, structural stubs, unresolved redirects,
missing implementations, and modules without executable entrypoints are
discarded with an explicit reason. Words such as `todo` or `placeholder` in an
ordinary comment do not alone determine that reason.

The generic adapter supports discovered Python callables and CLI emitters.
Dedicated adapters are selected for implementations whose calling conventions,
parameter spaces, resources, or capacity behavior need specific handling. The
current registry includes Banjori, BazarBackdoor, Charbot, Chinad, Corebot,
Darkcracks, Dmsniff, Fobber, Fosniw, Gozi, Locky, M0yv, MoneroDownloader,
Mydoom, Newgoz, Ngioweb, Nymaim, Orchard, Qsnatch, and Zloader.

## Generation modes

`proportional_redistributed` is the default. It assigns a `planned_quota`,
tracks the runtime `effective_quota`, and attempts to reach `--target-count`.
When an implementation cannot deliver its remaining quota, redistribution
first considers eligible implementations in the same generation-mechanism
category, then other categories. Candidate receivers are ordered by capacity,
health, and observed unique yield, subject to their effective-quota limits.

`ordered_capped` processes a fixed algorithm order and applies
`--per-algorithm-cap` to each implementation. It does not redistribute or
backfill unused capacity, so its final corpus size is variable:

```bash
python -m dga_fraudulents_dataset \
  --algorithms-root /path/to/dga_algorithms \
  --output-dir results \
  --generation-mode ordered_capped \
  --per-algorithm-cap 200000
```

The inferred `date_based`, `seed_based`, and uncategorized/other categories
describe generation mechanisms, not necessarily malware families.

## Configuration

`--config` accepts a JSON object. Recognized top-level keys provide defaults
for the corresponding CLI settings, and `algorithm_overrides` contains
per-implementation discovery or adapter overrides. If the same top-level
setting appears in JSON and is explicitly supplied on the command line, the
CLI value takes precedence.

```json
{
  "target_count": 100000,
  "batch_size": 5000,
  "algorithm_overrides": {
    "algorithm_a": {
      "adapter_type": "python_function",
      "priority_weight": 2.0
    }
  }
}
```

The generation plan can also provide algorithm-level values. Use CLI help as
the authoritative list of defaults and accepted strategy names.

## Validation and normalization

Generated values are stripped, converted to lowercase, and have one trailing
dot removed. Unicode labels are converted to ASCII with Python's built-in IDNA
codec. The resulting domain must:

- contain at least two labels and no empty label;
- be no longer than 253 characters;
- use labels no longer than 63 characters;
- contain only ASCII letters, digits, and interior hyphens;
- end in a 2--63 character alphabetic suffix.

Invalid-domain reasons are counted in the persisted statistics.

## Deduplication

Global deduplication uses SQLite. `domain` is the primary key and batches use
`INSERT OR IGNORE`, so a domain generated by more than one implementation is
stored once with the algorithm code from its first successful insertion. The
database is committed at checkpoints and on close. The final CSV is streamed
from the database in insertion order with the header
`domain,algorithm_code`.

## Checkpoint and resume

Checkpoints persist the SQLite database and `udcdga_generation_state.json`,
including plans, counters, statuses, explored parameters, active algorithms,
and the database path. Resume uses the existing output directory:

```bash
python -m dga_fraudulents_dataset \
  --algorithms-root /path/to/dga_algorithms \
  --output-dir results \
  --target-count 15000000 \
  --resume
```

SQLite uniqueness prevents reinserting previously stored domains. Moving a
state file created with an absolute database path may require adjusting the
environment or path; the current state format has no explicit schema-version
field.

## Algorithm health states

The persisted initial runnable state is named `usable` (rather than `active`).
During execution the framework records deterministic status transitions:

- `usable`: eligible for normal generation;
- `partial`: usable but unable or not expected to satisfy the full allocation;
- `degraded`: still eligible, with errors, timeouts, or low yield affecting
  health;
- `saturated`: recent generation has low marginal unique yield but limited
  probing may continue;
- `exhausted`: no further useful unique output is expected;
- `discarded`: incompatible, invalid, explicitly disabled, or repeatedly
  failing and therefore not executed further.

`missing` is additionally used by `ordered_capped` when a requested
implementation is absent. Plans, logs, state, and statistics use these names
and preserve transition reasons.

## Outputs

The output directory contains:

- `udcdga_dga_domains.csv`: unique generated domains and algorithm codes;
- `udcdga_dga_domains_stats.json`: totals, distributions, health metrics,
  timings, effective parameters, and errors;
- `udcdga_dga_generation_plan.json`: discovery results and quota planning;
- `udcdga_generation_state.json`: checkpoint/resume state;
- `udcdga_dedup.sqlite3`: persistent deduplication store;
- `logs/dga_fraudulents_dataset.log`: execution log.

The plan distinguishes `planned_quota`, `effective_quota`, delivered unique
domains, remaining deficit, and redistributed quota. Partial controlled runs
also write valid JSON artifacts when the pipeline reaches its normal reporting
path.

## Reproducibility

Seed and date strategies, configured date bounds, algorithm order, external
implementation contents, Python version, and implementation-specific behavior
all affect the output. Parameter exploration is deterministic for equivalent
inputs and settings, but byte-identical results are only expected when the
external implementations, their dependencies, configuration, and execution
environment are also equivalent. Filesystem and subprocess behavior can differ
across operating systems.

CLI subprocess execution does not use `shell=True`. It captures stdout and
stderr, records return codes, applies invocation and batch timeouts, and kills
the child process after an invocation timeout.

### Security considerations

DGA implementations are executed as trusted code. Python-based implementations
may run within the main generator process, while CLI-based implementations run
as ordinary subprocesses. UDCDGA_Generator does not provide sandboxing,
privilege separation, filesystem or network isolation, syscall filtering, or
operating-system resource limits.

Users should inspect third-party DGA implementations before execution.
Untrusted implementations should be evaluated within an appropriately
isolated container or virtual machine.

## Testing

The self-contained unit suite does not require the external algorithm
repository:

```bash
pytest -m "not integration"
```

Dedicated-adapter integration tests are marked `integration`. Running `pytest`
executes them when their corresponding implementation directory is available
under `dga_algorithms/`; otherwise they are skipped with an explicit reason.

## Citation

If this software supports published research, cite the associated SoftwareX
article and identify the software version and external algorithm collection
used. Complete bibliographic details should be added here when available.

## License

Copyright 2026 Victor Carneiro. Licensed under the Apache License, Version 2.0.
See `LICENSE` for the license text.
