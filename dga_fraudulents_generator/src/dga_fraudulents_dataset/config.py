from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


DEFAULT_TARGET_COUNT = 15_000_000
DEFAULT_GENERATION_MODE = "proportional_redistributed"
ORDERED_CAPPED_GENERATION_MODE = "ordered_capped"
DEFAULT_PER_ALGORITHM_CAP = 200_000
DEFAULT_CSV_NAME = "udcdga_dga_domains.csv"
DEFAULT_STATS_NAME = "udcdga_dga_domains_stats.json"
DEFAULT_PLAN_NAME = "udcdga_dga_generation_plan.json"
DEFAULT_STATE_NAME = "udcdga_generation_state.json"
DEFAULT_SQLITE_NAME = "udcdga_dedup.sqlite3"

DEFAULT_ORDERED_ALGORITHMS = [
    "dnschanger",
    "darkcracks",
    "ngioweb",
    "ramnit",
    "tinba",
    "nymaim2",
    "reconyc",
    "sharkbot",
    "vawtrak",
    "ranbyus",
    "simda",
    "expiro",
    "orchard",
    "charbot",
    "unnamed_downloader",
    "m0yv",
    "dmsniff",
    "tempedreve",
    "shiotob",
    "zloader",
    "nymaim",
    "qsnatch",
    "corebot",
    "bazarbackdoor",
    "banjori",
    "dircrypt",
    "fobber",
    "fosniw",
    "gozi",
    "bumblebee",
    "monerodownloader",
    "mydoom",
    "locky",
    "pitou",
    "proslikefan",
    "qakbot",
    "qadars",
    "ramdo",
    "newgoz",
    "necurs",
    "chinad",
    "tufik",
    "verblecon",
    "mock_algo",
    "unnamed_javascript_dga",
]


@dataclass
class AppConfig:
    algorithms_root: Path
    output_dir: Path
    target_count: int = DEFAULT_TARGET_COUNT
    generation_mode: str = DEFAULT_GENERATION_MODE
    per_algorithm_cap: int = DEFAULT_PER_ALGORITHM_CAP
    ordered_algorithms: list[str] = field(default_factory=list)
    threads: int = 1
    batch_size: int = 5_000
    checkpoint_every: int = 100_000
    plan_file: Path | None = None
    config_file: Path | None = None
    resume: bool = False
    log_level: str = "INFO"
    dedup_backend: str = "sqlite"

    seed_strategy: str = "sequential"
    date_strategy: str = "daily_forward"

    dry_run: bool = False
    validate_only: bool = False

    min_unique_yield_ratio: float = 0.01
    discard_after_consecutive_empty: int = 2
    discard_after_consecutive_low_yield: int = 5
    low_yield_grace_rounds: int = 2
    max_algorithm_errors: int = 5
    algorithm_timeout_seconds: int = 10
    algorithm_batch_timeout_seconds: int = 30
    max_cli_invocations_per_batch: int = 40
    heartbeat_seconds: int = 15
    saturation_window: int = 8
    saturation_min_yield: float = 0.03
    exhausted_after_zero_unique_batches: int = 4
    near_quota_exhaustion_margin: int = 200
    near_quota_max_retries: int = 4
    max_effective_quota_multiplier: float = 4.0
    redistribution_capacity_threshold: float = 0.2

    date_start: str = "2018-01-01"
    date_end: str = "2030-12-31"
    date_max_years_forward: int = 8
    date_max_years_backward: int = 8
    date_wrap_policy: str = "clamp"

    @property
    def csv_path(self) -> Path:
        return self.output_dir / DEFAULT_CSV_NAME

    @property
    def stats_path(self) -> Path:
        return self.output_dir / DEFAULT_STATS_NAME

    @property
    def default_plan_path(self) -> Path:
        if self.plan_file:
            return self.plan_file
        return self.output_dir / DEFAULT_PLAN_NAME

    @property
    def state_path(self) -> Path:
        return self.output_dir / DEFAULT_STATE_NAME

    @property
    def sqlite_path(self) -> Path:
        return self.output_dir / DEFAULT_SQLITE_NAME


def ensure_dirs(cfg: AppConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "logs").mkdir(parents=True, exist_ok=True)
