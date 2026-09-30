from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParseConfig:
    delimiter: str | None = None
    domain_column: str | None = None
    header_auto_detect: bool = True
    validation_strictness: str = "balanced"


@dataclass(frozen=True)
class BuildConfig:
    input_dir: Path
    output_csv: Path
    report_path: Path
    target_size: int = 15_000_000
    default_target_size_used: bool = False
    seed: int = 42
    selection_strategy: str = "random_per_file"
    threads: int = 8
    count_workers: int | None = None
    count_log_every: int = 500_000
    use_count_cache: bool = False
    count_cache_path: Path | None = None
    count_validation_mode: str = "raw_rows_only"
    parse: ParseConfig = field(default_factory=ParseConfig)
    deduplicate_per_file: bool = False
    deduplicate_final: bool = False
    allow_replacement: bool = True
    strict_no_refill: bool = False
    log_path: Path = Path("logs/domain_aggregation.log")


@dataclass
class FileStats:
    path: Path
    discovered: bool = True
    size_bytes: int = 0
    raw_rows_seen: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    usable_rows_for_quota: int = 0
    duplicates_removed_per_file: int = 0
    duplicates_removed_final: int = 0
    quota: int = 0
    initial_quota: int = 0
    effective_quota: int = 0
    raw_quota: float = 0.0
    quota_floor: int = 0
    quota_fractional_remainder: float = 0.0
    quota_rounding_adjustment: int = 0
    accepted_domains: int = 0
    rejected_domains: int = 0
    acceptance_rate: float = 0.0
    exhausted_before_quota: bool = False
    redistributed_deficit_in: int = 0
    redistributed_deficit_out: int = 0
    generation_attempted_rows: int = 0
    generation_elapsed_seconds: float = 0.0
    emitted_rows: int = 0
    sampled_rows_before_final_dedup: int = 0
    used_replacement: bool = False
    count_elapsed_seconds: float = 0.0
    count_avg_rate_lps: float = 0.0
    reused_from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["assigned_quota"] = self.effective_quota
        data["elapsed_seconds"] = self.generation_elapsed_seconds
        return data


@dataclass
class RunReport:
    config: BuildConfig
    input_files: list[Path]
    files: list[FileStats]
    total_valid_rows: int
    total_usable_rows_for_quota: int
    total_invalid_rows: int
    total_duplicates_removed_per_file: int
    total_duplicates_removed_final: int
    final_output_rows: int
    status: str
    allocation_basis: str = "raw_rows"
    total_raw_rows: int = 0
    total_rejected_domains: int = 0
    generation_phase_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0
    redistributions_performed: int = 0
    total_redistributed_deficit: int = 0
    count_phase_elapsed_seconds: float = 0.0
    count_phase_workers: int = 0
    count_phase_files_total: int = 0
    count_phase_files_from_cache: int = 0
    count_phase_cache_enabled: bool = False
    count_phase_cache_path: str | None = None
    count_phase_validation_mode: str = "raw_rows_only"
    count_phase_log_every: int = 500_000
    rounding_strategy: str = "largest_remainder"
    quota_floor_total: int = 0
    rounding_adjustment_total: int = 0
    total_assigned_quota: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        per_file = [f.to_dict() for f in self.files]
        return {
            "target_size": self.config.target_size,
            "generated_total": self.final_output_rows,
            "selection_strategy": self.config.selection_strategy,
            "allocation_basis": self.allocation_basis,
            "rounding_method": self.rounding_strategy,
            "total_raw_rows": self.total_raw_rows,
            "default_target_size_used": self.config.default_target_size_used,
            "config": {
                "input_dir": str(self.config.input_dir),
                "output_csv": str(self.config.output_csv),
                "report_path": str(self.config.report_path),
                "target_size": self.config.target_size,
                "default_target_size_used": self.config.default_target_size_used,
                "seed": self.config.seed,
                "selection_strategy": self.config.selection_strategy,
                "threads": self.config.threads,
                "count_workers": self.config.count_workers,
                "count_log_every": self.config.count_log_every,
                "use_count_cache": self.config.use_count_cache,
                "count_cache_path": str(self.config.count_cache_path) if self.config.count_cache_path is not None else None,
                "count_validation_mode": self.config.count_validation_mode,
                "delimiter": self.config.parse.delimiter,
                "domain_column": self.config.parse.domain_column,
                "header_auto_detect": self.config.parse.header_auto_detect,
                "validation_strictness": self.config.parse.validation_strictness,
                "deduplicate_per_file": self.config.deduplicate_per_file,
                "deduplicate_final": self.config.deduplicate_final,
                "allow_replacement": self.config.allow_replacement,
                "strict_no_refill": self.config.strict_no_refill,
                "log_path": str(self.config.log_path),
            },
            "input_files": [str(p) for p in self.input_files],
            "files": per_file,
            "totals": {
                "raw_rows": self.total_raw_rows,
                "valid_rows": self.total_valid_rows,
                "usable_rows_for_quota": self.total_usable_rows_for_quota,
                "invalid_rows": self.total_invalid_rows,
                "rejected_domains": self.total_rejected_domains,
                "duplicates_removed_per_file": self.total_duplicates_removed_per_file,
                "duplicates_removed_final": self.total_duplicates_removed_final,
                "final_output_rows": self.final_output_rows,
                "target_size": self.config.target_size,
                "assigned_quota_total": self.total_assigned_quota,
                "quota_floor_total": self.quota_floor_total,
                "rounding_adjustment_total": self.rounding_adjustment_total,
                "redistributions_performed": self.redistributions_performed,
                "total_redistributed_deficit": self.total_redistributed_deficit,
            },
            "counting": {
                "count_workers": self.count_phase_workers,
                "count_log_every": self.count_phase_log_every,
                "count_validation_mode": self.count_phase_validation_mode,
                "cache_enabled": self.count_phase_cache_enabled,
                "cache_path": self.count_phase_cache_path,
                "files_total": self.count_phase_files_total,
                "files_reused_from_cache": self.count_phase_files_from_cache,
                "elapsed_seconds": self.count_phase_elapsed_seconds,
            },
            "timings": {
                "count_raw_rows_seconds": self.count_phase_elapsed_seconds,
                "generation_seconds": self.generation_phase_elapsed_seconds,
                "total_seconds": self.total_elapsed_seconds,
            },
            "quota_allocation": {
                "target_size": self.config.target_size,
                "generated_total": self.final_output_rows,
                "allocation_basis": self.allocation_basis,
                "rounding_strategy": self.rounding_strategy,
                "rounding_method": self.rounding_strategy,
                "rounding_adjustment_total": self.rounding_adjustment_total,
                "total_raw_rows": self.total_raw_rows,
                "redistributions_performed": self.redistributions_performed,
                "total_redistributed_deficit": self.total_redistributed_deficit,
                "per_file": [
                    {
                        "path": str(file_stats.path),
                        "raw_rows": file_stats.raw_rows_seen,
                        "usable_rows_for_quota": file_stats.usable_rows_for_quota,
                        "raw_quota": file_stats.raw_quota,
                        "quota_floor": file_stats.quota_floor,
                        "quota_rounding_adjustment": file_stats.quota_rounding_adjustment,
                        "initial_quota": file_stats.initial_quota,
                        "assigned_quota": file_stats.quota,
                        "effective_quota": file_stats.effective_quota,
                        "accepted_domains": file_stats.accepted_domains,
                        "rejected_domains": file_stats.rejected_domains,
                        "acceptance_rate": file_stats.acceptance_rate,
                        "exhausted_before_quota": file_stats.exhausted_before_quota,
                        "redistributed_deficit_in": file_stats.redistributed_deficit_in,
                        "redistributed_deficit_out": file_stats.redistributed_deficit_out,
                        "emitted_rows": file_stats.emitted_rows,
                    }
                    for file_stats in self.files
                ],
            },
            "status": self.status,
            "warnings": self.warnings,
        }
