from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def normalize_status(value: str | None) -> str:
    if not value:
        return "usable"
    m = {
        "usable": "usable",
        "degraded": "degraded",
        "saturated": "saturated",
        "exhausted": "exhausted",
        "partial": "partial",
        "discarded": "discarded",
        "missing": "missing",
        "parcial": "partial",
        "descartado": "discarded",
    }
    return m.get(value.lower(), value.lower())


@dataclass
class StatusTransition:
    ts: str
    from_status: str
    to_status: str
    reason: str


@dataclass
class AlgorithmInspection:
    algorithm_code: str
    path: str
    python_files: list[str] = field(default_factory=list)
    txt_files: list[str] = field(default_factory=list)
    category: str | None = None
    strategy: str | None = None
    entrypoint: str | None = None
    callable_name: str | None = None
    required_params: list[str] = field(default_factory=list)
    default_params: dict[str, Any] = field(default_factory=dict)
    requires_seed: bool = False
    requires_date: bool = False
    inferred_tlds: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status: str = "usable"
    discard_reason: str | None = None

    # Manual override/adapter hints
    module_name: str | None = None
    adapter_type: str | None = None
    force_scalar_mode: bool = False
    force_batch_mode: bool = False
    seed_parameter_name: str | None = None
    date_parameter_name: str | None = None
    counter_parameter_name: str | None = None
    parameter_strategy: dict[str, Any] = field(default_factory=dict)
    discard: bool = False
    priority_weight: float = 1.0
    max_cli_invocations_per_batch: int | None = None
    algorithm_batch_timeout_seconds: int | None = None

    # Profiling output
    supported_parameter_axes: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    date_window: dict[str, Any] = field(default_factory=dict)
    date_wrap_policy: str | None = None


@dataclass
class AlgorithmPlan:
    algorithm_code: str
    path: str
    category: str | None
    strategy: str
    entrypoint: str | None
    callable_name: str | None
    required_params: list[str]
    default_params: dict[str, Any]
    requires_seed: bool
    requires_date: bool
    target_count: int

    # Quotas
    planned_quota: int = 0
    effective_quota: int = 0
    redistributed_quota: int = 0
    maximum_effective_quota: int = 0
    redistributed_in: int = 0
    redistributed_out: int = 0
    deficit_absorbed_total: int = 0

    # Generation counters
    generated_count: int = 0
    unique_valid_count: int = 0
    attempted_count: int = 0
    valid_count: int = 0
    duplicate_count: int = 0
    invalid_count: int = 0
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)

    # Reliability counters
    error_count: int = 0
    timeout_count: int = 0
    empty_batches: int = 0
    consecutive_failures: int = 0
    consecutive_low_yield_rounds: int = 0
    consecutive_zero_unique_batches: int = 0
    consecutive_subthreshold_batches: int = 0

    # Yield metrics
    last_batch_unique_yield_ratio: float = 0.0
    effective_unique_yield_ratio: float = 0.0
    batch_yield_history: list[float] = field(default_factory=list)
    recent_batch_yield_ratios: list[float] = field(default_factory=list)
    recent_unique_gains: list[int] = field(default_factory=list)
    rolling_duplicate_rate: float = 0.0
    rolling_unique_gain: int = 0
    marginal_unique_gain: float = 0.0

    # Health/traceability
    observations: list[str] = field(default_factory=list)
    status: str = "usable"
    discard_reason: str | None = None
    status_history: list[StatusTransition] = field(default_factory=list)
    last_effective_params: dict[str, Any] = field(default_factory=dict)
    supported_parameter_axes: list[str] = field(default_factory=list)
    capacity_score: float = 1.0
    health_score: float = 1.0
    profiling: dict[str, Any] = field(default_factory=dict)
    saturation_score: float = 0.0
    expected_diversity_score: float = 0.5
    observed_unique_capacity_estimate: int = 0
    quota_confidence: float = 0.5
    redistribution_eligibility: bool = True
    exhaustion_reason: str | None = None
    adaptive_batch_mode: str = "normal"
    near_quota_retries: int = 0
    recommended_max_effective_quota_multiplier: float = 1.0
    date_window: dict[str, Any] = field(default_factory=dict)
    date_wrap_policy: str = "clamp"
    order_index: int | None = None
    per_algorithm_cap: int | None = None
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        if self.maximum_effective_quota <= 0:
            self.maximum_effective_quota = max(self.effective_quota, self.planned_quota, self.target_count)

    def quota_deficit(self) -> int:
        return max(self.effective_quota - self.unique_valid_count, 0)

    def delivered_unique(self) -> int:
        return self.unique_valid_count

    def remaining_quota(self) -> int:
        return max(self.effective_quota - self.unique_valid_count, 0)

    def oversupply(self) -> int:
        return max(self.unique_valid_count - self.effective_quota, 0)

    def deficit_remaining(self) -> int:
        return self.remaining_quota()


@dataclass
class GenerationRuntimeState:
    started_at: str
    target_count: int
    db_path: str
    plans: dict[str, AlgorithmPlan]
    active_algorithms: list[str]
    global_generated_unique: int = 0
    global_attempted: int = 0
    total_duplicates: int = 0
    finished: bool = False
    generation_mode: str = "proportional_redistributed"
    per_algorithm_cap: int | None = None
    ordered_algorithms: list[str] = field(default_factory=list)
    current_algorithm_index: int = 0
    ignored_discovered_algorithms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        plans = {
            k: _json_safe(
                {
                    **asdict(v),
                    "status": normalize_status(v.status),
                    "status_history": [asdict(h) for h in v.status_history],
                }
            )
            for k, v in self.plans.items()
        }
        return {
            "started_at": self.started_at,
            "target_count": self.target_count,
            "db_path": self.db_path,
            "plans": plans,
            "active_algorithms": self.active_algorithms,
            "global_generated_unique": self.global_generated_unique,
            "global_attempted": self.global_attempted,
            "total_duplicates": self.total_duplicates,
            "finished": self.finished,
            "generation_mode": self.generation_mode,
            "per_algorithm_cap": self.per_algorithm_cap,
            "ordered_algorithms": self.ordered_algorithms,
            "current_algorithm_index": self.current_algorithm_index,
            "ignored_discovered_algorithms": self.ignored_discovered_algorithms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationRuntimeState":
        plans: dict[str, AlgorithmPlan] = {}
        defaults = {
            "planned_quota": 0,
            "effective_quota": 0,
            "redistributed_quota": 0,
            "maximum_effective_quota": 0,
            "redistributed_in": 0,
            "redistributed_out": 0,
            "deficit_absorbed_total": 0,
            "consecutive_zero_unique_batches": 0,
            "consecutive_subthreshold_batches": 0,
            "recent_batch_yield_ratios": [],
            "recent_unique_gains": [],
            "rolling_duplicate_rate": 0.0,
            "rolling_unique_gain": 0,
            "marginal_unique_gain": 0.0,
            "saturation_score": 0.0,
            "expected_diversity_score": 0.5,
            "observed_unique_capacity_estimate": 0,
            "quota_confidence": 0.5,
            "redistribution_eligibility": True,
            "exhaustion_reason": None,
            "adaptive_batch_mode": "normal",
            "near_quota_retries": 0,
            "recommended_max_effective_quota_multiplier": 1.0,
            "date_window": {},
            "date_wrap_policy": "clamp",
            "discard_reason": None,
            "invalid_reason_counts": {},
            "order_index": None,
            "per_algorithm_cap": None,
            "stopped_reason": None,
        }
        for k, v in payload["plans"].items():
            v = dict(v)
            v["status"] = normalize_status(v.get("status"))
            hist = []
            for h in v.get("status_history", []):
                if isinstance(h, dict):
                    hist.append(StatusTransition(**h))
            v["status_history"] = hist
            if "planned_quota" not in v:
                v["planned_quota"] = v.get("target_count", 0)
            if "effective_quota" not in v:
                v["effective_quota"] = v.get("target_count", 0)
            for dk, dv in defaults.items():
                if dk not in v:
                    v[dk] = dv
            if not v.get("maximum_effective_quota"):
                v["maximum_effective_quota"] = v.get("effective_quota", v.get("target_count", 0))
            plans[k] = AlgorithmPlan(**v)
        return cls(
            started_at=payload["started_at"],
            target_count=payload["target_count"],
            db_path=payload["db_path"],
            plans=plans,
            active_algorithms=payload.get("active_algorithms", list(plans.keys())),
            global_generated_unique=payload.get("global_generated_unique", 0),
            global_attempted=payload.get("global_attempted", 0),
            total_duplicates=payload.get("total_duplicates", 0),
            finished=payload.get("finished", False),
            generation_mode=payload.get("generation_mode", "proportional_redistributed"),
            per_algorithm_cap=payload.get("per_algorithm_cap"),
            ordered_algorithms=payload.get("ordered_algorithms", []),
            current_algorithm_index=payload.get("current_algorithm_index", 0),
            ignored_discovered_algorithms=payload.get("ignored_discovered_algorithms", []),
        )


@dataclass
class RunSummary:
    started_at: datetime
    finished_at: datetime
    target_count: int
    unique_count: int
    duplicate_count: int
    algorithms_detected: int
    algorithms_valid: int
    algorithms_discarded: int
    per_algorithm: dict[str, dict[str, Any]]
    per_category: dict[str, int]
    errors_by_algorithm: dict[str, list[str]]
    params: dict[str, Any]
