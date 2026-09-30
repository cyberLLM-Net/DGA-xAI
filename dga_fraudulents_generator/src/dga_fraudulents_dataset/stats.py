from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from statistics import median
from typing import Any

from .models import AlgorithmPlan, normalize_status
from .utils import iso


def build_stats_payload(
    started_at: datetime,
    finished_at: datetime,
    plans: dict[str, AlgorithmPlan],
    target_count: int,
    unique_count: int,
    duplicate_count: int,
    params: dict[str, Any],
    generation_mode: str = "proportional_redistributed",
    per_algorithm_cap: int | None = None,
    ordered_algorithms: list[str] | None = None,
    ignored_discovered_algorithms: list[str] | None = None,
) -> dict[str, Any]:
    detected = len(plans)
    valid = sum(1 for p in plans.values() if normalize_status(p.status) in {"usable", "degraded", "partial", "saturated", "exhausted"})
    discarded = sum(1 for p in plans.values() if normalize_status(p.status) == "discarded")

    per_algorithm: dict[str, dict[str, Any]] = {}
    errors_by_algorithm: dict[str, list[str]] = {}
    by_cat: dict[str, int] = defaultdict(int)

    if generation_mode == "ordered_capped" and ordered_algorithms:
        algo_items = [(code, plans[code]) for code in ordered_algorithms if code in plans]
    else:
        algo_items = sorted(plans.items())

    for code, p in algo_items:
        med = median(p.batch_yield_history) if p.batch_yield_history else 0.0
        redistributed_quota = p.redistributed_quota if generation_mode != "ordered_capped" else 0
        redistributed_in = p.redistributed_in if generation_mode != "ordered_capped" else 0
        redistributed_out = p.redistributed_out if generation_mode != "ordered_capped" else 0
        deficit_absorbed_total = p.deficit_absorbed_total if generation_mode != "ordered_capped" else 0
        per_algorithm[code] = {
            "planned_quota": p.planned_quota,
            "effective_quota": p.effective_quota,
            "maximum_effective_quota": p.maximum_effective_quota,
            "redistributed_quota": redistributed_quota,
            "redistributed_in": redistributed_in,
            "redistributed_out": redistributed_out,
            "deficit_absorbed_total": deficit_absorbed_total,
            "delivered_unique": p.delivered_unique(),
            "quota_deficit": p.quota_deficit(),
            "remaining_quota": p.remaining_quota(),
            "oversupply": p.oversupply(),
            "deficit_remaining": p.deficit_remaining(),
            "attempts_total": p.attempted_count,
            "generated_total": p.generated_count,
            "valid_total": p.valid_count,
            "inserted_unique_total": p.unique_valid_count,
            "duplicates_total": p.duplicate_count,
            "invalid_total": p.invalid_count,
            "invalid_reason_counts": p.invalid_reason_counts,
            "empty_batches": p.empty_batches,
            "error_count": p.error_count,
            "timeout_count": p.timeout_count,
            "consecutive_failures": p.consecutive_failures,
            "consecutive_low_yield_rounds": p.consecutive_low_yield_rounds,
            "effective_unique_yield_ratio": p.effective_unique_yield_ratio,
            "last_batch_unique_yield_ratio": p.last_batch_unique_yield_ratio,
            "median_batch_unique_yield_ratio": med,
            "rolling_yield_summary": {
                "window": len(p.recent_batch_yield_ratios),
                "values": p.recent_batch_yield_ratios[-20:],
                "rolling_unique_gain": p.rolling_unique_gain,
                "rolling_duplicate_rate": p.rolling_duplicate_rate,
                "marginal_unique_gain": p.marginal_unique_gain,
            },
            "last_effective_params": p.last_effective_params,
            "supported_parameter_axes": p.supported_parameter_axes,
            "status": normalize_status(p.status),
            "discard_reason": p.discard_reason,
            "category": p.category,
            "strategy": p.strategy,
            "capacity_score": p.capacity_score,
            "health_score": p.health_score,
            "saturation_score": p.saturation_score,
            "expected_diversity_score": p.expected_diversity_score,
            "observed_unique_capacity_estimate": p.observed_unique_capacity_estimate,
            "quota_confidence": p.quota_confidence,
            "redistribution_eligibility": p.redistribution_eligibility,
            "adaptive_batch_mode": p.adaptive_batch_mode,
            "exhaustion_reason": p.exhaustion_reason,
            "prefix_entropy": float((p.profiling or {}).get("prefix_entropy", 0.0) or 0.0),
            "suffix_constancy": float((p.profiling or {}).get("suffix_constancy", 0.0) or 0.0),
            "estimated_capacity": int(
                (p.profiling or {}).get(
                    "theoretical_capacity_estimate",
                    (p.profiling or {}).get("estimated_max_unique_capacity", 0),
                )
                or 0
            ),
            "saturation_point": (p.last_effective_params or {}).get("saturation_point"),
            "date_window": p.date_window,
            "date_wrap_policy": p.date_wrap_policy,
            "status_history": [h.__dict__ for h in p.status_history],
            "profiling": p.profiling,
            "order_index": p.order_index,
            "per_algorithm_cap": p.per_algorithm_cap if p.per_algorithm_cap is not None else per_algorithm_cap,
            "stopped_reason": p.stopped_reason,
        }
        if generation_mode == "ordered_capped":
            per_algorithm[code]["redistribution_note"] = "disabled_in_ordered_capped_mode"
        if p.observations:
            errors_by_algorithm[code] = p.observations[-20:]
        by_cat[p.category or "uncategorized"] += p.unique_valid_count

    duration = (finished_at - started_at).total_seconds()

    payload = {
        "started_at": iso(started_at),
        "finished_at": iso(finished_at),
        "duration_seconds": duration,
        "algorithms_detected": detected,
        "algorithms_valid": valid,
        "algorithms_discarded": discarded,
        "target_domains": target_count if generation_mode != "ordered_capped" else None,
        "generated_domains": sum(p.generated_count for p in plans.values()),
        "unique_domains": unique_count,
        "duplicates_detected": duplicate_count,
        "duplicates_discarded": duplicate_count,
        "distribution_by_algorithm": per_algorithm,
        "distribution_by_category": dict(by_cat),
        "errors_by_algorithm": errors_by_algorithm,
        "performance": {
            "domains_per_second": (unique_count / duration) if duration > 0 else 0.0,
            "per_algorithm_valid_per_attempt": {
                code: (p.valid_count / p.attempted_count if p.attempted_count > 0 else 0.0)
                for code, p in plans.items()
            },
            "per_algorithm_unique_per_generated": {
                code: (p.unique_valid_count / p.generated_count if p.generated_count > 0 else 0.0)
                for code, p in plans.items()
            },
        },
        "effective_params": params,
    }
    if generation_mode == "ordered_capped":
        payload["generation_mode"] = "ordered_capped"
        payload["per_algorithm_cap"] = per_algorithm_cap
        payload["ordered_algorithms"] = list(ordered_algorithms or [])
        payload["ignored_discovered_algorithms"] = list(ignored_discovered_algorithms or [])
        payload["global_target_count"] = None
        payload["effective_target_count"] = unique_count
        payload["final_unique_domains"] = unique_count
        payload["redistribution_enabled"] = False
    return payload
