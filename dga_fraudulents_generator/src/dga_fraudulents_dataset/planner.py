from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import AlgorithmInspection, AlgorithmPlan, normalize_status


def _split_even(total: int, n: int) -> list[int]:
    if n <= 0:
        return []
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _normalize_category(cat: str | None) -> str:
    return cat if cat else "uncategorized"


def build_generation_plan(
    inspections: list[AlgorithmInspection],
    target_count: int,
    category_weights: dict[str, float] | None = None,
) -> dict[str, AlgorithmPlan]:
    usable = [i for i in inspections if normalize_status(i.status) in {"usable", "partial", "degraded"} and not i.discard]
    plans: dict[str, AlgorithmPlan] = {}

    if not usable:
        for i in inspections:
            plans[i.algorithm_code] = AlgorithmPlan(
                algorithm_code=i.algorithm_code,
                path=i.path,
                category=i.category,
                strategy=i.strategy or "unknown",
                entrypoint=i.entrypoint,
                callable_name=i.callable_name,
                required_params=i.required_params,
                default_params=i.default_params,
                requires_seed=i.requires_seed,
                requires_date=i.requires_date,
                target_count=0,
                planned_quota=0,
                effective_quota=0,
                observations=i.notes,
                status="discarded",
                discard_reason=i.discard_reason,
            )
        return plans

    categories: dict[str, list[AlgorithmInspection]] = defaultdict(list)
    for ins in usable:
        categories[_normalize_category(ins.category)].append(ins)

    algo_targets: dict[str, int] = {}

    if len(categories) == 1 and "uncategorized" in categories:
        splits = _split_even(target_count, len(usable))
        for ins, q in zip(usable, splits):
            algo_targets[ins.algorithm_code] = q
    else:
        cat_names = sorted(categories.keys())
        if category_weights:
            total_w = sum(max(category_weights.get(c, 0.0), 0.0) for c in cat_names)
            if total_w <= 0:
                cat_shares = {c: 1.0 / len(cat_names) for c in cat_names}
            else:
                cat_shares = {c: max(category_weights.get(c, 0.0), 0.0) / total_w for c in cat_names}
            cat_targets = {c: int(target_count * cat_shares[c]) for c in cat_names}
            assigned = sum(cat_targets.values())
            for c in cat_names[: target_count - assigned]:
                cat_targets[c] += 1
        else:
            cat_splits = _split_even(target_count, len(cat_names))
            cat_targets = {c: q for c, q in zip(cat_names, cat_splits)}

        for cat, cat_target in cat_targets.items():
            algos = categories[cat]
            splits = _split_even(cat_target, len(algos))
            for ins, q in zip(algos, splits):
                algo_targets[ins.algorithm_code] = q

    for ins in inspections:
        status = normalize_status(ins.status)
        target = algo_targets.get(ins.algorithm_code, 0)
        if ins.discard:
            status = "discarded"
            target = 0
        if target == 0 and status in {"usable", "degraded"}:
            status = "partial"
        profile = ins.profile or {}
        generation_mode = str(profile.get("generation_mode", "")).strip()
        strategy_value = generation_mode if generation_mode else (ins.strategy or "unknown")
        cap = float(profile.get("initial_capacity_score", 1.0))
        health = float(profile.get("initial_health_score", 1.0))
        diversity = float(profile.get("expected_diversity_score", 0.5))
        quota_mult = float(profile.get("recommended_max_effective_quota_multiplier", 1.0))
        supported_axes = list(ins.supported_parameter_axes or profile.get("supported_parameter_axes", []))
        max_effective = max(target, int(target * max(1.0, quota_mult)))
        estimated_capacity = int(profile.get("estimated_max_unique_capacity", 0) or 0)
        recommended_effective_cap = int(profile.get("recommended_effective_quota_cap", 0) or 0)
        if estimated_capacity > 0 and target > estimated_capacity:
            ins.notes.append(f"quota_capped_by_estimated_capacity:{estimated_capacity}")
            target = estimated_capacity
            max_effective = min(max_effective, estimated_capacity)
        if recommended_effective_cap > 0 and target > recommended_effective_cap:
            ins.notes.append(f"quota_capped_by_recommended_effective_cap:{recommended_effective_cap}")
            target = recommended_effective_cap
            max_effective = min(max_effective, recommended_effective_cap)

        plans[ins.algorithm_code] = AlgorithmPlan(
            algorithm_code=ins.algorithm_code,
            path=ins.path,
            category=ins.category,
            strategy=strategy_value,
            entrypoint=ins.entrypoint,
            callable_name=ins.callable_name,
            required_params=ins.required_params,
            default_params=ins.default_params,
            requires_seed=ins.requires_seed,
            requires_date=ins.requires_date,
            target_count=target,
            planned_quota=target,
            effective_quota=target,
            maximum_effective_quota=max_effective,
            observations=ins.notes,
            status=status,
            discard_reason=ins.discard_reason,
            supported_parameter_axes=supported_axes,
            capacity_score=cap,
            health_score=health,
            expected_diversity_score=diversity,
            recommended_max_effective_quota_multiplier=quota_mult,
            profiling=profile,
            observed_unique_capacity_estimate=max(1, int(target * max(0.2, diversity))),
            quota_confidence=max(0.1, min(1.0, diversity + 0.1)),
            date_window=ins.date_window or profile.get("recommended_date_window", {}),
            date_wrap_policy=ins.date_wrap_policy or "clamp",
        )
        if estimated_capacity > 0:
            p = plans[ins.algorithm_code]
            p.maximum_effective_quota = min(max(p.maximum_effective_quota, p.effective_quota), estimated_capacity)
            p.effective_quota = min(p.effective_quota, estimated_capacity)
            p.planned_quota = min(p.planned_quota, estimated_capacity)
            p.target_count = min(p.target_count, estimated_capacity)
            if p.remaining_quota() <= 0:
                p.redistribution_eligibility = False
        if recommended_effective_cap > 0:
            p = plans[ins.algorithm_code]
            p.maximum_effective_quota = min(max(p.maximum_effective_quota, p.effective_quota), recommended_effective_cap)
            p.effective_quota = min(p.effective_quota, recommended_effective_cap)
            p.planned_quota = min(p.planned_quota, recommended_effective_cap)
            p.target_count = min(p.target_count, recommended_effective_cap)
            if p.remaining_quota() <= 0:
                p.redistribution_eligibility = False
        if bool(profile.get("low_structural_diversity", False)):
            p = plans[ins.algorithm_code]
            p.redistribution_eligibility = False
            p.observations.append("low_structural_diversity_profile")

    return plans


def build_ordered_capped_plan(
    inspections: list[AlgorithmInspection],
    ordered_algorithms: list[str],
    per_algorithm_cap: int,
) -> dict[str, AlgorithmPlan]:
    plans: dict[str, AlgorithmPlan] = {}
    by_code = {ins.algorithm_code.lower(): ins for ins in inspections}
    for idx, algo_code in enumerate(ordered_algorithms):
        ins = by_code.get(algo_code.lower())
        if ins is None:
            plans[algo_code] = AlgorithmPlan(
                algorithm_code=algo_code,
                path="",
                category=None,
                strategy="ordered_capped_missing",
                entrypoint=None,
                callable_name=None,
                required_params=[],
                default_params={},
                requires_seed=False,
                requires_date=False,
                target_count=0,
                planned_quota=0,
                effective_quota=0,
                maximum_effective_quota=0,
                observations=["ordered_capped:missing_algorithm"],
                status="missing",
                discard_reason="missing_algorithm",
                redistribution_eligibility=False,
                order_index=idx,
                per_algorithm_cap=per_algorithm_cap,
                stopped_reason="missing_algorithm",
            )
            continue

        profile = ins.profile or {}
        generation_mode = str(profile.get("generation_mode", "")).strip()
        strategy_value = generation_mode if generation_mode else (ins.strategy or "unknown")
        status = normalize_status(ins.status)
        cap = float(profile.get("initial_capacity_score", 1.0))
        health = float(profile.get("initial_health_score", 1.0))
        diversity = float(profile.get("expected_diversity_score", 0.5))
        supported_axes = list(ins.supported_parameter_axes or profile.get("supported_parameter_axes", []))
        target = per_algorithm_cap
        if ins.discard or status == "discarded":
            status = "discarded"
            target = 0
        elif status not in {"usable", "degraded", "partial", "saturated", "exhausted"}:
            status = "discarded"
            target = 0
            if not ins.discard_reason:
                ins.discard_reason = "invalid_algorithm"

        plans[algo_code] = AlgorithmPlan(
            algorithm_code=algo_code,
            path=ins.path,
            category=ins.category,
            strategy=strategy_value,
            entrypoint=ins.entrypoint,
            callable_name=ins.callable_name,
            required_params=ins.required_params,
            default_params=ins.default_params,
            requires_seed=ins.requires_seed,
            requires_date=ins.requires_date,
            target_count=target,
            planned_quota=target,
            effective_quota=target,
            maximum_effective_quota=target,
            observations=ins.notes,
            status=status,
            discard_reason=ins.discard_reason,
            supported_parameter_axes=supported_axes,
            capacity_score=cap,
            health_score=health,
            expected_diversity_score=diversity,
            recommended_max_effective_quota_multiplier=1.0,
            profiling=profile,
            observed_unique_capacity_estimate=max(1, int(max(target, 1) * max(0.2, diversity))),
            quota_confidence=max(0.1, min(1.0, diversity + 0.1)),
            date_window=ins.date_window or profile.get("recommended_date_window", {}),
            date_wrap_policy=ins.date_wrap_policy or "clamp",
            redistribution_eligibility=False,
            order_index=idx,
            per_algorithm_cap=per_algorithm_cap,
        )
    return plans


def plan_to_json(
    plans: dict[str, AlgorithmPlan],
    detected: list[AlgorithmInspection],
    *,
    generation_mode: str = "proportional_redistributed",
    ordered_algorithms: list[str] | None = None,
    per_algorithm_cap: int | None = None,
    ignored_discovered_algorithms: list[str] | None = None,
) -> dict[str, Any]:
    detected_map = {d.algorithm_code: d for d in detected}
    items = []
    if generation_mode == "ordered_capped" and ordered_algorithms:
        algo_codes = list(ordered_algorithms)
    else:
        algo_codes = sorted(set(detected_map.keys()).union(plans.keys()))

    for algo_code in algo_codes:
        p = plans.get(algo_code)
        d = detected_map.get(algo_code)
        path = p.path if p else (d.path if d else "")
        category = p.category if p else (d.category if d else None)
        strategy = p.strategy if p else (d.strategy if d else "unknown")
        entrypoint = p.entrypoint if p else (d.entrypoint if d else None)
        callable_name = p.callable_name if p else (d.callable_name if d else None)
        required_params = p.required_params if p else (d.required_params if d else [])
        default_params = p.default_params if p else (d.default_params if d else {})
        requires_seed = p.requires_seed if p else (d.requires_seed if d else False)
        requires_date = p.requires_date if p else (d.requires_date if d else False)
        supported_parameter_axes = p.supported_parameter_axes if p else (d.supported_parameter_axes if d else [])
        status = p.status if p else normalize_status(d.status if d else "missing")
        discard_reason = p.discard_reason if p else (d.discard_reason if d else None)
        observations = p.observations if p else (d.notes if d else [])
        profiling = p.profiling if p else (d.profile if d else {})
        max_cli_invocations_per_batch = d.max_cli_invocations_per_batch if d else None
        algorithm_batch_timeout_seconds = d.algorithm_batch_timeout_seconds if d else None
        txt_files = d.txt_files if d else []
        python_files = d.python_files if d else []
        items.append(
            {
                "algorithm_code": algo_code,
                "path": path,
                "category": category,
                "strategy": strategy,
                "entrypoint": entrypoint,
                "callable_name": callable_name,
                "required_params": required_params,
                "default_params": default_params,
                "requires_seed": requires_seed,
                "requires_date": requires_date,
                "supported_parameter_axes": supported_parameter_axes,
                "planned_quota": p.planned_quota if p else 0,
                "effective_quota": p.effective_quota if p else 0,
                "redistributed_quota": p.redistributed_quota if p else 0,
                "redistributed_in": p.redistributed_in if p else 0,
                "redistributed_out": p.redistributed_out if p else 0,
                "deficit_absorbed_total": p.deficit_absorbed_total if p else 0,
                "maximum_effective_quota": p.maximum_effective_quota if p else 0,
                "target_count": p.target_count if p else 0,
                "generated_count": p.generated_count if p else 0,
                "unique_valid_count": p.unique_valid_count if p else 0,
                "delivered_unique": p.delivered_unique() if p else 0,
                "remaining_quota": p.remaining_quota() if p else 0,
                "oversupply": p.oversupply() if p else 0,
                "attempts_total": p.attempted_count if p else 0,
                "duplicates_total": p.duplicate_count if p else 0,
                "invalid_total": p.invalid_count if p else 0,
                "empty_batches": p.empty_batches if p else 0,
                "error_count": p.error_count if p else 0,
                "timeout_count": p.timeout_count if p else 0,
                "effective_unique_yield_ratio": p.effective_unique_yield_ratio if p else 0.0,
                "last_batch_unique_yield_ratio": p.last_batch_unique_yield_ratio if p else 0.0,
                "rolling_yield_summary": {
                    "window": len(p.recent_batch_yield_ratios) if p else 0,
                    "values": p.recent_batch_yield_ratios[-20:] if p else [],
                    "rolling_unique_gain": p.rolling_unique_gain if p else 0,
                    "rolling_duplicate_rate": p.rolling_duplicate_rate if p else 0.0,
                    "marginal_unique_gain": p.marginal_unique_gain if p else 0.0,
                },
                "last_effective_params": p.last_effective_params if p else {},
                "capacity_score": p.capacity_score if p else 0.0,
                "redistribution_eligibility": p.redistribution_eligibility if p else False,
                "saturation_score": p.saturation_score if p else 0.0,
                "expected_diversity_score": p.expected_diversity_score if p else 0.0,
                "quota_confidence": p.quota_confidence if p else 0.0,
                "status": status,
                "discard_reason": discard_reason,
                "status_history": [h.__dict__ for h in p.status_history] if p else [],
                "exhaustion_reason": p.exhaustion_reason if p else None,
                "adaptive_batch_mode": p.adaptive_batch_mode if p else "normal",
                "observations": observations,
                "profiling": profiling,
                "max_cli_invocations_per_batch": max_cli_invocations_per_batch,
                "algorithm_batch_timeout_seconds": algorithm_batch_timeout_seconds,
                "date_window": p.date_window if p else (d.date_window if d else {}),
                "date_wrap_policy": p.date_wrap_policy if p else (d.date_wrap_policy if d else None),
                "txt_files": txt_files,
                "python_files": python_files,
                "order_index": p.order_index if p else None,
                "per_algorithm_cap": p.per_algorithm_cap if p else per_algorithm_cap,
                "stopped_reason": p.stopped_reason if p else None,
            }
        )

    payload: dict[str, Any] = {
        "algorithms": items,
        "summary": {
            "detected": len(detected),
            "planned": len(plans),
            "target_total": sum(v.target_count for v in plans.values()),
            "effective_total": sum(v.effective_quota for v in plans.values()),
            "generation_mode": generation_mode,
            "per_algorithm_cap": per_algorithm_cap,
        },
    }
    if generation_mode == "ordered_capped":
        payload["ordered_algorithms"] = list(ordered_algorithms or [])
        payload["ignored_discovered_algorithms"] = list(ignored_discovered_algorithms or [])
    return payload
