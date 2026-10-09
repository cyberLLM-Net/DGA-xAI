from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from .adapter_base import AlgorithmAdapter
from .adapter_registry import get_adapter
from .classifier import apply_categories
from .config import (
    AppConfig,
    DEFAULT_GENERATION_MODE,
    DEFAULT_ORDERED_ALGORITHMS,
    ORDERED_CAPPED_GENERATION_MODE,
)
from .dedup import SQLiteDedupStore
from .discovery import discover_and_inspect
from .models import AlgorithmPlan, GenerationRuntimeState, StatusTransition, normalize_status
from .planner import build_generation_plan, build_ordered_capped_plan, plan_to_json
from .resume import load_state, save_state
from .stats import build_stats_payload
from .utils import parse_iso, read_json, utc_now, write_json
from .validation import validate_domain
from .writer import export_csv

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"usable", "degraded", "partial", "saturated"}


class GenerationError(RuntimeError):
    pass


def _is_ordered_capped_mode(cfg: AppConfig) -> bool:
    return str(cfg.generation_mode).strip().lower() == ORDERED_CAPPED_GENERATION_MODE


def _normalized_ordered_algorithms(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        code = str(item).strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def _resolve_ordered_algorithms(cfg: AppConfig) -> list[str]:
    if cfg.ordered_algorithms:
        return _normalized_ordered_algorithms(cfg.ordered_algorithms)
    return _normalized_ordered_algorithms(list(DEFAULT_ORDERED_ALGORITHMS))


def _inc_reason(plan: AlgorithmPlan, reason: str | None) -> None:
    if not reason:
        return
    plan.invalid_reason_counts[reason] = int(plan.invalid_reason_counts.get(reason, 0)) + 1


def _prevalidate_reason(raw: str) -> str | None:
    if raw is None:
        return "none"
    d = raw.strip().lower().rstrip(".")
    if not d:
        return "empty"
    if ".." in d:
        return "double_dot"
    if any(x == "" for x in d.split(".")):
        return "empty_label"
    return None


def _status_change(plan: AlgorithmPlan, to_status: str, reason: str) -> None:
    to_status = normalize_status(to_status)
    from_status = normalize_status(plan.status)
    if from_status == to_status:
        return
    plan.status = to_status
    plan.status_history.append(
        StatusTransition(
            ts=utc_now().isoformat(),
            from_status=from_status,
            to_status=to_status,
            reason=reason,
        )
    )
    plan.observations.append(f"status:{from_status}->{to_status}:{reason}")
    logger.info(
        "algorithm=%s transitioned %s->%s reason=%s",
        plan.algorithm_code,
        from_status,
        to_status,
        reason,
    )


def _state_from_plans(cfg: AppConfig, plans: dict[str, AlgorithmPlan]) -> GenerationRuntimeState:
    ordered_algorithms = _resolve_ordered_algorithms(cfg) if _is_ordered_capped_mode(cfg) else []
    if _is_ordered_capped_mode(cfg):
        active = [c for c in ordered_algorithms if c in plans and normalize_status(plans[c].status) in ACTIVE_STATUSES]
    else:
        active = _active_codes(plans)
    return GenerationRuntimeState(
        started_at=utc_now().isoformat(),
        target_count=cfg.target_count,
        db_path=str(cfg.sqlite_path),
        plans=plans,
        active_algorithms=active,
        global_generated_unique=0,
        global_attempted=0,
        total_duplicates=0,
        finished=False,
        generation_mode=ORDERED_CAPPED_GENERATION_MODE if _is_ordered_capped_mode(cfg) else DEFAULT_GENERATION_MODE,
        per_algorithm_cap=cfg.per_algorithm_cap if _is_ordered_capped_mode(cfg) else None,
        ordered_algorithms=ordered_algorithms,
        current_algorithm_index=0,
        ignored_discovered_algorithms=[],
    )


def _active_codes(plans: dict[str, AlgorithmPlan]) -> list[str]:
    out: list[str] = []
    for code, p in plans.items():
        st = normalize_status(p.status)
        if st not in ACTIVE_STATUSES:
            continue
        if p.remaining_quota() <= 0:
            continue
        out.append(code)
    return out


def _redistribute_from_algorithm(plans: dict[str, AlgorithmPlan], donor_code: str, reason: str) -> int:
    donor = plans[donor_code]
    deficit = donor.quota_deficit()
    if deficit <= 0:
        return 0

    donor_cat = donor.category or "uncategorized"
    same_cat = [
        p for c, p in plans.items()
        if c != donor_code
        and (p.category or "uncategorized") == donor_cat
        and normalize_status(p.status) in ACTIVE_STATUSES
        and p.redistribution_eligibility
    ]
    pool = same_cat
    cross_category = False
    if not pool:
        pool = [
            p
            for c, p in plans.items()
            if c != donor_code
            and normalize_status(p.status) in ACTIVE_STATUSES
            and p.redistribution_eligibility
        ]
        cross_category = True

    if not pool:
        return 0

    scored = sorted(
        pool,
        key=lambda p: (p.capacity_score, p.health_score, p.effective_unique_yield_ratio),
        reverse=True,
    )
    moved = 0
    for tgt in scored:
        if moved >= deficit:
            break
        headroom = max(0, tgt.maximum_effective_quota - tgt.effective_quota)
        if headroom <= 0:
            continue
        step = min(deficit - moved, max(1, headroom))
        tgt.effective_quota += step
        tgt.redistributed_quota += step
        tgt.redistributed_in += step
        tgt.deficit_absorbed_total += step
        moved += step
        logger.info(
            "redistribution source=%s deficit=%s target=%s accepted=%s reason=%s mode=%s",
            donor_code,
            deficit,
            tgt.algorithm_code,
            step,
            reason,
            "cross_category" if cross_category else "same_category",
        )

    donor.effective_quota -= moved
    donor.redistributed_out += moved
    if cross_category:
        donor.observations.append(f"cross_category_redistribution:{moved}:{reason}")
    else:
        donor.observations.append(f"intra_category_redistribution:{moved}:{reason}")
    return moved


def _global_rebalance(plans: dict[str, AlgorithmPlan], remaining_global: int) -> None:
    active = [
        p
        for p in plans.values()
        if normalize_status(p.status) in ACTIVE_STATUSES and p.redistribution_eligibility
    ]
    if not active or remaining_global <= 0:
        return

    total_cap = sum(max(0.01, p.capacity_score * p.health_score) for p in active)
    assigned = 0
    for p in active:
        weight = max(0.01, p.capacity_score * p.health_score) / total_cap
        add = int(remaining_global * weight)
        if add <= 0:
            continue
        cap_room = max(0, p.maximum_effective_quota - p.effective_quota)
        accepted = min(add, cap_room)
        if accepted <= 0:
            continue
        prev = p.effective_quota
        p.effective_quota += accepted
        p.redistributed_quota += accepted
        p.redistributed_in += accepted
        p.deficit_absorbed_total += accepted
        assigned += accepted
        logger.info(
            "algorithm=%s quota_increase effective_quota=%s previous=%s reason=global_rebalance",
            p.algorithm_code,
            p.effective_quota,
            prev,
        )

    rem = remaining_global - assigned
    if rem > 0:
        scored = sorted(active, key=lambda p: p.capacity_score, reverse=True)
        idx = 0
        while rem > 0 and scored:
            tgt = scored[idx % len(scored)]
            if tgt.effective_quota < tgt.maximum_effective_quota:
                prev = tgt.effective_quota
                tgt.effective_quota += 1
                tgt.redistributed_quota += 1
                tgt.redistributed_in += 1
                tgt.deficit_absorbed_total += 1
                logger.info(
                    "algorithm=%s quota_increase effective_quota=%s previous=%s reason=global_rebalance_remainder",
                    tgt.algorithm_code,
                    tgt.effective_quota,
                    prev,
                )
            else:
                idx += 1
                if idx > len(scored) * 2:
                    break
                continue
            rem -= 1
            idx += 1


def _evaluate_algorithm_health(
    cfg: AppConfig,
    plan: AlgorithmPlan,
    inserted: int,
    generated: int,
    timed_out: bool,
    timeout_events: int = 0,
) -> None:
    if timed_out:
        plan.timeout_count += max(1, timeout_events)

    if generated == 0:
        plan.empty_batches += 1
        plan.consecutive_failures += 1
        plan.last_batch_unique_yield_ratio = 0.0
    else:
        plan.consecutive_failures = 0
        plan.last_batch_unique_yield_ratio = inserted / max(generated, 1)

    if inserted == 0:
        plan.consecutive_zero_unique_batches += 1
    else:
        plan.consecutive_zero_unique_batches = 0

    if plan.last_batch_unique_yield_ratio < cfg.saturation_min_yield:
        plan.consecutive_subthreshold_batches += 1
    else:
        plan.consecutive_subthreshold_batches = 0

    if plan.last_batch_unique_yield_ratio < cfg.min_unique_yield_ratio:
        plan.consecutive_low_yield_rounds += 1
    else:
        plan.consecutive_low_yield_rounds = 0

    plan.batch_yield_history.append(plan.last_batch_unique_yield_ratio)
    if len(plan.batch_yield_history) > 200:
        plan.batch_yield_history = plan.batch_yield_history[-200:]

    plan.recent_batch_yield_ratios.append(plan.last_batch_unique_yield_ratio)
    if len(plan.recent_batch_yield_ratios) > cfg.saturation_window:
        plan.recent_batch_yield_ratios = plan.recent_batch_yield_ratios[-cfg.saturation_window :]
    plan.recent_unique_gains.append(inserted)
    if len(plan.recent_unique_gains) > cfg.saturation_window:
        plan.recent_unique_gains = plan.recent_unique_gains[-cfg.saturation_window :]

    plan.rolling_unique_gain = sum(plan.recent_unique_gains)
    dup_recent = []
    for y in plan.recent_batch_yield_ratios:
        dup_recent.append(1.0 - y)
    plan.rolling_duplicate_rate = sum(dup_recent) / max(len(dup_recent), 1)
    plan.marginal_unique_gain = plan.rolling_unique_gain / max(len(plan.recent_unique_gains), 1)
    plan.effective_unique_yield_ratio = plan.unique_valid_count / max(plan.generated_count, 1)

    # Health / capacity evolve with yield + stability.
    med = median(plan.recent_batch_yield_ratios) if plan.recent_batch_yield_ratios else 0.0
    plan.health_score = max(
        0.01,
        min(1.5, med * 1.2 + (0.2 if plan.consecutive_failures == 0 else 0.0)),
    )
    plan.saturation_score = max(
        0.0,
        min(
            1.0,
            (plan.consecutive_zero_unique_batches / max(cfg.exhausted_after_zero_unique_batches, 1))
            * 0.6
            + (1.0 - med) * 0.4,
        ),
    )
    plan.capacity_score = max(
        0.01,
        min(
            3.0,
            plan.effective_unique_yield_ratio * 2.8
            + plan.expected_diversity_score * 0.8
            + (0.1 if plan.timeout_count == 0 else 0.0),
        ),
    )
    min_capacity_for_redistribution = cfg.redistribution_capacity_threshold
    if plan.algorithm_code.lower() == "zloader" and bool((plan.profiling or {}).get("collision_prone", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.6)
    if plan.algorithm_code.lower() == "nymaim" and bool((plan.profiling or {}).get("nymaim_medium_capacity", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.5)
    if plan.algorithm_code.lower() == "monerodownloader" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.35)
    if plan.algorithm_code.lower() == "bazarbackdoor" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.5)
    if plan.algorithm_code.lower() == "corebot" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.5)
    if plan.algorithm_code.lower() == "darkcracks" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.6)
    if plan.algorithm_code.lower() == "gozi" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.6)
    if plan.algorithm_code.lower() == "fobber" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.55)
    if plan.algorithm_code.lower() == "fosniw" and bool((plan.profiling or {}).get("structured_finite_combination", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.75)
    if bool((plan.profiling or {}).get("low_structural_diversity", False)):
        min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.9)
    if plan.algorithm_code.lower() == "qsnatch":
        invalid_ratio_total = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
        if invalid_ratio_total > 0.04 or plan.effective_unique_yield_ratio < 0.45:
            min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.55)
    if plan.algorithm_code.lower() == "bazarbackdoor":
        if plan.effective_unique_yield_ratio < 0.45:
            min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.6)
    if plan.algorithm_code.lower() == "corebot":
        if plan.effective_unique_yield_ratio < 0.5:
            min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.6)
    if plan.algorithm_code.lower() == "darkcracks":
        if plan.effective_unique_yield_ratio < 0.5:
            min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.65)
    if plan.algorithm_code.lower() == "gozi":
        if plan.effective_unique_yield_ratio < 0.5:
            min_capacity_for_redistribution = max(min_capacity_for_redistribution, 0.65)

    plan.redistribution_eligibility = (
        plan.capacity_score >= min_capacity_for_redistribution
        and normalize_status(plan.status) not in {"saturated", "exhausted", "discarded"}
        and plan.saturation_score < 0.85
    )

    # Capacity estimation and quota capping.
    plan.observed_unique_capacity_estimate = max(
        plan.observed_unique_capacity_estimate,
        max(plan.unique_valid_count, int(plan.unique_valid_count + plan.marginal_unique_gain * cfg.saturation_window)),
    )
    cap_from_observation = int(max(plan.planned_quota, plan.observed_unique_capacity_estimate * 1.2))
    cap_from_multiplier = int(max(plan.planned_quota, plan.planned_quota * cfg.max_effective_quota_multiplier))
    new_max_effective = min(max(cap_from_observation, plan.effective_quota), cap_from_multiplier)
    if new_max_effective < plan.maximum_effective_quota:
        logger.info(
            "algorithm=%s quota_cap_reduced previous=%s new=%s reason=observed_capacity",
            plan.algorithm_code,
            plan.maximum_effective_quota,
            new_max_effective,
        )
    plan.maximum_effective_quota = max(plan.effective_quota, new_max_effective)
    prof_estimated_cap = int((plan.profiling or {}).get("estimated_max_unique_capacity", 0) or 0)
    if prof_estimated_cap > 0:
        plan.maximum_effective_quota = min(plan.maximum_effective_quota, prof_estimated_cap)
        plan.effective_quota = min(plan.effective_quota, prof_estimated_cap)
    if plan.effective_quota > plan.maximum_effective_quota:
        plan.effective_quota = plan.maximum_effective_quota

    status = normalize_status(plan.status)
    if plan.error_count >= cfg.max_algorithm_errors:
        _status_change(plan, "discarded", "max_algorithm_errors")
        return
    if plan.timeout_count >= 4:
        _status_change(plan, "discarded", "repeated_timeouts")
        return
    if plan.timeout_count >= 2 and normalize_status(plan.status) in {"usable"}:
        _status_change(plan, "degraded", "timeout_threshold")
    if plan.empty_batches >= cfg.discard_after_consecutive_empty:
        _status_change(plan, "discarded", "consecutive_empty_batches")
        return

    remaining = plan.remaining_quota()
    if remaining <= cfg.near_quota_exhaustion_margin and inserted == 0:
        plan.near_quota_retries += 1
    else:
        plan.near_quota_retries = 0

    if plan.near_quota_retries >= cfg.near_quota_max_retries:
        plan.exhaustion_reason = "near_quota_retries_exceeded"
        _status_change(plan, "exhausted", "near_quota_retries_exceeded")
        return

    if plan.consecutive_zero_unique_batches >= cfg.exhausted_after_zero_unique_batches:
        if remaining <= cfg.near_quota_exhaustion_margin or med < cfg.saturation_min_yield:
            plan.exhaustion_reason = "zero_unique_batches"
            _status_change(plan, "exhausted", "zero_unique_batches")
            return
        _status_change(plan, "saturated", "zero_unique_batches")

    if len(plan.recent_batch_yield_ratios) >= cfg.saturation_window and med < cfg.saturation_min_yield:
        if status != "exhausted":
            _status_change(plan, "saturated", f"rolling_yield_below_threshold:{med:.5f}")

    if bool((plan.profiling or {}).get("structured_finite_combination", False)):
        delivered = plan.delivered_unique()
        quota = max(plan.effective_quota, plan.maximum_effective_quota, 1)
        near_ratio = float((plan.profiling or {}).get("recommended_near_capacity_ratio", 0.92) or 0.92)
        near_capacity = (delivered / quota) >= near_ratio or plan.remaining_quota() <= cfg.near_quota_exhaustion_margin
        if near_capacity and (
            plan.consecutive_subthreshold_batches >= 2
            or plan.consecutive_zero_unique_batches >= 2
            or plan.last_batch_unique_yield_ratio <= 0.005
        ):
            plan.exhaustion_reason = plan.exhaustion_reason or "near_capacity_exhaustion"
            _status_change(plan, "exhausted", "near_capacity_exhaustion")
            plan.redistribution_eligibility = False
            return

    if plan.consecutive_low_yield_rounds >= cfg.discard_after_consecutive_low_yield:
        _status_change(plan, "discarded", "consecutive_low_yield")
    elif plan.consecutive_low_yield_rounds >= cfg.low_yield_grace_rounds + 1:
        _status_change(plan, "partial", "extended_low_yield")
    elif plan.consecutive_low_yield_rounds >= cfg.low_yield_grace_rounds:
        _status_change(plan, "degraded", "low_yield")
    elif status in {"degraded", "partial", "saturated"} and plan.last_batch_unique_yield_ratio >= cfg.min_unique_yield_ratio * 1.5:
        _status_change(plan, "usable", "yield_recovered")

    current = normalize_status(plan.status)
    if current in {"saturated", "exhausted", "discarded"}:
        plan.redistribution_eligibility = False


def _build_plan(cfg: AppConfig):
    inspections = discover_and_inspect(cfg.algorithms_root)
    apply_categories(inspections)
    _apply_manual_overrides(cfg, inspections)
    ordered_mode = _is_ordered_capped_mode(cfg)
    ordered_algorithms = _resolve_ordered_algorithms(cfg) if ordered_mode else []
    ordered_set = {code.lower() for code in ordered_algorithms}
    ignored_discovered = (
        sorted([ins.algorithm_code for ins in inspections if ins.algorithm_code.lower() not in ordered_set])
        if ordered_mode
        else []
    )

    # Lightweight profiling stage before planning.
    for ins in inspections:
        if ordered_mode and ins.algorithm_code.lower() not in ordered_set:
            continue
        if normalize_status(ins.status) == "discarded" or ins.discard:
            ins.status = "discarded"
            if ins.discard_reason:
                logger.warning(
                    "algorithm=%s discarded_before_profiling reason=%s",
                    ins.algorithm_code,
                    ins.discard_reason,
                )
            continue
        try:
            dstart, dend, dpolicy = _date_bounds_for_inspection(cfg, ins)
            adapter = get_adapter(
                ins,
                cfg.seed_strategy,
                cfg.date_strategy,
                cfg.algorithm_timeout_seconds,
                cfg.algorithm_batch_timeout_seconds,
                cfg.max_cli_invocations_per_batch,
                dstart,
                dend,
                cfg.date_max_years_forward,
                cfg.date_max_years_backward,
                dpolicy,
            )
            profile = adapter.profile(sample_size=64)
            ins.profile = profile
            ins.supported_parameter_axes = profile.get("supported_parameter_axes", [])
            if not profile.get("generates_any", False):
                ins.status = "partial"
                ins.notes.append("profiling:no_output")
            else:
                ins.notes.append(
                    f"profiling:unique_yield={profile.get('sample_unique_yield', 0.0):.4f}"
                )
        except Exception as exc:
            ins.status = "discarded"
            if not ins.discard_reason:
                ins.discard_reason = "missing_implementation"
            ins.notes.append(f"profiling_error={exc}")

    if ordered_mode:
        selected = [ins for ins in inspections if ins.algorithm_code.lower() in ordered_set]
        plans = build_ordered_capped_plan(selected, ordered_algorithms, cfg.per_algorithm_cap)
        for idx, algo_code in enumerate(ordered_algorithms):
            p = plans.get(algo_code)
            if p is None:
                continue
            p.per_algorithm_cap = cfg.per_algorithm_cap
            p.order_index = idx
            p.redistribution_eligibility = False
            if not p.date_window:
                p.date_window = {"start": cfg.date_start, "end": cfg.date_end}
            if not p.date_wrap_policy:
                p.date_wrap_policy = cfg.date_wrap_policy
        return selected, plans, ignored_discovered

    plans = build_generation_plan(inspections, cfg.target_count)
    for code, p in plans.items():
        prof_mult = p.recommended_max_effective_quota_multiplier or 1.0
        multiplier = max(1.0, min(cfg.max_effective_quota_multiplier, max(2.5, prof_mult * 2.0)))
        p.maximum_effective_quota = max(
            p.effective_quota,
            int(max(p.planned_quota, p.planned_quota * multiplier)),
        )
        estimated_cap = int((p.profiling or {}).get("estimated_max_unique_capacity", 0) or 0)
        capacity_exhausted = False
        if estimated_cap > 0:
            p.maximum_effective_quota = min(p.maximum_effective_quota, estimated_cap)
            p.effective_quota = min(p.effective_quota, estimated_cap)
            p.planned_quota = min(p.planned_quota, estimated_cap)
            p.target_count = min(p.target_count, estimated_cap)
            if p.remaining_quota() <= 0:
                capacity_exhausted = True
        p.quota_confidence = max(0.1, min(1.0, p.expected_diversity_score + 0.1))
        p.redistribution_eligibility = p.capacity_score >= cfg.redistribution_capacity_threshold
        if p.algorithm_code.lower() == "qsnatch":
            invalid_rate = float((p.profiling or {}).get("invalid_rate", 0.0) or 0.0)
            if invalid_rate >= 0.05:
                p.redistribution_eligibility = False
            if invalid_rate >= 0.02:
                p.quota_confidence = min(p.quota_confidence, 0.7)
        if p.algorithm_code.lower() == "darkcracks":
            invalid_rate = float((p.profiling or {}).get("invalid_rate", 0.0) or 0.0)
            if invalid_rate >= 0.03:
                p.redistribution_eligibility = False
            if invalid_rate >= 0.01:
                p.quota_confidence = min(p.quota_confidence, 0.65)
        if p.algorithm_code.lower() == "gozi":
            invalid_rate = float((p.profiling or {}).get("invalid_rate", 0.0) or 0.0)
            if invalid_rate >= 0.02:
                p.redistribution_eligibility = False
            if invalid_rate >= 0.01:
                p.quota_confidence = min(p.quota_confidence, 0.7)
        if capacity_exhausted:
            p.redistribution_eligibility = False
        if not p.date_window:
            p.date_window = {"start": cfg.date_start, "end": cfg.date_end}
        if not p.date_wrap_policy:
            p.date_wrap_policy = cfg.date_wrap_policy
    return inspections, plans, []


def _date_bounds_for_inspection(cfg: AppConfig, ins) -> tuple[str, str, str]:
    window = ins.date_window or {}
    start = str(window.get("start", cfg.date_start))
    end = str(window.get("end", cfg.date_end))
    policy = ins.date_wrap_policy or cfg.date_wrap_policy
    return start, end, policy


def _apply_manual_overrides(cfg: AppConfig, inspections) -> None:
    overrides: dict[str, dict] = {}

    if cfg.plan_file and cfg.plan_file.exists():
        payload = read_json(cfg.plan_file)
        for item in payload.get("algorithms", []):
            code = item.get("algorithm_code")
            if code:
                overrides[code] = item

    if cfg.config_file and cfg.config_file.exists():
        payload = read_json(cfg.config_file)
        for code, ov in payload.get("algorithm_overrides", {}).items():
            base = overrides.get(code, {})
            base.update(ov)
            overrides[code] = base

    for ins in inspections:
        ov = overrides.get(ins.algorithm_code)
        if not ov:
            continue
        if "category" in ov:
            ins.category = ov["category"]
        if "strategy" in ov:
            ins.strategy = ov["strategy"]
        if "entrypoint" in ov:
            ins.entrypoint = ov["entrypoint"]
        if "callable_name" in ov:
            ins.callable_name = ov["callable_name"]
        if "module_name" in ov:
            ins.module_name = ov["module_name"]
        if "adapter_type" in ov:
            val = str(ov["adapter_type"]).strip().lower()
            ins.adapter_type = "cli_subprocess" if val == "cli" else val
        if "required_params" in ov and isinstance(ov["required_params"], list):
            ins.required_params = list(ov["required_params"])
        if "default_params" in ov and isinstance(ov["default_params"], dict):
            ins.default_params.update(ov["default_params"])
        if "requires_seed" in ov:
            ins.requires_seed = bool(ov["requires_seed"])
        if "requires_date" in ov:
            ins.requires_date = bool(ov["requires_date"])
        if "seed_parameter_name" in ov:
            ins.seed_parameter_name = ov["seed_parameter_name"]
        if "date_parameter_name" in ov:
            ins.date_parameter_name = ov["date_parameter_name"]
        if "counter_parameter_name" in ov:
            ins.counter_parameter_name = ov["counter_parameter_name"]
        if "force_scalar_mode" in ov:
            ins.force_scalar_mode = bool(ov["force_scalar_mode"])
        if "force_batch_mode" in ov:
            ins.force_batch_mode = bool(ov["force_batch_mode"])
        if "parameter_strategy" in ov and isinstance(ov["parameter_strategy"], dict):
            ins.parameter_strategy.update(ov["parameter_strategy"])
        if "discard" in ov:
            ins.discard = bool(ov["discard"])
            if ins.discard:
                ins.status = "discarded"
        if "priority_weight" in ov:
            try:
                ins.priority_weight = float(ov["priority_weight"])
            except Exception:
                pass
        if "max_cli_invocations_per_batch" in ov:
            try:
                ins.max_cli_invocations_per_batch = int(ov["max_cli_invocations_per_batch"])
            except Exception:
                pass
        if "algorithm_batch_timeout_seconds" in ov:
            try:
                ins.algorithm_batch_timeout_seconds = int(ov["algorithm_batch_timeout_seconds"])
            except Exception:
                pass
        if "status" in ov:
            ins.status = normalize_status(ov["status"])
        if "date_window" in ov and isinstance(ov["date_window"], dict):
            ins.date_window = ov["date_window"]
        if "date_start" in ov or "date_end" in ov:
            ins.date_window = {
                "start": ov.get("date_start", ins.date_window.get("start") if ins.date_window else None),
                "end": ov.get("date_end", ins.date_window.get("end") if ins.date_window else None),
            }
        if "date_wrap_policy" in ov:
            ins.date_wrap_policy = str(ov["date_wrap_policy"])
        if "max_effective_quota_multiplier" in ov:
            try:
                ins.parameter_strategy["max_effective_quota_multiplier"] = float(ov["max_effective_quota_multiplier"])
            except Exception:
                pass


def _materialize_plan_file(
    cfg: AppConfig,
    inspections,
    plans,
    ignored_discovered_algorithms: list[str] | None = None,
) -> None:
    payload = plan_to_json(
        plans,
        inspections,
        generation_mode=ORDERED_CAPPED_GENERATION_MODE if _is_ordered_capped_mode(cfg) else DEFAULT_GENERATION_MODE,
        ordered_algorithms=_resolve_ordered_algorithms(cfg) if _is_ordered_capped_mode(cfg) else None,
        per_algorithm_cap=cfg.per_algorithm_cap if _is_ordered_capped_mode(cfg) else None,
        ignored_discovered_algorithms=ignored_discovered_algorithms or [],
    )
    write_json(cfg.default_plan_path, payload)


def _load_or_create_state(cfg: AppConfig) -> tuple[GenerationRuntimeState, list, dict[str, AlgorithmPlan]]:
    inspections, plans, ignored_discovered = _build_plan(cfg)
    _materialize_plan_file(cfg, inspections, plans, ignored_discovered)
    ordered_mode = _is_ordered_capped_mode(cfg)
    expected_order = _resolve_ordered_algorithms(cfg) if ordered_mode else []

    if cfg.resume and cfg.state_path.exists():
        state = load_state(cfg.state_path)
        logger.info("Loaded resume state from %s", cfg.state_path)
        state_mode = str(state.generation_mode or DEFAULT_GENERATION_MODE).strip().lower()
        cfg_mode = ORDERED_CAPPED_GENERATION_MODE if ordered_mode else DEFAULT_GENERATION_MODE
        if state_mode != cfg_mode:
            raise GenerationError(
                f"Resume state generation_mode mismatch: state={state_mode} cfg={cfg_mode}"
            )
        if ordered_mode:
            if state.ordered_algorithms and state.ordered_algorithms != expected_order:
                raise GenerationError(
                    "Resume state ordered_algorithms mismatch for ordered_capped mode"
                )
            if state.per_algorithm_cap is not None and int(state.per_algorithm_cap) != int(cfg.per_algorithm_cap):
                raise GenerationError(
                    f"Resume state per_algorithm_cap mismatch: state={state.per_algorithm_cap} cfg={cfg.per_algorithm_cap}"
                )
            state.ordered_algorithms = expected_order
            state.per_algorithm_cap = cfg.per_algorithm_cap
            state.generation_mode = ORDERED_CAPPED_GENERATION_MODE
            state.ignored_discovered_algorithms = ignored_discovered
        # align missing plans from new discovery
        for code, plan in plans.items():
            if code not in state.plans:
                state.plans[code] = plan
        if ordered_mode:
            for idx, code in enumerate(expected_order):
                p = state.plans.get(code)
                if p is None:
                    continue
                p.order_index = idx
                p.per_algorithm_cap = cfg.per_algorithm_cap
                if normalize_status(p.status) in ACTIVE_STATUSES:
                    p.effective_quota = min(max(0, p.effective_quota), cfg.per_algorithm_cap)
                    p.planned_quota = min(max(0, p.planned_quota), cfg.per_algorithm_cap)
                    p.maximum_effective_quota = min(max(0, p.maximum_effective_quota), cfg.per_algorithm_cap)
        return state, inspections, state.plans

    state = _state_from_plans(cfg, plans)
    if ordered_mode:
        state.ignored_discovered_algorithms = ignored_discovered
    save_state(cfg.state_path, state)
    return state, inspections, plans


def _load_existing_unique(store: SQLiteDedupStore, state: GenerationRuntimeState) -> None:
    unique = store.count_unique()
    state.global_generated_unique = unique
    by_algo = store.count_by_algorithm()
    for code, cnt in by_algo.items():
        if code in state.plans:
            state.plans[code].unique_valid_count = cnt


def _validate_only(cfg: AppConfig) -> dict:
    store = SQLiteDedupStore(cfg.sqlite_path)
    try:
        unique = store.count_unique()
        if _is_ordered_capped_mode(cfg):
            counts = store.count_by_algorithm()
            over = {k: v for k, v in counts.items() if v > cfg.per_algorithm_cap}
            if over:
                raise GenerationError(f"Validation failed: algorithms above per_algorithm_cap={cfg.per_algorithm_cap}: {over}")
            return {"unique_domains": unique, "target_domains": None, "ok": True}
    finally:
        store.close()

    if unique != cfg.target_count:
        raise GenerationError(
            f"Validation failed: unique domains={unique}, expected={cfg.target_count}"
        )
    return {"unique_domains": unique, "target_domains": cfg.target_count, "ok": True}


def _adaptive_batch_request(cfg: AppConfig, plan: AlgorithmPlan, remaining_global: int) -> int:
    remaining_algo = plan.remaining_quota()
    if remaining_global <= 0:
        return 0

    if remaining_algo <= 0:
        plan.adaptive_batch_mode = "off_quota"
        return 0

    y = plan.last_batch_unique_yield_ratio
    if y >= 0.8:
        plan.adaptive_batch_mode = "normal"
        base = cfg.batch_size
    elif y >= 0.3:
        plan.adaptive_batch_mode = "medium"
        base = max(500, cfg.batch_size // 2)
    elif y >= 0.05:
        plan.adaptive_batch_mode = "small"
        base = max(100, cfg.batch_size // 10)
    else:
        plan.adaptive_batch_mode = "probe"
        base = min(50, max(5, cfg.batch_size // 100))

    if normalize_status(plan.status) == "saturated":
        plan.adaptive_batch_mode = "probe"
        base = min(base, 25)
    if normalize_status(plan.status) == "degraded":
        base = min(base, max(25, cfg.batch_size // 8))
    if normalize_status(plan.status) == "partial":
        base = min(base, max(10, cfg.batch_size // 20))

    if remaining_algo <= cfg.near_quota_exhaustion_margin:
        probe = min(remaining_algo, max(1, min(25, cfg.near_quota_exhaustion_margin // 8)))
        base = min(base, probe)
        plan.adaptive_batch_mode = "near_quota_probe"

    if bool((plan.profiling or {}).get("collision_prone", False)):
        if y < 0.02:
            base = min(base, max(10, cfg.batch_size // 200))
            plan.adaptive_batch_mode = "collision_probe"
        elif y < 0.08:
            base = min(base, max(25, cfg.batch_size // 100))
            plan.adaptive_batch_mode = "collision_small"
        else:
            base = min(base, max(100, cfg.batch_size // 20))
            plan.adaptive_batch_mode = "collision_medium"

    if plan.algorithm_code.lower() == "nymaim" and bool((plan.profiling or {}).get("nymaim_medium_capacity", False)):
        if y < 0.03:
            base = min(base, max(20, cfg.batch_size // 120))
            plan.adaptive_batch_mode = "nymaim_probe"
        elif y < 0.12:
            base = min(base, max(75, cfg.batch_size // 40))
            plan.adaptive_batch_mode = "nymaim_small"
        elif y < 0.3:
            base = min(base, max(200, cfg.batch_size // 12))
            plan.adaptive_batch_mode = "nymaim_medium"

    if bool((plan.profiling or {}).get("structured_finite_combination", False)):
        quota = max(plan.effective_quota, plan.maximum_effective_quota, 1)
        progress = plan.delivered_unique() / quota
        if progress >= 0.92 or remaining_algo <= cfg.near_quota_exhaustion_margin * 2:
            base = min(base, max(15, cfg.batch_size // 120))
            plan.adaptive_batch_mode = "structured_probe"
        elif progress >= 0.8:
            base = min(base, max(75, cfg.batch_size // 40))
            plan.adaptive_batch_mode = "structured_small"

    if bool((plan.profiling or {}).get("low_structural_diversity", False)):
        if y < 0.06:
            base = min(base, max(25, cfg.batch_size // 120))
            plan.adaptive_batch_mode = "low_diversity_probe"
        elif y < 0.15:
            base = min(base, max(80, cfg.batch_size // 35))
            plan.adaptive_batch_mode = "low_diversity_small"

    if plan.algorithm_code.lower() == "bazarbackdoor":
        if y < 0.04:
            base = min(base, max(20, cfg.batch_size // 140))
            plan.adaptive_batch_mode = "bazar_probe"
        elif y < 0.15:
            base = min(base, max(80, cfg.batch_size // 35))
            plan.adaptive_batch_mode = "bazar_small"

    if plan.algorithm_code.lower() == "corebot":
        inv_ratio = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
        if inv_ratio >= 0.05:
            base = min(base, max(20, cfg.batch_size // 120))
            plan.adaptive_batch_mode = "corebot_invalid_probe"
        elif y < 0.1:
            base = min(base, max(80, cfg.batch_size // 30))
            plan.adaptive_batch_mode = "corebot_small"

    if plan.algorithm_code.lower() == "darkcracks":
        inv_ratio = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
        if inv_ratio >= 0.03:
            base = min(base, max(20, cfg.batch_size // 140))
            plan.adaptive_batch_mode = "darkcracks_invalid_probe"
        elif y < 0.15:
            base = min(base, max(60, cfg.batch_size // 40))
            plan.adaptive_batch_mode = "darkcracks_small"

    if plan.algorithm_code.lower() == "gozi":
        inv_ratio = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
        if inv_ratio >= 0.02:
            base = min(base, max(20, cfg.batch_size // 140))
            plan.adaptive_batch_mode = "gozi_invalid_probe"
        elif y < 0.15:
            base = min(base, max(60, cfg.batch_size // 40))
            plan.adaptive_batch_mode = "gozi_small"

    if plan.algorithm_code.lower() == "qsnatch":
        inv_ratio = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
        if inv_ratio >= 0.08:
            base = min(base, max(20, cfg.batch_size // 120))
            plan.adaptive_batch_mode = "qsnatch_invalid_probe"
        elif inv_ratio >= 0.03:
            base = min(base, max(80, cfg.batch_size // 40))
            plan.adaptive_batch_mode = "qsnatch_invalid_small"
        elif y < 0.12:
            base = min(base, max(120, cfg.batch_size // 25))
            plan.adaptive_batch_mode = "qsnatch_dup_small"

    if plan.algorithm_code.lower() == "mydoom":
        if y < 0.5:
            base = min(base, max(40, cfg.batch_size // 80))
            plan.adaptive_batch_mode = "mydoom_probe"
        elif y < 0.75:
            base = min(base, max(120, cfg.batch_size // 20))
            plan.adaptive_batch_mode = "mydoom_small"
        elif y < 0.9:
            base = min(base, max(400, cfg.batch_size // 6))
            plan.adaptive_batch_mode = "mydoom_medium"

    if plan.algorithm_code.lower() == "newgoz":
        if y < 0.4:
            base = min(base, max(30, cfg.batch_size // 100))
            plan.adaptive_batch_mode = "newgoz_probe"
        elif y < 0.7:
            base = min(base, max(120, cfg.batch_size // 24))
            plan.adaptive_batch_mode = "newgoz_small"
        elif y < 0.85:
            base = min(base, max(350, cfg.batch_size // 8))
            plan.adaptive_batch_mode = "newgoz_medium"

    if plan.algorithm_code.lower() == "chinad":
        if y < 0.6:
            base = min(base, max(40, cfg.batch_size // 80))
            plan.adaptive_batch_mode = "chinad_probe"
        elif y < 0.85:
            base = min(base, max(160, cfg.batch_size // 18))
            plan.adaptive_batch_mode = "chinad_small"
        elif y < 0.95:
            base = min(base, max(500, cfg.batch_size // 5))
            plan.adaptive_batch_mode = "chinad_medium"

    ask = min(base, remaining_algo, remaining_global)
    return max(0, ask)


def _ordered_cap_for_plan(cfg: AppConfig, plan: AlgorithmPlan) -> int:
    cap = int(plan.per_algorithm_cap or cfg.per_algorithm_cap or 0)
    return max(0, cap)


def _ordered_stopped_reason(cfg: AppConfig, plan: AlgorithmPlan) -> str | None:
    status = normalize_status(plan.status)
    cap = _ordered_cap_for_plan(cfg, plan)
    if status == "missing":
        return "missing_algorithm"
    if cap > 0 and plan.delivered_unique() >= cap:
        return "cap_reached"
    if status == "saturated":
        return "saturated"
    if status == "exhausted":
        return "exhausted"
    if status == "discarded":
        if (plan.discard_reason or "") in {
            "missing_algorithm",
            "missing_implementation",
            "invalid_stub",
            "unresolved_redirect",
            "placeholder_module",
            "adapter_not_available",
            "invalid_algorithm",
        }:
            return "invalid_algorithm"
        if plan.generated_count == 0 and plan.delivered_unique() == 0:
            return "no_output"
        return "discarded"
    if plan.generated_count == 0 and plan.delivered_unique() == 0:
        return "no_output"
    return None


def _validate_ordered_capped_outputs(
    cfg: AppConfig,
    state: GenerationRuntimeState,
    payload: dict,
    exported_rows: int,
) -> None:
    ordered = payload.get("ordered_algorithms", [])
    expected_order = list(state.ordered_algorithms or _resolve_ordered_algorithms(cfg))
    if ordered != expected_order:
        raise GenerationError("ordered_capped validation failed: ordered_algorithms mismatch in stats payload")

    final_unique = int(payload.get("final_unique_domains", -1))
    unique_domains = int(payload.get("unique_domains", -1))
    if final_unique != unique_domains:
        raise GenerationError("ordered_capped validation failed: final_unique_domains != unique_domains")
    if exported_rows != final_unique:
        raise GenerationError(
            f"ordered_capped validation failed: CSV exported rows ({exported_rows}) != final_unique_domains ({final_unique})"
        )

    distribution = payload.get("distribution_by_algorithm", {}) or {}
    delivered_sum = 0
    for idx, code in enumerate(expected_order):
        algo_stats = distribution.get(code)
        if not isinstance(algo_stats, dict):
            raise GenerationError(f"ordered_capped validation failed: missing algorithm stats for '{code}'")
        if int(algo_stats.get("order_index", -1)) != idx:
            raise GenerationError(f"ordered_capped validation failed: order_index mismatch for '{code}'")
        delivered = int(algo_stats.get("delivered_unique", 0))
        cap = int(algo_stats.get("per_algorithm_cap", cfg.per_algorithm_cap))
        if delivered > cap:
            raise GenerationError(
                f"ordered_capped validation failed: algorithm '{code}' exceeded cap ({delivered}>{cap})"
            )
        delivered_sum += delivered

    if delivered_sum != final_unique:
        raise GenerationError(
            f"ordered_capped validation failed: sum(delivered_unique)={delivered_sum} != final_unique_domains={final_unique}"
        )


def run_pipeline(cfg: AppConfig) -> dict:
    if cfg.validate_only:
        return _validate_only(cfg)

    state, inspections, _ = _load_or_create_state(cfg)
    store = SQLiteDedupStore(Path(state.db_path))
    _load_existing_unique(store, state)

    started = parse_iso(state.started_at)
    adapters: dict[str, AlgorithmAdapter] = {}
    for ins in inspections:
        plan = state.plans.get(ins.algorithm_code)
        if not plan:
            continue
        if normalize_status(plan.status) in ACTIVE_STATUSES:
            try:
                if ins.algorithm_code.lower() == "orchard":
                    resume_domain_index = int((plan.last_effective_params or {}).get("next_domain_index", 0) or 0)
                    resume_record_index = int((plan.last_effective_params or {}).get("next_record_index", 0) or 0)
                    ins.parameter_strategy["resume_next_domain_index"] = resume_domain_index
                    ins.parameter_strategy["resume_next_record_index"] = resume_record_index
                if ins.algorithm_code.lower() == "m0yv":
                    resume_seed_offset = int((plan.last_effective_params or {}).get("next_seed_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_seed_offset"] = resume_seed_offset
                if ins.algorithm_code.lower() == "zloader":
                    resume_seed_offset = int((plan.last_effective_params or {}).get("next_seed_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_seed_offset"] = resume_seed_offset
                if ins.algorithm_code.lower() == "monerodownloader":
                    resume_domain_index = int((plan.last_effective_params or {}).get("next_domain_index", 0) or 0)
                    ins.parameter_strategy["resume_next_domain_index"] = resume_domain_index
                if ins.algorithm_code.lower() == "bazarbackdoor":
                    resume_idx = int((plan.last_effective_params or {}).get("next_global_index", 0) or 0)
                    ins.parameter_strategy["resume_next_global_index"] = resume_idx
                if ins.algorithm_code.lower() == "corebot":
                    resume_off = int((plan.last_effective_params or {}).get("next_schedule_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_schedule_offset"] = resume_off
                if ins.algorithm_code.lower() == "darkcracks":
                    resume_slot = int((plan.last_effective_params or {}).get("next_slot", 0) or 0)
                    ins.parameter_strategy["resume_next_slot"] = resume_slot
                if ins.algorithm_code.lower() == "gozi":
                    resume_slot = int((plan.last_effective_params or {}).get("next_slot", 0) or 0)
                    ins.parameter_strategy["resume_next_slot"] = resume_slot
                if ins.algorithm_code.lower() == "fobber":
                    sched = (plan.last_effective_params or {}).get("counter_schedule", {}) or {}
                    resume_v1 = int(sched.get("v1_next_counter", (plan.last_effective_params or {}).get("v1_next_counter", 0)) or 0)
                    resume_v2 = int(sched.get("v2_next_counter", (plan.last_effective_params or {}).get("v2_next_counter", 0)) or 0)
                    ins.parameter_strategy["resume_v1_next_counter"] = resume_v1
                    ins.parameter_strategy["resume_v2_next_counter"] = resume_v2
                if ins.algorithm_code.lower() == "fosniw":
                    resume_idx = int((plan.last_effective_params or {}).get("next_global_index", 0) or 0)
                    ins.parameter_strategy["resume_next_global_index"] = resume_idx
                if ins.algorithm_code.lower() == "banjori":
                    resume_counter = int((plan.last_effective_params or {}).get("next_counter", 0) or 0)
                    resume_domain = str((plan.last_effective_params or {}).get("current_domain", "") or "")
                    ins.parameter_strategy["resume_next_counter"] = resume_counter
                    if resume_domain:
                        ins.parameter_strategy["resume_current_domain"] = resume_domain
                if ins.algorithm_code.lower() == "dmsniff":
                    resume_idx = int((plan.last_effective_params or {}).get("next_global_index", 0) or 0)
                    ins.parameter_strategy["resume_next_global_index"] = resume_idx
                if ins.algorithm_code.lower() == "locky":
                    resume_off = int((plan.last_effective_params or {}).get("next_schedule_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_schedule_offset"] = resume_off
                if ins.algorithm_code.lower() == "ngioweb":
                    resume_counter = int((plan.last_effective_params or {}).get("next_counter", 0) or 0)
                    resume_seed_states = (plan.last_effective_params or {}).get("seed_states", {}) or {}
                    ins.parameter_strategy["resume_next_counter"] = resume_counter
                    ins.parameter_strategy["resume_seed_states"] = resume_seed_states
                if ins.algorithm_code.lower() == "nymaim":
                    resume_next_counter = int((plan.last_effective_params or {}).get("next_counter", 0) or 0)
                    ins.parameter_strategy["resume_next_counter"] = resume_next_counter
                if ins.algorithm_code.lower() == "mydoom":
                    resume_off = int((plan.last_effective_params or {}).get("next_schedule_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_schedule_offset"] = resume_off
                if ins.algorithm_code.lower() == "newgoz":
                    resume_off = int((plan.last_effective_params or {}).get("next_schedule_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_schedule_offset"] = resume_off
                if ins.algorithm_code.lower() == "chinad":
                    resume_off = int((plan.last_effective_params or {}).get("next_schedule_offset", 0) or 0)
                    ins.parameter_strategy["resume_next_schedule_offset"] = resume_off
                if ins.algorithm_code.lower() == "qsnatch":
                    resume_idx = int((plan.last_effective_params or {}).get("next_global_index", 0) or 0)
                    ins.parameter_strategy["resume_next_global_index"] = resume_idx
                if plan.strategy == "cli_subprocess" or ins.adapter_type in {"cli", "cli_subprocess"}:
                    cli_params = plan.last_effective_params or {}
                    ins.parameter_strategy["resume_cli_invocation_count"] = int(
                        cli_params.get("cli_invocation_count", 0) or 0
                    )
                    ins.parameter_strategy["resume_cli_sequence_key"] = cli_params.get(
                        "cli_sequence_key"
                    )
                    ins.parameter_strategy["resume_cli_sequence_offset"] = int(
                        cli_params.get("cli_sequence_offset", 0) or 0
                    )
                dstart, dend, dpolicy = _date_bounds_for_inspection(cfg, ins)
                adapters[ins.algorithm_code] = get_adapter(
                    ins,
                    cfg.seed_strategy,
                    cfg.date_strategy,
                    cfg.algorithm_timeout_seconds,
                    cfg.algorithm_batch_timeout_seconds,
                    cfg.max_cli_invocations_per_batch,
                    dstart,
                    dend,
                    cfg.date_max_years_forward,
                    cfg.date_max_years_backward,
                    dpolicy,
                )
            except Exception as exc:
                _status_change(plan, "discarded", f"adapter_error:{exc}")

    ordered_mode = _is_ordered_capped_mode(cfg)
    ordered_algorithms = list(state.ordered_algorithms or _resolve_ordered_algorithms(cfg))
    if ordered_mode:
        for idx, code in enumerate(ordered_algorithms):
            if code in state.plans:
                state.plans[code].order_index = idx
                state.plans[code].per_algorithm_cap = cfg.per_algorithm_cap
    if cfg.dry_run:
        if ordered_mode:
            sample_target = min(cfg.target_count, max(1, cfg.per_algorithm_cap * max(len(ordered_algorithms), 1)))
        else:
            sample_target = min(cfg.target_count, max(20_000, len(adapters) * 250))
    else:
        if ordered_mode:
            sample_target = sum(
                _ordered_cap_for_plan(cfg, state.plans[c])
                for c in ordered_algorithms
                if c in state.plans
            )
        else:
            sample_target = cfg.target_count

    q = deque(sorted(_active_codes(state.plans)))
    new_since_checkpoint = 0
    last_heartbeat = time.monotonic()
    loop_rounds = 0
    stagnant_rounds = 0
    last_unique_snapshot = state.global_generated_unique

    while state.global_generated_unique < sample_target:
        loop_rounds += 1

        now = time.monotonic()
        if now - last_heartbeat >= cfg.heartbeat_seconds:
            if ordered_mode:
                active = [
                    c
                    for c in ordered_algorithms[state.current_algorithm_index :]
                    if c in state.plans and normalize_status(state.plans[c].status) in ACTIVE_STATUSES
                ]
            else:
                active = _active_codes(state.plans)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            recent_rate = (
                (state.global_generated_unique - last_unique_snapshot) / max(now - last_heartbeat, 1e-6)
            )
            logger.info(
                "Heartbeat elapsed=%.1fs global_unique=%s/%s active_algorithms=%s recent_insert_rate=%.2f/s",
                elapsed,
                state.global_generated_unique,
                sample_target,
                len(active),
                recent_rate,
            )
            last_heartbeat = now
            last_unique_snapshot = state.global_generated_unique

        if ordered_mode:
            if state.current_algorithm_index >= len(ordered_algorithms):
                break
            algo = ordered_algorithms[state.current_algorithm_index]
            plan = state.plans.get(algo)
            if plan is None:
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
                continue

            cap = _ordered_cap_for_plan(cfg, plan)
            plan.per_algorithm_cap = cap
            plan.effective_quota = cap
            plan.planned_quota = cap
            plan.maximum_effective_quota = cap
            if normalize_status(plan.status) == "missing":
                plan.stopped_reason = "missing_algorithm"
                logger.warning(
                    "Stopping %s: missing_algorithm at ordered position %s",
                    algo,
                    state.current_algorithm_index + 1,
                )
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
                continue
            if plan.stopped_reason is None and normalize_status(plan.status) == "discarded" and plan.discard_reason:
                plan.stopped_reason = "invalid_algorithm"
            if plan.delivered_unique() >= cap:
                plan.stopped_reason = "cap_reached"
                logger.info(
                    "Stopping %s: cap_reached at %s unique domains",
                    algo,
                    plan.delivered_unique(),
                )
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
                continue
            status_norm = normalize_status(plan.status)
            if status_norm in {"discarded", "exhausted", "saturated"}:
                plan.stopped_reason = _ordered_stopped_reason(cfg, plan)
                logger.info(
                    "Stopping %s: %s after %s unique domains",
                    algo,
                    plan.stopped_reason or status_norm,
                    plan.delivered_unique(),
                )
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
                continue
            start_marker = f"ordered_started_index:{state.current_algorithm_index}"
            if start_marker not in plan.observations:
                logger.info(
                    "Starting ordered_capped algorithm %s/%s: %s (current_unique=%s)",
                    state.current_algorithm_index + 1,
                    len(ordered_algorithms),
                    algo,
                    plan.delivered_unique(),
                )
                plan.observations.append(start_marker)
        elif not q:
            remaining = sample_target - state.global_generated_unique
            _global_rebalance(state.plans, remaining)
            active = _active_codes(state.plans)
            if not active:
                break
            q = deque(sorted(active))
            algo = q.popleft()
            plan = state.plans[algo]
            status_norm = normalize_status(plan.status)
            if status_norm in {"discarded", "exhausted"}:
                continue
        else:
            algo = q.popleft()
            plan = state.plans[algo]
            status_norm = normalize_status(plan.status)
            if status_norm in {"discarded", "exhausted"}:
                continue

        if ordered_mode:
            remaining_global = max(0, _ordered_cap_for_plan(cfg, plan) - plan.delivered_unique())
            if remaining_global <= 0:
                plan.stopped_reason = "cap_reached"
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
                continue
        else:
            remaining_global = sample_target - state.global_generated_unique
            if remaining_global <= 0:
                break

        # Enforce hard quota semantics. No automatic oversupply.
        ask = _adaptive_batch_request(cfg, plan, remaining_global)
        if ask <= 0:
            if plan.remaining_quota() <= 0:
                logger.debug(
                    "algorithm=%s off_quota delivered=%s effective_quota=%s waiting_redistribution=%s",
                    algo,
                    plan.delivered_unique(),
                    plan.effective_quota,
                    plan.redistribution_eligibility,
                )
                if bool((plan.profiling or {}).get("collision_prone", False)) and normalize_status(plan.status) != "exhausted":
                    plan.exhaustion_reason = plan.exhaustion_reason or "observed_capacity_capped"
                    _status_change(plan, "exhausted", "observed_capacity_capped")
                    plan.redistribution_eligibility = False
            if ordered_mode:
                reason = _ordered_stopped_reason(cfg, plan)
                if reason:
                    plan.stopped_reason = reason
                    logger.info(
                        "Stopping %s: %s after %s unique domains",
                        algo,
                        plan.stopped_reason,
                        plan.delivered_unique(),
                    )
                    state.current_algorithm_index += 1
                    save_state(cfg.state_path, state)
            continue
        baseline = min(cfg.batch_size, remaining_global, max(plan.remaining_quota(), 1))
        if ask < baseline:
            logger.info(
                "algorithm=%s batch_size_reduced from=%s to=%s reason=%s",
                algo,
                baseline,
                ask,
                plan.adaptive_batch_mode,
            )

        adapter = adapters.get(algo)
        if adapter is None:
            _status_change(plan, "discarded", "adapter_not_available")
            if ordered_mode:
                plan.stopped_reason = "invalid_algorithm"
                logger.info(
                    "Stopping %s: invalid_algorithm after %s unique domains",
                    algo,
                    plan.delivered_unique(),
                )
                state.current_algorithm_index += 1
                save_state(cfg.state_path, state)
            else:
                _redistribute_from_algorithm(state.plans, algo, "adapter_not_available")
            continue

        logger.info(
            "Before batch algorithm=%s adapter=%s status=%s batch_request=%s delivered=%s/%s remaining_quota=%s oversupply=%s batch_mode=%s axes=%s",
            algo,
            adapter.__class__.__name__,
            plan.status,
            ask,
            plan.delivered_unique(),
            plan.effective_quota,
            plan.remaining_quota(),
            plan.oversupply(),
            plan.adaptive_batch_mode,
            plan.supported_parameter_axes,
        )

        t0 = time.monotonic()
        try:
            result = adapter.generate(ask)
        except Exception as exc:
            plan.error_count += 1
            plan.consecutive_failures += 1
            plan.observations.append(f"generate_error={exc}")
            logger.exception("Algorithm %s failed during generation", algo)
            _evaluate_algorithm_health(cfg, plan, inserted=0, generated=0, timed_out=False, timeout_events=0)
            logger.error("Batch aborted algorithm=%s reason=adapter_exception", algo)
            if ordered_mode:
                reason = _ordered_stopped_reason(cfg, plan)
                if reason:
                    plan.stopped_reason = reason
                    logger.info(
                        "Stopping %s: %s after %s unique domains",
                        algo,
                        plan.stopped_reason,
                        plan.delivered_unique(),
                    )
                    state.current_algorithm_index += 1
                    save_state(cfg.state_path, state)
            elif normalize_status(plan.status) != "discarded":
                q.append(algo)
            else:
                _redistribute_from_algorithm(state.plans, algo, "generation_exception")
            continue

        plan.last_effective_params = result.last_effective_params
        plan.supported_parameter_axes = sorted(set(plan.supported_parameter_axes + result.supported_parameter_axes))
        adapter_invalid_diag = (result.last_effective_params or {}).get("invalid_diagnostics", {})
        if isinstance(adapter_invalid_diag, dict):
            direct_reasons = {
                "double_dot",
                "empty_label",
                "label_too_long",
                "invalid_character",
                "invalid_suffix",
                "validator_rejected_multilevel_suffix",
                "empty",
                "none",
                "too_long",
                "format",
            }
            for k, v in adapter_invalid_diag.items():
                try:
                    n = int(v)
                except Exception:
                    continue
                if n <= 0:
                    continue
                key = k if k in direct_reasons else f"adapter_{k}"
                plan.invalid_reason_counts[key] = int(plan.invalid_reason_counts.get(key, 0)) + n

        finite_space = bool((result.last_effective_params or {}).get("finite_space", False))
        structured_space = bool((result.last_effective_params or {}).get("structured_space", False))
        if finite_space:
            est_cap = int((result.last_effective_params or {}).get("estimated_max_unique_capacity", 0) or 0)
            remaining_capacity = int((result.last_effective_params or {}).get("remaining_unique_capacity", 0) or 0)
            if est_cap > 0:
                plan.observed_unique_capacity_estimate = max(plan.observed_unique_capacity_estimate, est_cap)
                plan.maximum_effective_quota = min(max(plan.effective_quota, min(plan.maximum_effective_quota, est_cap)), est_cap)
                plan.effective_quota = min(plan.effective_quota, est_cap)
            if remaining_capacity <= max(1, cfg.near_quota_exhaustion_margin // 8):
                plan.redistribution_eligibility = False
            if structured_space and remaining_capacity <= max(1, cfg.near_quota_exhaustion_margin):
                plan.observations.append("structured_near_capacity_probe_mode")

        rows: list[tuple[str, str]] = []
        valid_local = 0
        invalid_local = 0
        for d in result.domains:
            pre_reason = _prevalidate_reason(d)
            if pre_reason:
                invalid_local += 1
                _inc_reason(plan, pre_reason)
                if pre_reason == "double_dot":
                    _inc_reason(plan, "malformed_suffix_join")
                continue
            vr = validate_domain(d)
            if not vr.is_valid or not vr.normalized:
                invalid_local += 1
                _inc_reason(plan, vr.reason or "format")
                if vr.reason == "invalid_suffix" and str(d).count(".") >= 2:
                    _inc_reason(plan, "validator_rejected_multilevel_suffix")
                continue
            valid_local += 1
            rows.append((vr.normalized, algo))

        plan.attempted_count += result.attempts
        state.global_attempted += result.attempts
        plan.generated_count += result.generated
        plan.valid_count += valid_local
        plan.invalid_count += invalid_local
        if result.errors:
            plan.observations.extend([f"adapter_error:{e}" for e in result.errors[-3:]])

        stats = store.insert_many(rows)
        inserted = stats.inserted
        duplicates = stats.attempted - inserted

        plan.unique_valid_count += inserted
        plan.duplicate_count += duplicates

        state.global_generated_unique += inserted
        state.total_duplicates += duplicates
        new_since_checkpoint += inserted

        batch_elapsed = time.monotonic() - t0

        _evaluate_algorithm_health(
            cfg,
            plan,
            inserted=inserted,
            generated=max(result.generated, 0),
            timed_out=result.timed_out,
            timeout_events=result.timeout_events,
        )

        if finite_space:
            remaining_capacity = int((result.last_effective_params or {}).get("remaining_unique_capacity", 0) or 0)
            if remaining_capacity <= 0:
                if structured_space:
                    plan.exhaustion_reason = "structured_space_consumed"
                    _status_change(plan, "exhausted", "structured_space_consumed")
                else:
                    plan.exhaustion_reason = "estimated_capacity_reached"
                    _status_change(plan, "exhausted", "estimated_capacity_reached")
                plan.redistribution_eligibility = False

        if bool((plan.profiling or {}).get("collision_prone", False)):
            if plan.consecutive_zero_unique_batches >= 2 and plan.last_batch_unique_yield_ratio <= 0.001:
                if plan.remaining_quota() <= cfg.near_quota_exhaustion_margin or plan.consecutive_zero_unique_batches >= 3:
                    plan.exhaustion_reason = "collision_plateau"
                    _status_change(plan, "exhausted", "collision_plateau")
                else:
                    _status_change(plan, "saturated", "collision_plateau")
                plan.redistribution_eligibility = False
            elif plan.remaining_quota() <= 0 and normalize_status(plan.status) not in {"exhausted", "discarded"}:
                plan.exhaustion_reason = plan.exhaustion_reason or "observed_capacity_capped"
                _status_change(plan, "exhausted", "observed_capacity_capped")
                plan.redistribution_eligibility = False

        if plan.algorithm_code.lower() == "nymaim" and bool((plan.profiling or {}).get("nymaim_medium_capacity", False)):
            if plan.consecutive_subthreshold_batches >= 3 and plan.last_batch_unique_yield_ratio < max(cfg.saturation_min_yield, 0.1):
                if plan.remaining_quota() <= cfg.near_quota_exhaustion_margin or plan.consecutive_zero_unique_batches >= 2:
                    plan.exhaustion_reason = plan.exhaustion_reason or "nymaim_saturation_plateau"
                    _status_change(plan, "exhausted", "nymaim_saturation_plateau")
                else:
                    _status_change(plan, "saturated", "nymaim_saturation_plateau")
                plan.redistribution_eligibility = False

        if structured_space and plan.consecutive_zero_unique_batches >= 2 and normalize_status(plan.status) not in {"exhausted", "discarded"}:
            plan.exhaustion_reason = plan.exhaustion_reason or "near_capacity_exhaustion"
            _status_change(plan, "exhausted", "near_capacity_exhaustion")
            plan.redistribution_eligibility = False

        if bool((plan.profiling or {}).get("low_structural_diversity", False)):
            if plan.rolling_duplicate_rate >= 0.92 and plan.consecutive_subthreshold_batches >= 2:
                if plan.remaining_quota() <= cfg.near_quota_exhaustion_margin or plan.consecutive_zero_unique_batches >= 2:
                    plan.exhaustion_reason = plan.exhaustion_reason or "finite_capacity_exhaustion"
                    _status_change(plan, "exhausted", "finite_capacity_exhaustion")
                else:
                    _status_change(plan, "saturated", "finite_capacity_exhaustion")
                plan.redistribution_eligibility = False
                if isinstance(plan.last_effective_params, dict):
                    plan.last_effective_params["saturation_point"] = plan.unique_valid_count

        if plan.algorithm_code.lower() == "qsnatch":
            inv_ratio = plan.invalid_count / max(plan.valid_count + plan.invalid_count, 1)
            if inv_ratio >= 0.08 and normalize_status(plan.status) not in {"exhausted", "discarded"}:
                _status_change(plan, "degraded", "qsnatch_invalid_rate_high")
                plan.redistribution_eligibility = False

        if result.timed_out:
            plan.observations.append("timeout")
        if result.stdout_summary:
            plan.observations.append(f"stdout:{result.stdout_summary[:120]}")
        if result.stderr_summary:
            plan.observations.append(f"stderr:{result.stderr_summary[:120]}")

        if result.batch_aborted:
            logger.warning(
                "Batch aborted algorithm=%s reason=%s generated=%s valid=%s inserted_unique=%s duplicates=%s invalid=%s batch_elapsed=%.3fs status=%s params=%s",
                algo,
                result.abort_reason,
                result.generated,
                valid_local,
                inserted,
                duplicates,
                invalid_local,
                batch_elapsed,
                plan.status,
                result.last_effective_params,
            )
        else:
            logger.info(
                "After batch algorithm=%s generated=%s valid=%s inserted_unique=%s duplicates=%s invalid=%s batch_elapsed=%.3fs yield_ratio=%.5f status=%s params=%s",
                algo,
                result.generated,
                valid_local,
                inserted,
                duplicates,
                invalid_local,
                batch_elapsed,
                plan.last_batch_unique_yield_ratio,
                plan.status,
                result.last_effective_params,
            )

        if plan.oversupply() > 0:
            logger.warning(
                "algorithm=%s oversupply=%s delivered=%s effective_quota=%s reason=%s",
                algo,
                plan.oversupply(),
                plan.delivered_unique(),
                plan.effective_quota,
                "effective_quota_not_raised" if plan.redistributed_in == 0 else "redistribution_raised_quota",
            )

        status_now = normalize_status(plan.status)
        if ordered_mode:
            stop_reason = _ordered_stopped_reason(cfg, plan)
            if stop_reason:
                plan.stopped_reason = stop_reason
                logger.info(
                    "Stopping %s: %s after %s unique domains",
                    algo,
                    stop_reason,
                    plan.delivered_unique(),
                )
                state.current_algorithm_index += 1
            else:
                plan.stopped_reason = None
        elif status_now in {"discarded", "exhausted"}:
            moved = _redistribute_from_algorithm(state.plans, algo, f"{status_now}_after_batch")
            if moved > 0:
                logger.info(
                    "algorithm=%s redistributed_out=%s reason=%s remaining_deficit=%s",
                    algo,
                    moved,
                    status_now,
                    plan.remaining_quota(),
                )
        else:
            q.append(algo)

        if not ordered_mode:
            if inserted == 0:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0

            if stagnant_rounds > max(len(adapters) * 3, 30):
                remaining = sample_target - state.global_generated_unique
                _global_rebalance(state.plans, remaining)
                active = _active_codes(state.plans)
                if not active:
                    break
                if all(state.plans[c].capacity_score < 0.02 for c in active):
                    logger.warning("No feasible productive algorithms remain; stopping early")
                    break
                stagnant_rounds = 0

        if new_since_checkpoint >= cfg.checkpoint_every:
            store.commit()
            save_state(cfg.state_path, state)
            new_since_checkpoint = 0
            logger.info("Checkpoint: unique=%s/%s", state.global_generated_unique, sample_target)

        # periodic persistence for long runs
        if loop_rounds % 50 == 0:
            save_state(cfg.state_path, state)

    store.commit()
    if ordered_mode:
        state.finished = state.current_algorithm_index >= len(ordered_algorithms)
    else:
        state.finished = state.global_generated_unique >= sample_target
    save_state(cfg.state_path, state)

    exported = export_csv(store, cfg.csv_path)
    finished = datetime.now(timezone.utc)

    unreachable = sample_target - state.global_generated_unique
    if unreachable > 0 and not ordered_mode:
        logger.warning("Target not fully reached. remaining_unique_deficit=%s", unreachable)

    unique_count = store.count_unique()
    payload = build_stats_payload(
        started_at=started,
        finished_at=finished,
        plans=state.plans,
        target_count=sample_target,
        unique_count=unique_count,
        duplicate_count=state.total_duplicates,
        params={
            "algorithms_root": str(cfg.algorithms_root),
            "output_dir": str(cfg.output_dir),
            "target_count": cfg.target_count,
            "effective_target_count": unique_count if ordered_mode else sample_target,
            "batch_size": cfg.batch_size,
            "threads": cfg.threads,
            "checkpoint_every": cfg.checkpoint_every,
            "seed_strategy": cfg.seed_strategy,
            "date_strategy": cfg.date_strategy,
            "resume": cfg.resume,
            "dry_run": cfg.dry_run,
            "generation_mode": ORDERED_CAPPED_GENERATION_MODE if ordered_mode else DEFAULT_GENERATION_MODE,
            "per_algorithm_cap": cfg.per_algorithm_cap if ordered_mode else None,
            "ordered_algorithms": ordered_algorithms if ordered_mode else None,
            "global_target_count": None if ordered_mode else cfg.target_count,
            "dedup_backend": cfg.dedup_backend,
            "algorithm_timeout_seconds": cfg.algorithm_timeout_seconds,
            "algorithm_batch_timeout_seconds": cfg.algorithm_batch_timeout_seconds,
            "max_cli_invocations_per_batch": cfg.max_cli_invocations_per_batch,
            "min_unique_yield_ratio": cfg.min_unique_yield_ratio,
            "discard_after_consecutive_empty": cfg.discard_after_consecutive_empty,
            "discard_after_consecutive_low_yield": cfg.discard_after_consecutive_low_yield,
            "low_yield_grace_rounds": cfg.low_yield_grace_rounds,
            "max_algorithm_errors": cfg.max_algorithm_errors,
            "heartbeat_seconds": cfg.heartbeat_seconds,
            "saturation_window": cfg.saturation_window,
            "saturation_min_yield": cfg.saturation_min_yield,
            "exhausted_after_zero_unique_batches": cfg.exhausted_after_zero_unique_batches,
            "near_quota_exhaustion_margin": cfg.near_quota_exhaustion_margin,
            "near_quota_max_retries": cfg.near_quota_max_retries,
            "date_start": cfg.date_start,
            "date_end": cfg.date_end,
            "date_max_years_forward": cfg.date_max_years_forward,
            "date_max_years_backward": cfg.date_max_years_backward,
            "date_wrap_policy": cfg.date_wrap_policy,
            "max_effective_quota_multiplier": cfg.max_effective_quota_multiplier,
            "redistribution_capacity_threshold": cfg.redistribution_capacity_threshold,
            "exported_rows": exported,
            "target_unreachable": (unreachable > 0) if not ordered_mode else False,
            "remaining_deficit": max(0, unreachable) if not ordered_mode else 0,
            "ignored_discovered_algorithms": state.ignored_discovered_algorithms if ordered_mode else [],
            "failed_algorithms": [
                c for c, p in state.plans.items() if normalize_status(p.status) == "discarded"
            ],
        },
        generation_mode=ORDERED_CAPPED_GENERATION_MODE if ordered_mode else DEFAULT_GENERATION_MODE,
        per_algorithm_cap=cfg.per_algorithm_cap if ordered_mode else None,
        ordered_algorithms=ordered_algorithms if ordered_mode else None,
        ignored_discovered_algorithms=state.ignored_discovered_algorithms if ordered_mode else None,
    )
    write_json(cfg.stats_path, payload)
    _materialize_plan_file(cfg, inspections, state.plans, state.ignored_discovered_algorithms)

    if ordered_mode:
        _validate_ordered_capped_outputs(cfg, state, payload, exported)

    store.close()

    if not ordered_mode and not cfg.dry_run and state.global_generated_unique < cfg.target_count:
        raise GenerationError(
            f"Could not reach target unique domains. generated={state.global_generated_unique} target={cfg.target_count}"
        )

    return payload
