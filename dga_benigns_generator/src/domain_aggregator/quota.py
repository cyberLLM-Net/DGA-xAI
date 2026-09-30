from __future__ import annotations

from dataclasses import dataclass
from math import floor

ROUNDING_STRATEGY_LARGEST_REMAINDER = "largest_remainder"


@dataclass(frozen=True)
class QuotaAllocationDetail:
    index: int
    count: int
    raw_quota: float
    floor_quota: int
    fractional_remainder: float
    rounding_adjustment: int
    assigned_quota: int


@dataclass(frozen=True)
class QuotaAllocation:
    quotas: list[int]
    details: list[QuotaAllocationDetail]
    target_size: int
    floor_total: int
    rounding_adjustment_total: int
    rounding_strategy: str = ROUNDING_STRATEGY_LARGEST_REMAINDER


def allocate_quotas_with_details(counts: list[int], target_size: int) -> QuotaAllocation:
    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    if not counts:
        return QuotaAllocation(
            quotas=[],
            details=[],
            target_size=target_size,
            floor_total=0,
            rounding_adjustment_total=0,
        )

    total = sum(counts)
    if total <= 0:
        raise ValueError("Total rows for allocation must be greater than zero")

    raw_quotas = [target_size * count / total for count in counts]
    floor_quotas = [floor(raw) for raw in raw_quotas]
    fractional_remainders = [raw_quotas[i] - floor_quotas[i] for i in range(len(counts))]

    floor_total = sum(floor_quotas)
    remainder = target_size - floor_total
    adjustments = [0] * len(counts)

    ranked = sorted(
        [(fractional_remainders[i], i) for i in range(len(counts))],
        key=lambda pair: (-pair[0], pair[1]),
    )
    for _, idx in ranked[:remainder]:
        adjustments[idx] += 1

    quotas = [floor_quotas[i] + adjustments[i] for i in range(len(counts))]
    details = [
        QuotaAllocationDetail(
            index=i,
            count=counts[i],
            raw_quota=raw_quotas[i],
            floor_quota=floor_quotas[i],
            fractional_remainder=fractional_remainders[i],
            rounding_adjustment=adjustments[i],
            assigned_quota=quotas[i],
        )
        for i in range(len(counts))
    ]

    return QuotaAllocation(
        quotas=quotas,
        details=details,
        target_size=target_size,
        floor_total=floor_total,
        rounding_adjustment_total=remainder,
    )


def allocate_quotas(counts: list[int], target_size: int) -> list[int]:
    return allocate_quotas_with_details(counts, target_size).quotas


def allocate_quotas_from_raw_rows(raw_rows: list[int], target_size: int) -> QuotaAllocation:
    return allocate_quotas_with_details(raw_rows, target_size)
