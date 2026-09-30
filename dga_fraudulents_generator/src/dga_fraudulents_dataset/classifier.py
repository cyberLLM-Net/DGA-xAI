from __future__ import annotations

from .models import AlgorithmInspection


def infer_category(inspection: AlgorithmInspection) -> str | None:
    params = {p.lower() for p in inspection.required_params}
    name = inspection.algorithm_code.lower()

    if "date" in params or params.intersection({"d", "dt", "when", "time"}):
        return "date_based"
    if "seed" in params or "magic" in params:
        return "seed_based"
    if any(k in name for k in ["locky", "qakbot", "necurs", "ramnit", "tinba"]):
        return "known_family"
    return None


def apply_categories(inspections: list[AlgorithmInspection]) -> None:
    for ins in inspections:
        if ins.category is None:
            ins.category = infer_category(ins)
