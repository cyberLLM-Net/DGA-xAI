from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


@dataclass(frozen=True)
class DmsniffEntry:
    family: str
    index: int
    domain: str


class DmsniffAdapter(AlgorithmAdapter):
    """Dedicated low-capacity structured adapter for dmsniff."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        if self.dga_fn is None or not callable(self.dga_fn):
            raise RuntimeError("dmsniff:missing_implementation:callable_dga_not_found")

        self.supported_families = self._resolve_supported_families()
        self.requested_families = self._resolve_requested_families()
        self.accepted_families: list[str] = []
        self.rejected_families: dict[str, str] = {}

        self.family_domains: dict[str, list[str]] = {}
        self.profile_data = self._build_profile_data()
        self.entries = self._build_entries()
        self.estimated_max_unique_capacity = len(self.entries)

        self.next_global_index = self._resolve_resume_global_index()
        self.next_global_index = min(self.next_global_index, self.estimated_max_unique_capacity)
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"dmsniff:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_dmsniff", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"dmsniff:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_supported_families(self) -> list[str]:
        # implementation exposes exactly these in argparse
        return ["al", "sn"]

    def _resolve_requested_families(self) -> list[str]:
        raw = self.defaults.get("dmsniff_families", self.strategy_cfg.get("dmsniff_families", "all"))
        if isinstance(raw, str):
            txt = raw.strip().lower()
            if txt in {"all", "*"}:
                return list(self.supported_families)
            return [x.strip().lower() for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip().lower() for x in raw if str(x).strip()]
        return list(self.supported_families)

    def _resolve_resume_global_index(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_global_index")
            or self.strategy_cfg.get("next_global_index")
            or self.defaults.get("resume_next_global_index")
            or self.defaults.get("next_global_index")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _domains_for_family(self, fam: str) -> tuple[list[str], dict[str, int]]:
        generated = 0
        valid = 0
        invalid = 0
        unique: list[str] = []
        seen: set[str] = set()
        for d in self.dga_fn(fam):
            generated += 1
            vr = validate_domain(str(d))
            if vr.is_valid and vr.normalized:
                valid += 1
                if vr.normalized not in seen:
                    seen.add(vr.normalized)
                    unique.append(vr.normalized)
            else:
                invalid += 1
        return unique, {"generated": generated, "valid": valid, "invalid": invalid}

    def _build_profile_data(self) -> dict[str, Any]:
        tested = sorted(set(self.requested_families))
        for fam in tested:
            if fam not in self.supported_families:
                self.rejected_families[fam] = "unsupported_family"

        per_family: list[dict[str, Any]] = []
        combined: list[str] = []
        for fam in tested:
            if fam in self.rejected_families:
                continue
            domains, st = self._domains_for_family(fam)
            self.family_domains[fam] = domains
            combined.extend(domains)
            unique = len(set(domains))
            generated = int(st["generated"])
            valid = int(st["valid"])
            dup_ratio = 1.0 - (unique / max(valid, 1))
            info = {
                "family": fam,
                "generated": generated,
                "valid": valid,
                "invalid": int(st["invalid"]),
                "unique": unique,
                "unique_yield": unique / max(valid, 1),
                "duplicate_ratio": dup_ratio,
                "capacity_estimate": unique,
                "prefix_constancy": 1.0,
                "small_tld_set": [".com", ".org", ".net", ".ru", ".in"],
            }
            per_family.append(info)
            if unique > 0:
                self.accepted_families.append(fam)

        self.accepted_families = sorted(set(self.accepted_families))
        all_unique = len(set(combined))
        all_generated = sum(int(x["valid"]) for x in per_family)
        overall_yield = all_unique / max(all_generated, 1)
        duplicate_curve = []
        running: set[str] = set()
        seen = 0
        for i, d in enumerate(combined, start=1):
            seen += 1
            running.add(d)
            if i % max(5, len(combined) // 6 or 1) == 0:
                duplicate_curve.append(
                    {
                        "sample_step": i,
                        "unique": len(running),
                        "duplicate_ratio": 1.0 - (len(running) / max(seen, 1)),
                    }
                )

        family_caps = {x["family"]: int(x["capacity_estimate"]) for x in per_family}
        estimated_capacity = max(1, all_unique)
        cap_aggressive = min(estimated_capacity, max(50, min(500, estimated_capacity)))
        return {
            "supported_families": self.supported_families,
            "requested_families": tested,
            "accepted_families": self.accepted_families,
            "rejected_families": self.rejected_families,
            "per_family": sorted(per_family, key=lambda x: x["family"]),
            "family_capacity_estimates": family_caps,
            "sample_generated": all_generated,
            "sample_unique": all_unique,
            "sample_unique_yield": overall_yield,
            "duplicate_growth_curve": duplicate_curve,
            "prefix_constancy": 1.0,
            "low_capacity_structured": True,
            "low_structural_diversity": True,
            "theoretical_capacity_estimate": estimated_capacity,
            "estimated_max_unique_capacity": estimated_capacity,
            "recommended_effective_quota_cap": cap_aggressive,
            "recommended_near_capacity_ratio": 0.8,
            "failure_classification": {
                "unsupported_pattern": "unsupported_family",
                "invalid_pattern_schedule": "no_accepted_families",
                "finite_capacity_exhaustion": "low_capacity_structured",
            },
        }

    def _build_entries(self) -> list[DmsniffEntry]:
        out: list[DmsniffEntry] = []
        seen: set[str] = set()
        for fam in self.accepted_families:
            for idx, d in enumerate(self.family_domains.get(fam, [])):
                if d in seen:
                    continue
                seen.add(d)
                out.append(DmsniffEntry(family=fam, index=idx, domain=d))
        return out

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="dmsniff_dedicated")
        start = self.next_global_index
        end = min(start + batch_size, self.estimated_max_unique_capacity)
        selected = self.entries[start:end]
        self.next_global_index = end
        fam_usage: dict[str, int] = {}
        for e in selected:
            fam_usage[e.family] = fam_usage.get(e.family, 0) + 1
        params = {
            "generation_mode": "family_structured_low_capacity",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "detected_families": self.accepted_families,
            "family_capacity_estimates": self.profile_data["family_capacity_estimates"],
            "selected_family_schedule": [e.family for e in selected[:32]],
            "family_usage": fam_usage,
            "prefix_constancy": self.profile_data["prefix_constancy"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "theoretical_capacity_estimate": self.profile_data["theoretical_capacity_estimate"],
            "recommended_effective_quota_cap": self.profile_data["recommended_effective_quota_cap"],
            "next_global_index": self.next_global_index,
            "remaining_unique_capacity": max(0, self.estimated_max_unique_capacity - self.next_global_index),
            "low_capacity_structured": True,
            "low_structural_diversity": True,
            "finite_space": True,
            "structured_space": True,
        }
        return BatchGenerationResult(
            domains=[e.domain for e in selected],
            attempts=1,
            generated=len(selected),
            adapter_type="dmsniff_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["family", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        tail = median([f["unique_yield"] for f in self.profile_data["per_family"]]) if self.profile_data["per_family"] else 0.0
        self._profile_cache = {
            "generates_any": self.estimated_max_unique_capacity > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": False,
            "supported_parameter_axes": ["family", "counter"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": tail,
            "initial_health_score": min(0.3, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(0.25, max(0.05, self.profile_data["sample_unique_yield"])),
            "expected_diversity_score": min(0.2, max(0.02, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.15,
            "apparent_finite_space": True,
            "generation_mode": "family_structured_low_capacity",
            "adapter_type": "dmsniff_dedicated",
            "detected_families": self.profile_data["accepted_families"],
            "supported_families": self.profile_data["supported_families"],
            "rejected_families": self.profile_data["rejected_families"],
            "per_family": self.profile_data["per_family"],
            "family_capacity_estimates": self.profile_data["family_capacity_estimates"],
            "duplicate_growth_curve": self.profile_data["duplicate_growth_curve"],
            "prefix_constancy": self.profile_data["prefix_constancy"],
            "low_capacity_structured": True,
            "low_structural_diversity": True,
            "theoretical_capacity_estimate": self.profile_data["theoretical_capacity_estimate"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "recommended_effective_quota_cap": self.profile_data["recommended_effective_quota_cap"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "failure_classification": self.profile_data["failure_classification"],
        }
        return self._profile_cache
