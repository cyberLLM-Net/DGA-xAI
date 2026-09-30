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
class FosniwEntry:
    pattern: str
    counter: int
    domain: str


class FosniwAdapter(AlgorithmAdapter):
    """Pattern-aware dedicated adapter for fosniw.

    fosniw is a finite, template-driven generator keyed by a small
    pattern set (for example: koreasys, winsoft).
    """

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        if self.dga_fn is None or not callable(self.dga_fn):
            raise RuntimeError("fosniw:missing_implementation:callable_dga_not_found")

        self.supported_patterns = self._discover_supported_patterns()
        self.requested_patterns = self._resolve_requested_patterns()
        self.accepted_patterns: list[str] = []
        self.rejected_patterns: dict[str, str] = {}
        self.per_pattern: list[dict[str, Any]] = []

        self.pattern_domains: dict[str, list[str]] = {}
        self.entries: list[FosniwEntry] = []

        self.next_global_index = self._resolve_resume_global_index()
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self.next_global_index = min(self.next_global_index, self.estimated_max_unique_capacity)
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"fosniw:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_fosniw", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"fosniw:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _discover_supported_patterns(self) -> list[str]:
        patterns = getattr(self.module, "PATTERNS", None)
        if isinstance(patterns, dict):
            keys = [str(k).strip() for k in patterns.keys() if str(k).strip()]
            return sorted(set(keys))
        return []

    def _resolve_requested_patterns(self) -> list[str]:
        raw = self.defaults.get("fosniw_patterns", self.strategy_cfg.get("fosniw_patterns", "all"))
        if isinstance(raw, str):
            txt = raw.strip().lower()
            if txt in {"all", "*"}:
                return list(self.supported_patterns)
            return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        return list(self.supported_patterns)

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

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.02
        elif tail_yield <= 0.08:
            factor = 1.08
        elif tail_yield <= 0.2:
            factor = 1.18
        else:
            factor = 1.35
        return max(sample_unique, int(sample_unique * factor))

    def _domains_for_pattern(self, pattern: str) -> tuple[list[str], dict[str, int]]:
        generated = 0
        valid = 0
        invalid = 0
        unique: list[str] = []
        seen: set[str] = set()

        raw_iter = self.dga_fn(pattern)
        for item in raw_iter:
            generated += 1
            vr = validate_domain(str(item))
            if vr.is_valid and vr.normalized:
                valid += 1
                if vr.normalized not in seen:
                    seen.add(vr.normalized)
                    unique.append(vr.normalized)
            else:
                invalid += 1
        return unique, {"generated": generated, "valid": valid, "invalid": invalid}

    def _build_profile_data(self) -> dict[str, Any]:
        if not self.supported_patterns:
            raise RuntimeError("fosniw:missing_implementation:supported_patterns_not_found")

        tested = sorted(set(self.requested_patterns))
        for p in tested:
            if p not in self.supported_patterns:
                self.rejected_patterns[p] = "unsupported_pattern"

        all_unique: set[str] = set()
        all_valid = 0

        for p in tested:
            if p in self.rejected_patterns:
                continue
            domains, st = self._domains_for_pattern(p)
            self.pattern_domains[p] = domains
            all_unique.update(domains)
            all_valid += int(st["valid"])
            unique_count = len(domains)
            valid_rate = st["valid"] / max(st["generated"], 1)
            unique_yield = unique_count / max(st["valid"], 1)
            tail_yield = unique_yield
            info = {
                "pattern": p,
                "generated": int(st["generated"]),
                "valid": int(st["valid"]),
                "invalid": int(st["invalid"]),
                "valid_rate": valid_rate,
                "unique": unique_count,
                "unique_yield": unique_yield,
                "tail_unique_yield": tail_yield,
                "estimated_capacity": self._estimate_capacity(unique_count, tail_yield),
                "counter_span": unique_count,
            }
            self.per_pattern.append(info)
            if valid_rate >= 0.9 and unique_count > 0:
                self.accepted_patterns.append(p)
            else:
                self.rejected_patterns[p] = "non_productive_pattern"

        self.accepted_patterns = sorted(set(self.accepted_patterns))

        if not self.accepted_patterns and self.per_pattern:
            best = sorted(self.per_pattern, key=lambda x: (x["unique_yield"], x["valid_rate"]), reverse=True)[0]["pattern"]
            self.accepted_patterns = [best]
            self.rejected_patterns.pop(best, None)

        self.entries = []
        for p in self.accepted_patterns:
            for idx, domain in enumerate(self.pattern_domains.get(p, [])):
                self.entries.append(FosniwEntry(pattern=p, counter=idx, domain=domain))

        sample_unique = len({e.domain for e in self.entries})
        sample_generated = sum(int(x["valid"]) for x in self.per_pattern if x["pattern"] in self.accepted_patterns)
        sample_yield = sample_unique / max(sample_generated, 1)
        tails = [float(x["tail_unique_yield"]) for x in self.per_pattern if x["pattern"] in self.accepted_patterns]
        tail_med = median(tails) if tails else sample_yield
        est = min(len(self.entries), self._estimate_capacity(sample_unique, tail_med))

        return {
            "supported_patterns": self.supported_patterns,
            "requested_patterns": tested,
            "accepted_patterns": self.accepted_patterns,
            "rejected_patterns": self.rejected_patterns,
            "per_pattern": sorted(self.per_pattern, key=lambda x: x["pattern"]),
            "sample_generated": sample_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail_med,
            "estimated_max_unique_capacity": est,
            "structured_finite_combination": True,
            "recommended_near_capacity_ratio": 0.92,
            "failure_classification": {
                "unsupported_pattern": "pattern_not_in_supported_set",
                "invalid_pattern_schedule": "no_accepted_patterns_after_validation",
                "finite_capacity_exhaustion": "pattern_counter_space_consumed",
            },
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="fosniw_dedicated")

        if not self.accepted_patterns:
            return BatchGenerationResult(
                domains=[],
                attempts=1,
                generated=0,
                errors=["invalid_pattern_schedule"],
                adapter_type="fosniw_dedicated",
                last_effective_params={
                    "generation_mode": "pattern_counter_structured",
                    "supported_patterns": self.supported_patterns,
                    "requested_patterns": self.profile_data["requested_patterns"],
                    "accepted_patterns": [],
                    "rejected_patterns": self.rejected_patterns,
                    "finite_space": True,
                    "structured_space": True,
                    "failure_classification": self.profile_data["failure_classification"],
                },
                supported_parameter_axes=["pattern", "counter"],
            )

        start = self.next_global_index
        end = min(start + batch_size, self.estimated_max_unique_capacity)
        selected = self.entries[start:end]
        self.next_global_index = end

        pattern_usage: dict[str, int] = {}
        for e in selected:
            pattern_usage[e.pattern] = pattern_usage.get(e.pattern, 0) + 1

        remaining = max(0, self.estimated_max_unique_capacity - self.next_global_index)
        params = {
            "generation_mode": "pattern_counter_structured",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "supported_patterns": self.supported_patterns,
            "requested_patterns": self.profile_data["requested_patterns"],
            "accepted_patterns": self.accepted_patterns,
            "rejected_patterns": self.rejected_patterns,
            "selected_pattern_schedule": [e.pattern for e in selected[:32]],
            "pattern_usage": pattern_usage,
            "next_global_index": self.next_global_index,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining,
            "finite_space": True,
            "structured_space": True,
            "failure_classification": self.profile_data["failure_classification"],
        }

        return BatchGenerationResult(
            domains=[e.domain for e in selected],
            attempts=1,
            generated=len(selected),
            adapter_type="fosniw_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["pattern", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        invalid_total = sum(int(x["invalid"]) for x in self.per_pattern if x["pattern"] in self.accepted_patterns)
        valid_total = sum(int(x["valid"]) for x in self.per_pattern if x["pattern"] in self.accepted_patterns)
        invalid_rate = invalid_total / max(valid_total + invalid_total, 1)

        self._profile_cache = {
            "generates_any": bool(self.accepted_patterns and self.estimated_max_unique_capacity > 0),
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": False,
            "supported_parameter_axes": ["pattern", "counter"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "invalid_rate": invalid_rate,
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.5, max(0.05, self.profile_data["tail_unique_yield"] * 1.5)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.8,
            "apparent_finite_space": True,
            "generation_mode": "pattern_counter_structured",
            "adapter_type": "fosniw_dedicated",
            "supported_patterns": self.profile_data["supported_patterns"],
            "requested_patterns": self.profile_data["requested_patterns"],
            "accepted_patterns": self.profile_data["accepted_patterns"],
            "rejected_patterns": self.profile_data["rejected_patterns"],
            "per_pattern": self.profile_data["per_pattern"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "structured_finite_combination": True,
            "failure_classification": self.profile_data["failure_classification"],
        }
        return self._profile_cache
