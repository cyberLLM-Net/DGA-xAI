from __future__ import annotations

import importlib.util
import inspect
import io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class BazarBackdoorAdapter(AlgorithmAdapter):
    """Version-aware, month-seeded adapter for bazarbackdoor.

    The algorithm is seeded by month+year, so day-by-day traversal causes
    repetitive output. This adapter explores month buckets explicitly and
    profiles each supported version independently.
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
            raise RuntimeError("bazarbackdoor dga.py missing callable dga(date, version)")
        self.sig = inspect.signature(self.dga_fn)

        self.base_date = self._resolve_base_date()
        self.month_window = max(6, int(self.defaults.get("month_window", self.strategy_cfg.get("month_window", 72))))
        self.profile_months = max(4, int(self.defaults.get("profile_months", self.strategy_cfg.get("profile_months", 24))))
        self.next_global_index = self._resolve_resume_global_index()

        self.supported_versions = sorted(self._discover_supported_versions())
        self.requested_versions = self._resolve_requested_versions()
        self.tested_versions = sorted(set(self.requested_versions))
        self.rejected_versions: dict[str, str] = {}
        self.accepted_versions: list[str] = []
        self.version_stats: dict[str, dict[str, Any]] = {}
        self.version_weights: dict[str, float] = {}
        self._domain_cache: dict[tuple[str, int], tuple[list[str], dict[str, int]]] = {}
        self.emitted_unique: set[str] = set()

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"bazarbackdoor missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_bazarbackdoor", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import bazarbackdoor module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

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

    def _discover_supported_versions(self) -> set[str]:
        versions = getattr(self.module, "versions", None)
        if isinstance(versions, dict):
            return {str(k) for k in versions.keys()}
        return {"v2", "v3", "v4", "v5", "v6", "v7"}

    def _resolve_requested_versions(self) -> list[str]:
        raw = self.defaults.get("bazar_versions", self.strategy_cfg.get("bazar_versions", "all"))
        if isinstance(raw, str):
            txt = raw.strip().lower()
            if txt in {"all", "*"}:
                return list(self.supported_versions)
            return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple, set)):
            return [str(x).strip() for x in raw if str(x).strip()]
        return list(self.supported_versions)

    @staticmethod
    def _add_months(dt: datetime, months: int) -> datetime:
        year = dt.year + ((dt.month - 1 + months) // 12)
        month = ((dt.month - 1 + months) % 12) + 1
        day = min(dt.day, 28)
        return dt.replace(year=year, month=month, day=day)

    def _month_offset_for_slot(self, slot: int) -> int:
        # diversified month traversal to avoid correlated contiguous months
        return ((slot * 11) + 5) % self.month_window

    def _date_for_month_offset(self, month_offset: int) -> datetime:
        return self._add_months(self.base_date, -month_offset)

    def _domains_for_version_month(self, version: str, month_offset: int) -> tuple[list[str], dict[str, int]]:
        key = (version, month_offset)
        cached = self._domain_cache.get(key)
        if cached is not None:
            return cached

        when = self._date_for_month_offset(month_offset)
        sink = io.StringIO()
        with redirect_stdout(sink):
            raw_out = list(self.dga_fn(when, version))

        valid_domains: list[str] = []
        invalid = 0
        for d in raw_out:
            vr = validate_domain(str(d))
            if vr.is_valid and vr.normalized:
                valid_domains.append(vr.normalized)
            else:
                invalid += 1

        stats = {
            "generated_raw": len(raw_out),
            "valid": len(valid_domains),
            "invalid": invalid,
        }
        result = (valid_domains, stats)
        self._domain_cache[key] = result
        return result

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.08
        elif tail_yield <= 0.08:
            factor = 1.18
        elif tail_yield <= 0.2:
            factor = 1.35
        elif tail_yield <= 0.35:
            factor = 1.55
        else:
            factor = 1.85
        return max(sample_unique, int(sample_unique * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        # Reject unsupported versions up-front.
        for v in self.tested_versions:
            if v not in self.supported_versions:
                self.rejected_versions[v] = "unsupported_version"

        per_version: list[dict[str, Any]] = []
        all_unique: set[str] = set()
        all_generated = 0

        for version in self.tested_versions:
            if version in self.rejected_versions:
                continue
            unique: set[str] = set()
            generated = 0
            valid = 0
            curve: list[dict[str, Any]] = []

            for i in range(self.profile_months):
                mo = self._month_offset_for_slot(i)
                domains, st = self._domains_for_version_month(version, mo)
                generated += int(st["generated_raw"])
                valid += int(st["valid"])
                all_generated += int(st["valid"])
                unique.update(domains)
                all_unique.update(domains)
                curve.append(
                    {
                        "month_samples": i + 1,
                        "generated_valid": valid,
                        "unique": len(unique),
                        "collision_rate": 1.0 - (len(unique) / max(valid, 1)),
                    }
                )

            valid_rate = valid / max(generated, 1)
            unique_yield = len(unique) / max(valid, 1)
            if len(curve) >= 2:
                prev = curve[-2]
                last = curve[-1]
                tail_gen = max(1, last["generated_valid"] - prev["generated_valid"])
                tail_uni = max(0, last["unique"] - prev["unique"])
                tail_yield = tail_uni / tail_gen
            else:
                tail_yield = unique_yield

            info = {
                "version": version,
                "sample_months": self.profile_months,
                "generated_raw": generated,
                "valid": valid,
                "invalid": max(0, generated - valid),
                "valid_rate": valid_rate,
                "unique": len(unique),
                "unique_yield": unique_yield,
                "tail_unique_yield": tail_yield,
                "collision_curve": curve,
            }
            per_version.append(info)
            self.version_stats[version] = info

            if valid_rate < 0.8:
                self.rejected_versions[version] = "low_valid_rate"
            elif unique_yield < 0.08:
                self.rejected_versions[version] = "low_unique_yield"
            else:
                self.accepted_versions.append(version)

        self.accepted_versions = sorted(set(self.accepted_versions))
        if not self.accepted_versions and per_version:
            # keep best fallback if all versions are below threshold
            best = sorted(per_version, key=lambda x: (x["unique_yield"] * x["valid_rate"]), reverse=True)[0]["version"]
            self.accepted_versions = [best]
            self.rejected_versions.pop(best, None)

        for v in self.accepted_versions:
            st = self.version_stats[v]
            self.version_weights[v] = max(0.01, st["unique_yield"] * st["valid_rate"])

        sample_unique = len(all_unique)
        sample_yield = sample_unique / max(all_generated, 1)
        tail = median([self.version_stats[v]["tail_unique_yield"] for v in self.accepted_versions]) if self.accepted_versions else sample_yield
        estimated = self._estimate_capacity(sample_unique, tail)

        return {
            "supported_versions": sorted(self.supported_versions),
            "tested_versions": sorted(self.tested_versions),
            "accepted_versions": self.accepted_versions,
            "rejected_versions": self.rejected_versions,
            "per_version": sorted(per_version, key=lambda x: x["version"]),
            "sample_generated": all_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail,
            "estimated_max_unique_capacity": estimated,
            "structured_finite_combination": True,
            "recommended_near_capacity_ratio": 0.9,
            "reverse_mapping_tool": "domain_to_seed.py (generation disabled)",
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="bazarbackdoor_dedicated")
        if not self.accepted_versions:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="bazarbackdoor_dedicated")

        out: list[str] = []
        start_index = self.next_global_index
        attempts = 0
        usage: dict[str, int] = {v: 0 for v in self.accepted_versions}
        month_offsets_used: list[int] = []

        total_weight = sum(self.version_weights.get(v, 1.0) for v in self.accepted_versions)
        if total_weight <= 0:
            total_weight = float(len(self.accepted_versions))

        while len(out) < batch_size:
            for version in self.accepted_versions:
                if len(out) >= batch_size:
                    break
                w = self.version_weights.get(version, 1.0) / total_weight
                take = max(1, int((batch_size * w) / max(len(self.accepted_versions), 1)))
                take = min(take, batch_size - len(out))

                slot = self.next_global_index + usage[version]
                month_offset = self._month_offset_for_slot(slot)
                month_offsets_used.append(month_offset)
                domains, _ = self._domains_for_version_month(version, month_offset)
                if not domains:
                    usage[version] += 1
                    attempts += 1
                    continue
                dlen = len(domains)
                start = (slot * 17) % dlen
                for i in range(take):
                    out.append(domains[(start + i * 5) % dlen])
                usage[version] += 1
                attempts += 1
                if attempts > max(batch_size * 4, 128):
                    break
            if attempts > max(batch_size * 4, 128):
                break

        out = out[:batch_size]
        self.next_global_index += max(1, sum(usage.values()))
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "version_month_structured",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "supported_versions": sorted(self.supported_versions),
            "tested_versions": sorted(self.tested_versions),
            "accepted_versions": self.accepted_versions,
            "rejected_versions": self.rejected_versions,
            "version_weights": self.version_weights,
            "version_usage_slots": usage,
            "start_global_index": start_index,
            "next_global_index": self.next_global_index,
            "month_window": self.month_window,
            "month_offsets_used": month_offsets_used[:32],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "structured_space": True,
            "finite_space": True,
            "reverse_mapping_tool": "domain_to_seed.py",
        }
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="bazarbackdoor_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["version", "month_offset", "date"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": bool(self.profile_data["sample_generated"] > 0 and self.profile_data["accepted_versions"]),
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "supported_parameter_axes": ["version", "month_offset", "date"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.4, max(0.08, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": True,
            "generation_mode": "version_month_structured",
            "adapter_type": "bazarbackdoor_dedicated",
            "structured_finite_combination": True,
            "supported_versions": self.profile_data["supported_versions"],
            "tested_versions": self.profile_data["tested_versions"],
            "accepted_versions": self.profile_data["accepted_versions"],
            "rejected_versions": self.profile_data["rejected_versions"],
            "per_version": self.profile_data["per_version"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "reverse_mapping_tool": self.profile_data["reverse_mapping_tool"],
        }
        self._profile_cache = p
        return p
