from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class ChinadAdapter(AlgorithmAdapter):
    """Dedicated adapter for chinad (date + internal nr traversal)."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        if not callable(self.dga_fn):
            raise RuntimeError("chinad:missing_implementation:dga(date)")

        self.base_date = self._resolve_base_date()
        self.day_window = max(365, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 3650))))
        self.per_day_domains = int(self.defaults.get("per_day_domains", self.strategy_cfg.get("per_day_domains", 256)))
        self.per_day_domains = max(64, min(256, self.per_day_domains))
        self.next_schedule_offset = self._resolve_resume_offset()
        self.profile_slots = max(512, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 8192))))

        self._daily_cache: dict[str, list[str]] = {}
        self.emitted_unique: set[str] = set()
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"chinad:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_chinad", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"chinad:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_resume_offset(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_schedule_offset")
            or self.strategy_cfg.get("next_schedule_offset")
            or self.defaults.get("resume_next_schedule_offset")
            or self.defaults.get("next_schedule_offset")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _slot_to_params(self, schedule_offset: int) -> tuple[datetime, int]:
        day_index = schedule_offset // self.per_day_domains
        nr = schedule_offset % self.per_day_domains
        when = self.base_date + timedelta(days=(day_index % self.day_window))
        return when, nr

    def _domains_for_date(self, when: datetime) -> list[str]:
        key = when.date().isoformat()
        cached = self._daily_cache.get(key)
        if cached is not None:
            return cached
        out: list[str] = []
        for i, d in enumerate(self.dga_fn(when)):
            if i >= self.per_day_domains:
                break
            vr = validate_domain(str(d))
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
        self._daily_cache[key] = out
        if len(self._daily_cache) > 64:
            oldest = next(iter(self._daily_cache.keys()))
            del self._daily_cache[oldest]
        return out

    @staticmethod
    def _estimate_capacity(sample_unique: int, sample_generated: int, tail_yield: float, theoretical: int) -> int:
        if sample_generated <= 0:
            return 0
        if tail_yield > 0.98:
            factor = 3.0
        elif tail_yield > 0.92:
            factor = 2.5
        elif tail_yield > 0.85:
            factor = 2.0
        else:
            factor = 1.5
        observed = int(sample_unique * factor)
        return max(sample_unique, min(theoretical, observed))

    @staticmethod
    def _saturation_onset(curve: list[dict[str, Any]], fallback: int) -> int:
        for point in curve:
            if float(point.get("marginal_unique_yield", 1.0)) < 0.9:
                return int(point.get("sample_step", fallback))
        return fallback

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        invalid = 0
        tlds: dict[str, int] = {}
        seen_dates: set[str] = set()
        seen_nr: set[int] = set()
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(self.profile_slots * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}
        prev_generated = 0
        prev_unique = 0

        for i in range(self.profile_slots):
            when, nr = self._slot_to_params(i)
            seen_dates.add(when.date().isoformat())
            seen_nr.add(nr)
            daily = self._domains_for_date(when)
            dom = daily[nr] if nr < len(daily) else None
            if dom is None:
                invalid += 1
            else:
                generated += 1
                unique.add(dom)
                t = "." + dom.rsplit(".", 1)[-1]
                tlds[t] = tlds.get(t, 0) + 1

            if (i + 1) in checkpoints:
                cur_unique = len(unique)
                step_gen = max(0, generated - prev_generated)
                step_uni = max(0, cur_unique - prev_unique)
                prev_generated = generated
                prev_unique = cur_unique
                curve.append(
                    {
                        "sample_step": i + 1,
                        "generated": generated,
                        "unique": cur_unique,
                        "collision_rate": 1.0 - (cur_unique / max(generated, 1)),
                        "marginal_unique_yield": step_uni / max(step_gen, 1),
                    }
                )

        unique_yield = len(unique) / max(generated, 1)
        tail_yield = float(curve[-1]["marginal_unique_yield"]) if curve else unique_yield
        theoretical = self.day_window * self.per_day_domains
        estimated_cap = self._estimate_capacity(len(unique), generated, tail_yield, theoretical)
        sat_onset = self._saturation_onset(curve, fallback=max(1, int(generated * 0.95)))
        return {
            "sample_generated": generated,
            "sample_invalid": invalid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "collision_rate": 1.0 - unique_yield,
            "date_explored_count": len(seen_dates),
            "nr_explored_count": len(seen_nr),
            "tld_distribution": tlds,
            "observed_unique_capacity_estimate": max(len(unique), int(len(unique) * 1.2)),
            "estimated_max_unique_capacity": estimated_cap,
            "estimated_saturation_onset": sat_onset,
            "collision_prone": unique_yield < 0.95 or tail_yield < 0.9,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="chinad_dedicated")

        domains: list[str] = []
        attempts = 0
        start_offset = self.next_schedule_offset
        used_dates: set[str] = set()
        used_nr: set[int] = set()
        tlds: dict[str, int] = {}
        local_seen: set[str] = set()

        while len(domains) < batch_size:
            when, nr = self._slot_to_params(self.next_schedule_offset)
            self.next_schedule_offset += 1
            attempts += 1
            used_dates.add(when.date().isoformat())
            used_nr.add(nr)
            daily = self._domains_for_date(when)
            dom = daily[nr] if nr < len(daily) else None
            if dom:
                domains.append(dom)
                local_seen.add(dom)
                t = "." + dom.rsplit(".", 1)[-1]
                tlds[t] = tlds.get(t, 0) + 1
            if attempts > batch_size * 4:
                break

        domains = domains[:batch_size]
        self.emitted_unique.update(domains)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))
        batch_collision_rate = 1.0 - (len(local_seen) / max(len(domains), 1))

        eff = {
            "generation_mode": "date_nr_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "nr"],
            "start_schedule_offset": start_offset,
            "next_schedule_offset": self.next_schedule_offset,
            "day_window": self.day_window,
            "per_day_domains": self.per_day_domains,
            "date_explored_count": len(used_dates),
            "nr_explored_count": len(used_nr),
            "tld_distribution": tlds,
            "collision_rate": batch_collision_rate,
            "collision_growth_summary": {
                "sample_unique_yield": self.profile_data["sample_unique_yield"],
                "tail_unique_yield": self.profile_data["tail_unique_yield"],
            },
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "observed_unique_capacity_estimate": self.profile_data["observed_unique_capacity_estimate"],
            "estimated_capacity": self.estimated_max_unique_capacity,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "redistribution_absorption_reason": "high_unique_yield_with_stable_date_nr_traversal",
        }
        return BatchGenerationResult(
            domains=domains,
            attempts=attempts,
            generated=len(domains),
            adapter_type="chinad_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "nr"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        self._profile_cache = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "counter_sensitive": True,
            "supported_parameter_axes": ["date", "nr"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.2, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.5, max(0.2, self.profile_data["tail_unique_yield"] * 2.3)),
            "expected_diversity_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 3.0,
            "recommended_saturation_sensitivity": 0.72,
            "recommended_near_capacity_ratio": 0.97,
            "apparent_finite_space": False,
            "adapter_type": "chinad_dedicated",
            "generation_mode": "date_nr_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "nr"],
            "tld_distribution": self.profile_data["tld_distribution"],
            "collision_curve": self.profile_data["collision_curve"],
            "collision_rate": self.profile_data["collision_rate"],
            "collision_growth_summary": {
                "sample_unique_yield": self.profile_data["sample_unique_yield"],
                "tail_unique_yield": self.profile_data["tail_unique_yield"],
            },
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "observed_unique_capacity_estimate": self.profile_data["observed_unique_capacity_estimate"],
            "estimated_capacity": self.profile_data["estimated_max_unique_capacity"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "redistribution_absorption_reason": "high_unique_yield_with_stable_date_nr_traversal",
            "hidden_mode_detected": False,
        }
        return self._profile_cache
