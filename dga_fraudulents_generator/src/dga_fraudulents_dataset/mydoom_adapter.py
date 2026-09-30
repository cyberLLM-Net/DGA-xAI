from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class MydoomAdapter(AlgorithmAdapter):
    """Dedicated deterministic adapter for mydoom date+magic+number traversal."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        if not callable(self.dga_fn):
            raise RuntimeError("mydoom:missing_implementation:dga(date,magic,number)")

        self.base_date = self._resolve_base_date()
        self.day_window = max(365, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 3650))))
        self.number_span_requested = max(
            8, int(self.defaults.get("number_span", self.strategy_cfg.get("number_span", 50)))
        )
        self.include_post_reset_numbers = bool(
            self.defaults.get("include_post_reset_numbers", self.strategy_cfg.get("include_post_reset_numbers", False))
        )
        # In upstream mydoom, i==0x33 resets RNG with magic and collapses date-driven diversity.
        # Keeping number<=51 preserves high-yield date+number exploration.
        self.number_span = (
            self.number_span_requested if self.include_post_reset_numbers else min(self.number_span_requested, 50)
        )
        self.magic_values = self._resolve_magic_values()
        self.next_schedule_offset = self._resolve_resume_offset()
        self.profile_slots = max(256, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 8192))))

        self._sequence_cache: dict[tuple[str, int], list[str]] = {}
        self.emitted_unique: set[str] = set()
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"mydoom:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_mydoom", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"mydoom:invalid_resource_format:cannot_import:{path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

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

    @staticmethod
    def _parse_magic(value: Any) -> int:
        if isinstance(value, int):
            return value
        s = str(value).strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s)

    def _resolve_magic_values(self) -> list[int]:
        raw = self.defaults.get("magic_values", self.strategy_cfg.get("magic_values"))
        if raw is None:
            raw = self.defaults.get("magic", self.strategy_cfg.get("magic", "0xFA8"))
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        out: list[int] = []
        for v in values:
            try:
                out.append(self._parse_magic(v))
            except Exception:
                continue
        out = sorted(set(out))
        if not out:
            out = [0xFA8]
        return out

    def _slot_to_params(self, schedule_offset: int) -> tuple[datetime, int, int]:
        magic_count = max(1, len(self.magic_values))
        date_idx = schedule_offset // (magic_count * self.number_span)
        rem = schedule_offset % (magic_count * self.number_span)
        magic_idx = rem // self.number_span
        number_idx = rem % self.number_span
        date_value = self.base_date + timedelta(days=(date_idx % self.day_window))
        magic = self.magic_values[magic_idx % magic_count]
        # mydoom yields for i in range(1, number), so number>=2 emits at least one domain.
        number = number_idx + 2
        return date_value, magic, number

    def _sequence_for(self, date_value: datetime, magic: int) -> list[str]:
        key = (date_value.date().isoformat(), int(magic))
        cached = self._sequence_cache.get(key)
        if cached is not None:
            return cached
        seq: list[str] = []
        for item in self.dga_fn(date_value, int(magic), int(self.number_span + 2)):
            vr = validate_domain(str(item))
            if vr.is_valid and vr.normalized:
                seq.append(vr.normalized)
        self._sequence_cache[key] = seq
        if len(self._sequence_cache) > 32:
            # Keep bounded memory while still preserving locality.
            oldest = next(iter(self._sequence_cache.keys()))
            del self._sequence_cache[oldest]
        return seq

    def _domain_for(self, date_value: datetime, magic: int, number: int) -> str | None:
        seq = self._sequence_for(date_value, magic)
        idx = number - 2
        if idx < 0 or idx >= len(seq):
            return None
        return seq[idx]

    @staticmethod
    def _estimate_capacity(sample_unique: int, sample_generated: int, tail_yield: float, theoretical: int) -> int:
        if sample_generated <= 0:
            return 0
        # High-performing DGA: preserve strong quota while capping with observed tail behavior.
        if tail_yield > 0.95:
            factor = 3.2
        elif tail_yield > 0.9:
            factor = 2.6
        elif tail_yield > 0.8:
            factor = 2.0
        elif tail_yield > 0.6:
            factor = 1.6
        else:
            factor = 1.2
        observed = int(sample_unique * factor)
        return max(sample_unique, min(theoretical, observed))

    @staticmethod
    def _saturation_onset(curve: list[dict[str, Any]], fallback: int) -> int:
        for point in curve:
            if float(point.get("marginal_unique_yield", 1.0)) < 0.9:
                return int(point.get("sample_step", fallback))
        return fallback

    def _build_profile_data(self) -> dict[str, Any]:
        slots = self.profile_slots
        unique: set[str] = set()
        valid = 0
        invalid = 0
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(slots * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}
        seen_date: set[str] = set()
        seen_magic: set[int] = set()
        seen_number: set[int] = set()
        prev_valid = 0
        prev_unique = 0

        for i in range(slots):
            when, magic, number = self._slot_to_params(i)
            seen_date.add(when.date().isoformat())
            seen_magic.add(magic)
            seen_number.add(number)
            dom = self._domain_for(when, magic, number)
            if dom:
                valid += 1
                unique.add(dom)
            else:
                invalid += 1
            if (i + 1) in checkpoints:
                cur_unique = len(unique)
                step_valid = max(0, valid - prev_valid)
                step_unique = max(0, cur_unique - prev_unique)
                prev_valid = valid
                prev_unique = cur_unique
                curve.append(
                    {
                        "sample_step": i + 1,
                        "generated": valid,
                        "unique": cur_unique,
                        "collision_rate": 1.0 - (cur_unique / max(valid, 1)),
                        "marginal_unique_yield": step_unique / max(step_valid, 1),
                    }
                )

        unique_yield = len(unique) / max(valid, 1)
        if len(curve) >= 2:
            tail_yield = float(curve[-1]["marginal_unique_yield"])
        else:
            tail_yield = unique_yield

        date_probe = {self._domain_for(self.base_date + timedelta(days=i), self.magic_values[0], 2) for i in range(64)}
        magic_probe = {self._domain_for(self.base_date, m, 2) for m in self.magic_values}
        number_probe = {self._domain_for(self.base_date, self.magic_values[0], n + 2) for n in range(min(self.number_span, 128))}
        date_matters = len({x for x in date_probe if x}) > 1
        magic_matters = len({x for x in magic_probe if x}) > 1 if len(self.magic_values) > 1 else True
        number_matters = len({x for x in number_probe if x}) > 1

        theoretical = self.day_window * len(self.magic_values) * self.number_span
        estimated_cap = self._estimate_capacity(len(unique), valid, tail_yield, theoretical)
        saturation_onset = self._saturation_onset(curve, fallback=max(1, int(valid * 0.9)))
        return {
            "sample_generated": valid,
            "sample_invalid": invalid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "date_explored_count": len(seen_date),
            "magic_explored": sorted(seen_magic),
            "number_explored_count": len(seen_number),
            "date_matters": date_matters,
            "magic_matters": magic_matters,
            "number_matters": number_matters,
            "observed_unique_capacity_estimate": max(len(unique), int(len(unique) * 1.25)),
            "estimated_max_unique_capacity": estimated_cap,
            "estimated_saturation_onset": saturation_onset,
            "collision_prone": unique_yield < 0.9 or tail_yield < 0.85,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="mydoom_dedicated")

        out: list[str] = []
        attempts = 0
        start_offset = self.next_schedule_offset
        used_magic: set[int] = set()
        used_numbers: set[int] = set()
        used_dates: set[str] = set()

        while len(out) < batch_size:
            when, magic, number = self._slot_to_params(self.next_schedule_offset)
            self.next_schedule_offset += 1
            attempts += 1
            used_magic.add(magic)
            used_numbers.add(number)
            used_dates.add(when.date().isoformat())

            dom = self._domain_for(when, magic, number)
            if dom:
                out.append(dom)
            if attempts > batch_size * 6:
                break

        out = out[:batch_size]
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "date_magic_number_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "magic", "number"],
            "start_schedule_offset": start_offset,
            "next_schedule_offset": self.next_schedule_offset,
            "day_window": self.day_window,
            "number_span": self.number_span,
            "number_span_requested": self.number_span_requested,
            "include_post_reset_numbers": self.include_post_reset_numbers,
            "magic_values": [hex(m) for m in self.magic_values],
            "date_explored_count": len(used_dates),
            "magic_explored": [hex(m) for m in sorted(used_magic)],
            "number_explored_count": len(used_numbers),
            "collision_growth_curve": self.profile_data["collision_curve"],
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "observed_unique_capacity_estimate": self.profile_data["observed_unique_capacity_estimate"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "redistribution_absorption_reason": "high_unique_yield_with_multi_axis_schedule",
        }

        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="mydoom_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "magic", "number"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": bool(self.profile_data["magic_matters"]),
            "date_sensitive": bool(self.profile_data["date_matters"]),
            "counter_sensitive": bool(self.profile_data["number_matters"]),
            "supported_parameter_axes": ["date", "magic", "number"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.2, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.4, max(0.2, self.profile_data["tail_unique_yield"] * 2.2)),
            "expected_diversity_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 2.5,
            "recommended_saturation_sensitivity": 0.65,
            "recommended_near_capacity_ratio": 0.96,
            "apparent_finite_space": False,
            "adapter_type": "mydoom_dedicated",
            "generation_mode": "date_magic_number_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "magic", "number"],
            "number_span_requested": self.number_span_requested,
            "include_post_reset_numbers": self.include_post_reset_numbers,
            "collision_curve": self.profile_data["collision_curve"],
            "collision_growth_summary": {
                "sample_unique_yield": self.profile_data["sample_unique_yield"],
                "tail_unique_yield": self.profile_data["tail_unique_yield"],
            },
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "observed_unique_capacity_estimate": self.profile_data["observed_unique_capacity_estimate"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "date_explored_count": self.profile_data["date_explored_count"],
            "magic_explored": [hex(m) for m in self.profile_data["magic_explored"]],
            "number_explored_count": self.profile_data["number_explored_count"],
            "collision_prone": self.profile_data["collision_prone"],
            "redistribution_absorption_reason": "high_unique_yield_with_multi_axis_schedule",
        }
        self._profile_cache = p
        return p
