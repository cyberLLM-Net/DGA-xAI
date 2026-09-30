from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from math import gcd
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class NewgozAdapter(AlgorithmAdapter):
    """Dedicated adapter for newgoz (date + sequence/seed-space exploration)."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.create_domain = getattr(self.module, "create_domain", None)
        self.get_seed = getattr(self.module, "get_seed", None)
        if not callable(self.create_domain):
            raise RuntimeError("newgoz:missing_implementation:create_domain(seq_nr,date)")

        self.base_date = self._resolve_base_date()
        self.day_window = max(365, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 3650))))
        self.seq_span = max(128, int(self.defaults.get("seq_span", self.strategy_cfg.get("seq_span", 1000))))
        self.seq_stride = self._resolve_seq_stride()
        self.seed_offsets = self._resolve_seed_offsets()
        self.next_schedule_offset = self._resolve_resume_offset()
        self.profile_slots = max(512, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 8192))))

        self.emitted_unique: set[str] = set()
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"newgoz:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_newgoz", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"newgoz:invalid_resource_format:cannot_import:{path}")
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

    def _resolve_seq_stride(self) -> int:
        raw = int(self.defaults.get("seq_stride", self.strategy_cfg.get("seq_stride", 137)))
        stride = max(1, raw)
        if gcd(stride, self.seq_span) == 1:
            return stride
        for candidate in range(stride + 1, stride + self.seq_span + 1):
            if gcd(candidate, self.seq_span) == 1:
                return candidate
        return 1

    def _resolve_seed_offsets(self) -> list[int]:
        raw = self.defaults.get("seed_offsets", self.strategy_cfg.get("seed_offsets", [0, 251, 503, 757]))
        items = raw if isinstance(raw, (list, tuple, set)) else [raw]
        offsets: list[int] = []
        for item in items:
            try:
                offsets.append(int(item) % self.seq_span)
            except Exception:
                continue
        offsets = sorted(set(offsets))
        if not offsets:
            offsets = [0]
        return offsets

    def _slot_to_params(self, schedule_offset: int) -> tuple[datetime, int, int]:
        seed_count = max(1, len(self.seed_offsets))
        date_idx = schedule_offset // self.seq_span
        seq_idx = schedule_offset % self.seq_span
        virtual_seed_index = date_idx % seed_count
        date_value = self.base_date + timedelta(days=(date_idx % self.day_window))
        seq_nr = (seq_idx * self.seq_stride + self.seed_offsets[virtual_seed_index]) % self.seq_span
        return date_value, seq_nr, virtual_seed_index

    @staticmethod
    def _tld(domain: str) -> str:
        if "." not in domain:
            return ""
        return "." + domain.rsplit(".", 1)[-1]

    @staticmethod
    def _estimate_capacity(sample_unique: int, sample_generated: int, tail_yield: float, theoretical: int) -> int:
        if sample_generated <= 0:
            return 0
        if tail_yield > 0.9:
            factor = 2.6
        elif tail_yield > 0.8:
            factor = 2.1
        elif tail_yield > 0.65:
            factor = 1.7
        else:
            factor = 1.3
        observed = int(sample_unique * factor)
        return max(sample_unique, min(theoretical, observed))

    @staticmethod
    def _saturation_onset(curve: list[dict[str, Any]], fallback: int) -> int:
        for p in curve:
            if float(p.get("marginal_unique_yield", 1.0)) < 0.75:
                return int(p.get("sample_step", fallback))
        return fallback

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        invalid = 0
        seen_dates: set[str] = set()
        seen_seed_indices: set[int] = set()
        seen_seq: set[int] = set()
        tld_counts: dict[str, int] = {}
        label_len_counts: dict[int, int] = {}
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(self.profile_slots * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}
        prev_generated = 0
        prev_unique = 0

        for i in range(self.profile_slots):
            when, seq_nr, seed_idx = self._slot_to_params(i)
            seen_dates.add(when.date().isoformat())
            seen_seed_indices.add(seed_idx)
            seen_seq.add(seq_nr)
            out = self.create_domain(int(seq_nr), when)
            vr = validate_domain(str(out))
            if not vr.is_valid or not vr.normalized:
                invalid += 1
            else:
                generated += 1
                dom = vr.normalized
                unique.add(dom)
                t = self._tld(dom)
                tld_counts[t] = tld_counts.get(t, 0) + 1
                label = dom.split(".", 1)[0]
                label_len_counts[len(label)] = label_len_counts.get(len(label), 0) + 1

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

        sample_yield = len(unique) / max(generated, 1)
        tail_yield = float(curve[-1]["marginal_unique_yield"]) if curve else sample_yield
        theoretical = self.day_window * self.seq_span * len(self.seed_offsets)
        estimated_cap = self._estimate_capacity(len(unique), generated, tail_yield, theoretical)
        saturation_onset = self._saturation_onset(curve, fallback=max(1, int(generated * 0.9)))
        # newgoz is hash/alphanumeric based (not lexical dictionary). Keep the requested
        # diagnostics field while exposing the actual symbol-space size.
        dictionary_size = 36
        word_count_distribution = {"4": generated} if generated else {}

        return {
            "sample_generated": generated,
            "sample_invalid": invalid,
            "sample_unique": len(unique),
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "collision_rate": 1.0 - sample_yield,
            "date_explored_count": len(seen_dates),
            "seed_axis_explored_count": len(seen_seed_indices),
            "seq_explored_count": len(seen_seq),
            "tld_distribution": tld_counts,
            "label_length_distribution": {str(k): v for k, v in sorted(label_len_counts.items())},
            "dictionary_size": dictionary_size,
            "word_count_distribution": word_count_distribution,
            "observed_unique_capacity_estimate": max(len(unique), int(len(unique) * 1.3)),
            "estimated_max_unique_capacity": estimated_cap,
            "estimated_saturation_onset": saturation_onset,
            "collision_prone": sample_yield < 0.78 or tail_yield < 0.65,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="newgoz_dedicated")

        domains: list[str] = []
        attempts = 0
        start_offset = self.next_schedule_offset
        used_dates: set[str] = set()
        used_seed_indices: set[int] = set()
        used_seq: set[int] = set()
        local_seen: set[str] = set()

        while len(domains) < batch_size:
            when, seq_nr, seed_idx = self._slot_to_params(self.next_schedule_offset)
            self.next_schedule_offset += 1
            attempts += 1
            used_dates.add(when.date().isoformat())
            used_seed_indices.add(seed_idx)
            used_seq.add(seq_nr)

            out = self.create_domain(int(seq_nr), when)
            vr = validate_domain(str(out))
            if vr.is_valid and vr.normalized:
                dom = vr.normalized
                domains.append(dom)
                local_seen.add(dom)
            if attempts > batch_size * 5:
                break

        domains = domains[:batch_size]
        self.emitted_unique.update(domains)
        batch_collision_rate = 1.0 - (len(local_seen) / max(len(domains), 1))
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "date_seed_seq_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "seed", "seq_nr"],
            "start_schedule_offset": start_offset,
            "next_schedule_offset": self.next_schedule_offset,
            "day_window": self.day_window,
            "seq_span": self.seq_span,
            "seq_stride": self.seq_stride,
            "seed_offsets": self.seed_offsets,
            "date_explored_count": len(used_dates),
            "seed_axis_explored_count": len(used_seed_indices),
            "seq_explored_count": len(used_seq),
            "dictionary_size": self.profile_data["dictionary_size"],
            "word_count_distribution": self.profile_data["word_count_distribution"],
            "collision_rate": batch_collision_rate,
            "collision_growth_curve": self.profile_data["collision_curve"],
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "estimated_capacity": self.estimated_max_unique_capacity,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "redistribution_absorption_reason": "high_diversity_with_structured_seed_date_traversal",
        }
        return BatchGenerationResult(
            domains=domains,
            attempts=attempts,
            generated=len(domains),
            adapter_type="newgoz_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "seed", "seq_nr"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        self._profile_cache = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": True,
            "counter_sensitive": True,
            "supported_parameter_axes": ["date", "seed", "seq_nr"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.15, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.2, max(0.2, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 2.0,
            "recommended_saturation_sensitivity": 0.6,
            "recommended_near_capacity_ratio": 0.95,
            "apparent_finite_space": False,
            "adapter_type": "newgoz_dedicated",
            "generation_mode": "date_seed_seq_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "dictionary_size": self.profile_data["dictionary_size"],
            "word_count_distribution": self.profile_data["word_count_distribution"],
            "tld_distribution": self.profile_data["tld_distribution"],
            "label_length_distribution": self.profile_data["label_length_distribution"],
            "collision_curve": self.profile_data["collision_curve"],
            "collision_rate": self.profile_data["collision_rate"],
            "collision_prone": self.profile_data["collision_prone"],
            "estimated_saturation_onset": self.profile_data["estimated_saturation_onset"],
            "observed_unique_capacity_estimate": self.profile_data["observed_unique_capacity_estimate"],
            "estimated_capacity": self.profile_data["estimated_max_unique_capacity"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "redistribution_absorption_reason": "high_diversity_with_structured_seed_date_traversal",
        }
        return self._profile_cache
