from __future__ import annotations

import importlib.util
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection


class MoneroDownloaderAdapter(AlgorithmAdapter):
    """Structured finite-combination adapter for monerodownloader.

    The upstream algorithm yields domains from:
      base_label(date, nr) x fixed_tlds
    and supports a `back` argument that replays previous days. For large-scale
    generation this adapter keeps `back=0` and explores day/index deterministically
    to avoid replay amplification.
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
            raise RuntimeError("monerodownloader dga.py missing callable dga(date, back=0)")
        self.sig = inspect.signature(self.dga_fn)

        self.base_date = self._resolve_base_date()
        self.max_day_offsets = max(
            1,
            int(self.defaults.get("max_day_offsets", self.strategy_cfg.get("max_day_offsets", 256))),
        )
        self.profile_days = max(
            2,
            int(self.defaults.get("profile_days", self.strategy_cfg.get("profile_days", min(self.max_day_offsets, 64)))),
        )
        self.next_domain_index = self._resolve_resume_domain_index()
        self.day_cache: dict[int, list[str]] = {}
        self.emitted_unique: set[str] = set()

        # Discover structure from day 0.
        day0 = self._domains_for_day(0)
        if not day0:
            raise RuntimeError("monerodownloader returned no domains for day 0")
        self.domains_per_day = len(day0)
        self.tld_variants = sorted({self._tld_of(d) for d in day0})

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"monerodownloader missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_monerodownloader", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import monerodownloader module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_resume_domain_index(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_domain_index")
            or self.strategy_cfg.get("next_domain_index")
            or self.defaults.get("resume_next_domain_index")
            or self.defaults.get("next_domain_index")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    @staticmethod
    def _base_label(domain: str) -> str:
        return domain.split(".", 1)[0] if "." in domain else domain

    @staticmethod
    def _tld_of(domain: str) -> str:
        if "." not in domain:
            return ""
        return "." + domain.split(".", 1)[1]

    def _date_for_day_offset(self, day_offset: int) -> datetime:
        return self.base_date - timedelta(days=day_offset)

    def _domains_for_day(self, day_offset: int) -> list[str]:
        if day_offset < 0 or day_offset >= self.max_day_offsets:
            return []
        cached = self.day_cache.get(day_offset)
        if cached is not None:
            return cached
        when = self._date_for_day_offset(day_offset)
        domains = [str(x) for x in self.dga_fn(when, 0)]
        self.day_cache[day_offset] = domains
        return domains

    @staticmethod
    def _estimate_capacity(day0_unique: int, day_gain: int, day_count: int) -> int:
        if day_count <= 0:
            return 0
        if day_count == 1:
            return day0_unique
        return max(day0_unique, day0_unique + max(0, day_count - 1) * max(0, day_gain))

    def _build_profile_data(self) -> dict[str, Any]:
        sampled_days = min(self.profile_days, self.max_day_offsets)

        day_unique_counts: list[int] = []
        day_base_counts: list[int] = []
        sampled_tlds: set[str] = set()
        global_unique: set[str] = set()
        global_bases: set[str] = set()
        curve: list[dict[str, Any]] = []

        day0 = self._domains_for_day(0)
        day1 = self._domains_for_day(1) if sampled_days > 1 else []
        day0_set = set(day0)
        day1_set = set(day1)
        day_increment = len(day1_set - day0_set) if day1 else len(day0_set)

        day0_bases = {self._base_label(d) for d in day0}
        day1_bases = {self._base_label(d) for d in day1} if day1 else set()
        day_base_increment = len(day1_bases - day0_bases) if day1 else len(day0_bases)

        for i in range(sampled_days):
            domains = self._domains_for_day(i)
            dset = set(domains)
            bases = {self._base_label(d) for d in domains}
            day_unique_counts.append(len(dset))
            day_base_counts.append(len(bases))
            sampled_tlds.update({self._tld_of(d) for d in domains})
            global_unique.update(dset)
            global_bases.update(bases)
            generated = (i + 1) * max(1, self.domains_per_day)
            unique = len(global_unique)
            curve.append(
                {
                    "day_samples": i + 1,
                    "generated": generated,
                    "unique": unique,
                    "collision_rate": 1.0 - (unique / max(generated, 1)),
                }
            )

        sample_generated = sampled_days * max(1, self.domains_per_day)
        sample_unique = len(global_unique)
        sample_unique_yield = sample_unique / max(sample_generated, 1)

        day0_unique = len(day0_set)
        day0_base_count = len(day0_bases)
        est_total = self._estimate_capacity(day0_unique, day_increment, self.max_day_offsets)
        est_base_labels = self._estimate_capacity(day0_base_count, day_base_increment, self.max_day_offsets)
        est_total = min(est_total, est_base_labels * max(1, len(sampled_tlds or self.tld_variants)))

        tail_yield = sample_unique_yield
        if len(curve) >= 2:
            prev = curve[-2]
            last = curve[-1]
            tail_gen = max(1, last["generated"] - prev["generated"])
            tail_uni = max(0, last["unique"] - prev["unique"])
            tail_yield = tail_uni / tail_gen

        return {
            "sample_days": sampled_days,
            "domains_per_day": self.domains_per_day,
            "sample_generated": sample_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_unique_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "tld_variants": sorted(sampled_tlds) if sampled_tlds else self.tld_variants,
            "day_unique_median": int(median(day_unique_counts)) if day_unique_counts else 0,
            "day_base_median": int(median(day_base_counts)) if day_base_counts else 0,
            "day_increment_unique": day_increment,
            "day_increment_base_labels": day_base_increment,
            "distinct_base_labels_estimate": est_base_labels,
            "estimated_total_combination_space": est_base_labels * max(1, len(sampled_tlds or self.tld_variants)),
            "estimated_max_unique_capacity": est_total,
            "structured_finite_combination": True,
            "recommended_near_capacity_ratio": 0.92,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="monerodownloader_dedicated")

        out: list[str] = []
        start_index = self.next_domain_index
        attempts = 0

        while len(out) < batch_size:
            absolute_idx = self.next_domain_index
            day_offset = absolute_idx // self.domains_per_day
            within_day = absolute_idx % self.domains_per_day
            day_domains = self._domains_for_day(day_offset)
            if not day_domains:
                break
            take = min(batch_size - len(out), len(day_domains) - within_day)
            if take <= 0:
                break
            out.extend(day_domains[within_day : within_day + take])
            self.next_domain_index += take
            attempts += 1

        self.emitted_unique.update(out)
        remaining_practical = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "structured_date_index",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "day_offset", "nr", "tld_variant"],
            "back_parameter_mode": "fixed_zero_to_avoid_replay",
            "start_domain_index": start_index,
            "next_domain_index": self.next_domain_index,
            "domains_per_day": self.domains_per_day,
            "max_day_offsets": self.max_day_offsets,
            "current_day_offset": self.next_domain_index // self.domains_per_day,
            "current_date": self._date_for_day_offset(self.next_domain_index // self.domains_per_day).date().isoformat(),
            "tld_variants": self.profile_data["tld_variants"],
            "distinct_base_labels_estimate": self.profile_data["distinct_base_labels_estimate"],
            "estimated_total_combination_space": self.profile_data["estimated_total_combination_space"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_practical_capacity": remaining_practical,
            "remaining_unique_capacity": remaining_practical,
            "structured_space": True,
            "finite_space": True,
        }

        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="monerodownloader_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "counter", "day_offset"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "supported_parameter_axes": ["date", "day_offset", "nr", "tld_variant"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.0, max(0.2, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.35,
            "apparent_finite_space": True,
            "generation_mode": "structured_date_index",
            "adapter_type": "monerodownloader_dedicated",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "tld_variants": self.profile_data["tld_variants"],
            "domains_per_day": self.profile_data["domains_per_day"],
            "distinct_base_labels_estimate": self.profile_data["distinct_base_labels_estimate"],
            "estimated_total_combination_space": self.profile_data["estimated_total_combination_space"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "structured_finite_combination": True,
            "collision_curve": self.profile_data["collision_curve"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
        }
        self._profile_cache = p
        return p
