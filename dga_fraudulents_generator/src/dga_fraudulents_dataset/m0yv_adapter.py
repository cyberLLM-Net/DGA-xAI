from __future__ import annotations

import importlib.util
import inspect
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed


@dataclass(frozen=True)
class M0yvVariant:
    key: str
    filename: str
    requires_date: bool


VARIANTS: dict[str, M0yvVariant] = {
    "dga": M0yvVariant("dga", "dga.py", False),
    "td": M0yvVariant("td", "dga-td.py", True),
}


class M0yvAdapter(AlgorithmAdapter):
    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.strategy_cfg = inspection.parameter_strategy or {}
        self.defaults = inspection.default_params or {}

        self.variant, self.variant_path = self._resolve_variant()
        self.module = self._load_module(self.variant_path)
        self.fn = getattr(self.module, "dga", None)
        if self.fn is None:
            raise RuntimeError(f"m0yv selected variant has no dga() function: {self.variant_path}")
        self.sig = inspect.signature(self.fn)

        self.seed_base = self._resolve_seed_base()
        self.base_date = self._resolve_base_date()
        self.date_mode = str(self.defaults.get("td_date_mode", self.strategy_cfg.get("td_date_mode", "fixed"))).strip().lower()

        self.batch_domains_per_seed = 128
        self.next_seed_offset = self._resolve_resume_seed_offset()
        self.emitted_unique: set[str] = set()

        self.sample_seed_count = max(64, int(self.defaults.get("profile_seed_samples", self.strategy_cfg.get("profile_seed_samples", 1024))))
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _resolve_variant(self) -> tuple[M0yvVariant, Path]:
        raw = self.defaults.get("m0yv_variant", self.strategy_cfg.get("m0yv_variant", "dga"))
        variant_key = str(raw).strip().lower()
        if variant_key not in VARIANTS:
            raise RuntimeError(f"m0yv invalid variant '{variant_key}'. Supported: {sorted(VARIANTS)}")
        variant = VARIANTS[variant_key]
        p = Path(self.inspection.path) / variant.filename
        if not p.exists():
            raise RuntimeError(f"m0yv selected variant file does not exist: {p}")
        return variant, p

    def _resolve_seed_base(self) -> int:
        raw = self.defaults.get("seed_base", self.strategy_cfg.get("seed_base", 0x2484A18))
        try:
            return int(raw)
        except Exception:
            return stable_int_seed(f"{self.inspection.algorithm_code}:m0yv")

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("td_base_date", self.strategy_cfg.get("td_base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_resume_seed_offset(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_seed_offset")
            or self.strategy_cfg.get("next_seed_offset")
            or self.defaults.get("resume_next_seed_offset")
            or self.defaults.get("next_seed_offset")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _load_module(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"dga_fraudulents_dataset_m0yv_{path.stem.replace('-', '_')}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import m0yv module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _seed_for_offset(self, offset: int) -> int:
        if self.seed_strategy == "hashed_round_robin":
            return stable_int_seed(f"m0yv:{self.variant.key}:{offset}")
        if self.seed_strategy == "fixed":
            return self.seed_base
        return self.seed_base + offset

    def _date_for_offset(self, offset: int) -> datetime:
        if self.date_mode == "daily_window":
            return self.base_date + timedelta(days=offset % 31)
        if self.date_mode == "daily_forward":
            return self.base_date + timedelta(days=offset)
        return self.base_date

    @staticmethod
    def _to_list(items: Iterable[str] | Any) -> list[str]:
        if isinstance(items, list):
            return [str(x) for x in items]
        if isinstance(items, tuple):
            return [str(x) for x in items]
        if items is None:
            return []
        if isinstance(items, str):
            return [items]
        if hasattr(items, "__iter__"):
            out = []
            for x in items:
                out.append(str(x))
            return out
        return [str(items)]

    def _domains_for_seed_offset(self, offset: int) -> list[str]:
        seed = self._seed_for_offset(offset)
        if self.variant.requires_date:
            when = self._date_for_offset(offset)
            out = self.fn(seed, when)
        else:
            out = self.fn(seed)
        domains = self._to_list(out)
        return domains

    @staticmethod
    def _estimate_capacity(unique_seen: int, generated: int, tail_yield: float) -> int:
        if generated <= 0:
            return 0
        if tail_yield <= 0.002:
            factor = 1.03
        elif tail_yield <= 0.01:
            factor = 1.10
        elif tail_yield <= 0.03:
            factor = 1.22
        elif tail_yield <= 0.06:
            factor = 1.40
        else:
            factor = 2.0
        return max(unique_seen, int(unique_seen * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(self.sample_seed_count * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}

        for i in range(self.sample_seed_count):
            domains = self._domains_for_seed_offset(i)
            generated += len(domains)
            before = len(unique)
            unique.update(domains)
            after = len(unique)
            if (i + 1) in checkpoints:
                marginal_gain = after - before
                collision_rate = 1.0 - (after / max(generated, 1))
                curve.append(
                    {
                        "seed_samples": i + 1,
                        "generated": generated,
                        "unique": after,
                        "collision_rate": collision_rate,
                        "marginal_unique_gain": marginal_gain,
                    }
                )

        unique_seen = len(unique)
        overall_yield = unique_seen / max(generated, 1)
        if len(curve) >= 2:
            prev = curve[-2]
            last = curve[-1]
            tail_generated = max(1, last["generated"] - prev["generated"])
            tail_unique = max(0, last["unique"] - prev["unique"])
            tail_yield = tail_unique / tail_generated
        else:
            tail_yield = overall_yield

        estimated_capacity = self._estimate_capacity(unique_seen, generated, tail_yield)
        collision_prone = overall_yield < 0.12 or tail_yield < 0.02

        return {
            "sample_seed_count": self.sample_seed_count,
            "sample_generated": generated,
            "sample_unique": unique_seen,
            "sample_unique_yield": overall_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "collision_prone": collision_prone,
            "estimated_max_unique_capacity": estimated_capacity,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="m0yv_dedicated")

        start_offset = self.next_seed_offset
        seeds_to_use = max(1, math.ceil(batch_size / self.batch_domains_per_seed))

        out: list[str] = []
        for _ in range(seeds_to_use):
            offset = self.next_seed_offset
            out.extend(self._domains_for_seed_offset(offset))
            self.next_seed_offset += 1

        out = out[:batch_size]
        self.emitted_unique.update(out)

        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))
        params = {
            "generation_mode": "seed_collision_prone",
            "implementation_file": str(self.variant_path),
            "selected_variant": self.variant.key,
            "available_variants": {k: v.filename for k, v in VARIANTS.items()},
            "seed_offset_start": start_offset,
            "next_seed_offset": self.next_seed_offset,
            "seeds_used": seeds_to_use,
            "batch_domains_per_seed": self.batch_domains_per_seed,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "collision_prone": self.profile_data["collision_prone"],
            "collision_curve": self.profile_data["collision_curve"],
        }
        if self.variant.requires_date:
            params["date_mode"] = self.date_mode
            params["date_base"] = self.base_date.date().isoformat()

        return BatchGenerationResult(
            domains=out,
            attempts=seeds_to_use,
            generated=len(out),
            adapter_type="m0yv_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["seed", "seed_offset"] + (["date"] if self.variant.requires_date else []),
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        d0 = set(self._domains_for_seed_offset(0))
        d1 = set(self._domains_for_seed_offset(1))
        seed_sensitive = d0 != d1

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": seed_sensitive,
            "date_sensitive": self.variant.requires_date,
            "supported_parameter_axes": ["seed", "seed_offset"] + (["date"] if self.variant.requires_date else []),
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.0, max(0.02, self.profile_data["tail_unique_yield"] * 8.0)),
            "expected_diversity_score": min(1.0, max(0.02, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.2 if self.profile_data["collision_prone"] else 0.9,
            "recommended_date_window": {"start": "n/a", "end": "n/a"},
            "apparent_finite_space": True,
            "generation_mode": "seed_collision_prone",
            "adapter_type": "m0yv_dedicated",
            "implementation_file": str(self.variant_path),
            "selected_variant": self.variant.key,
            "available_variants": {k: v.filename for k, v in VARIANTS.items()},
            "collision_curve": self.profile_data["collision_curve"],
            "collision_prone": self.profile_data["collision_prone"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "tld_variants": [".biz"],
        }
        self._profile_cache = p
        return p
