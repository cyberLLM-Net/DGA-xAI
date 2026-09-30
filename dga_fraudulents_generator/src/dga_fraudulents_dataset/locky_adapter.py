from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


@dataclass(frozen=True)
class LockyVariant:
    key: str
    file_name: str
    module: ModuleType
    dga_fn: Any
    config_ids: list[int]


class LockyAdapter(AlgorithmAdapter):
    """Dedicated variant-aware adapter for locky (dgav2/dgav3)."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.base_date = self._resolve_base_date()
        self.day_window = max(365, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 3650))))
        self.domain_nr_span = max(8, int(self.defaults.get("domain_nr_span", self.strategy_cfg.get("domain_nr_span", 2048))))
        self.profile_slots = max(128, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 1536))))
        self.next_schedule_offset = self._resolve_resume_offset()

        self.variants = self._load_variants()
        self.variant_mode = self._resolve_variant_mode()
        self.selected_variant_keys = self._select_variants_from_mode(self.variant_mode)

        self.profile_data = self._build_profile_data()
        if self.variant_mode == "auto" and self.profile_data.get("best_variant"):
            self.selected_variant_keys = [str(self.profile_data["best_variant"])]
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

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

    def _load_module(self, path: Path, mod_name: str) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"locky:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"locky:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_variants(self) -> dict[str, LockyVariant]:
        root = Path(self.inspection.path)
        variants: dict[str, LockyVariant] = {}
        for key, fname in (("v2", "dgav2.py"), ("v3", "dgav3.py")):
            p = root / fname
            if not p.exists():
                continue
            mod = self._load_module(p, f"dga_fraudulents_dataset_locky_{key}")
            dga_fn = getattr(mod, "dga", None)
            cfg = getattr(mod, "config", None)
            if not callable(dga_fn) or not isinstance(cfg, dict) or not cfg:
                continue
            cfg_ids = sorted(int(k) for k in cfg.keys())
            variants[key] = LockyVariant(key=key, file_name=fname, module=mod, dga_fn=dga_fn, config_ids=cfg_ids)
        if not variants:
            raise RuntimeError("locky:missing_implementation:no_valid_dgav2_or_dgav3")
        return variants

    def _resolve_variant_mode(self) -> str:
        raw = self.defaults.get("locky_variant", self.strategy_cfg.get("locky_variant", "auto"))
        mode = str(raw).strip().lower()
        if mode in {"auto", "v2", "v3", "both"}:
            return mode
        raise RuntimeError(f"locky:unsupported_variant:{mode}")

    def _select_variants_from_mode(self, mode: str) -> list[str]:
        available = sorted(self.variants.keys())
        if mode == "both":
            return available
        if mode == "auto":
            return available
        if mode in self.variants:
            return [mode]
        raise RuntimeError(f"locky:missing_variant:{mode}")

    def _domain_args(self, variant: LockyVariant, logical_slot: int) -> tuple[datetime, int, int]:
        cfg_count = max(1, len(variant.config_ids))
        date_idx = logical_slot // (cfg_count * self.domain_nr_span)
        rem = logical_slot % (cfg_count * self.domain_nr_span)
        cfg_idx = rem // self.domain_nr_span
        domain_nr = rem % self.domain_nr_span
        date_value = self.base_date + timedelta(days=(date_idx % self.day_window))
        config_nr = variant.config_ids[cfg_idx % cfg_count]
        return date_value, config_nr, domain_nr

    def _slot_to_invocation(self, schedule_offset: int) -> tuple[LockyVariant, datetime, int, int]:
        selected = [self.variants[k] for k in self.selected_variant_keys]
        v = selected[schedule_offset % len(selected)]
        logical = schedule_offset // len(selected)
        d, cfg, nr = self._domain_args(v, logical)
        return v, d, cfg, nr

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float, cfg_count: int, day_window: int, domain_span: int) -> int:
        theoretical = max(1, cfg_count * day_window * domain_span)
        if tail_yield > 0.95:
            factor = 1.0
        elif tail_yield > 0.8:
            factor = 0.85
        elif tail_yield > 0.6:
            factor = 0.7
        else:
            factor = 0.5
        empirical = max(sample_unique, int(sample_unique / max(1e-6, 1.0 - min(tail_yield, 0.99))))
        return max(sample_unique, min(theoretical, int(max(sample_unique, empirical) * factor)))

    def _profile_variant(self, variant: LockyVariant, slots: int) -> dict[str, Any]:
        unique: set[str] = set()
        valid = 0
        invalid = 0
        curve: list[dict[str, Any]] = []
        seen_cfg: set[int] = set()
        seen_nr: set[int] = set()
        checkpoints = {max(1, int(slots * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}

        for i in range(slots):
            d, cfg, nr = self._domain_args(variant, i)
            seen_cfg.add(cfg)
            seen_nr.add(nr)
            out = variant.dga_fn(d, cfg, nr)
            vr = validate_domain(str(out))
            if vr.is_valid and vr.normalized:
                valid += 1
                unique.add(vr.normalized)
            else:
                invalid += 1
            if (i + 1) in checkpoints:
                curve.append(
                    {
                        "sample_step": i + 1,
                        "generated": valid,
                        "unique": len(unique),
                        "collision_rate": 1.0 - (len(unique) / max(valid, 1)),
                    }
                )

        unique_yield = len(unique) / max(valid, 1)
        valid_rate = valid / max(slots, 1)
        if len(curve) >= 2:
            prev, last = curve[-2], curve[-1]
            tail_gen = max(1, last["generated"] - prev["generated"])
            tail_uni = max(0, last["unique"] - prev["unique"])
            tail = tail_uni / tail_gen
        else:
            tail = unique_yield

        est = self._estimate_capacity(
            len(unique),
            tail,
            cfg_count=len(variant.config_ids),
            day_window=self.day_window,
            domain_span=self.domain_nr_span,
        )
        return {
            "variant": variant.key,
            "implementation_file": str(Path(self.inspection.path) / variant.file_name),
            "sample_generated": slots,
            "sample_valid": valid,
            "sample_invalid": invalid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "valid_rate": valid_rate,
            "tail_unique_yield": tail,
            "duplicate_growth_curve": curve,
            "config_nr_explored": sorted(seen_cfg),
            "domain_nr_explored_count": len(seen_nr),
            "estimated_capacity": est,
        }

    def _build_profile_data(self) -> dict[str, Any]:
        per_variant = [self._profile_variant(self.variants[k], self.profile_slots) for k in sorted(self.variants.keys())]
        best = sorted(per_variant, key=lambda x: (x["sample_unique_yield"], x["valid_rate"]), reverse=True)[0]["variant"]
        if self.variant_mode in {"v2", "v3"}:
            best = self.variant_mode
        selected = [x for x in per_variant if x["variant"] in (self.selected_variant_keys if self.variant_mode != "auto" else [best])]
        sample_generated = sum(int(x["sample_valid"]) for x in selected)
        sample_unique = sum(int(x["sample_unique"]) for x in selected)
        sample_yield = sample_unique / max(sample_generated, 1)
        tail = median([float(x["tail_unique_yield"]) for x in selected]) if selected else 0.0
        est = sum(int(x["estimated_capacity"]) for x in selected)
        return {
            "per_variant": per_variant,
            "best_variant": best,
            "selected_variant_mode": self.variant_mode,
            "selected_variants": [best] if self.variant_mode == "auto" else self.selected_variant_keys,
            "sample_generated": sample_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail,
            "estimated_max_unique_capacity": max(sample_unique, est),
            "redistribution_absorption_reason": "high_unique_yield_and_low_collision",
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="locky_dedicated")
        domains: list[str] = []
        attempts = 0
        start_offset = self.next_schedule_offset
        used_cfg: set[int] = set()
        used_nr: set[int] = set()
        used_variants: set[str] = set()
        while len(domains) < batch_size:
            v, d, cfg, nr = self._slot_to_invocation(self.next_schedule_offset)
            self.next_schedule_offset += 1
            attempts += 1
            used_cfg.add(cfg)
            used_nr.add(nr)
            used_variants.add(v.key)
            out = v.dga_fn(d, cfg, nr)
            vr = validate_domain(str(out))
            if vr.is_valid and vr.normalized:
                domains.append(vr.normalized)
            if attempts > batch_size * 4:
                break

        selected_variants = [self.profile_data["best_variant"]] if self.variant_mode == "auto" else self.selected_variant_keys
        impl_files = [str(Path(self.inspection.path) / self.variants[k].file_name) for k in selected_variants if k in self.variants]
        params = {
            "generation_mode": "date_config_domain_schedule",
            "selected_variant_mode": self.variant_mode,
            "selected_variants": selected_variants,
            "selected_implementation_files": impl_files,
            "next_schedule_offset": self.next_schedule_offset,
            "start_schedule_offset": start_offset,
            "config_nr_explored": sorted(used_cfg),
            "domain_nr_explored_count": len(used_nr),
            "domain_nr_span": self.domain_nr_span,
            "day_window": self.day_window,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "redistribution_absorption_reason": "high_unique_yield_and_low_collision",
        }
        return BatchGenerationResult(
            domains=domains[:batch_size],
            attempts=attempts,
            generated=min(len(domains), batch_size),
            adapter_type="locky_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["date", "config_nr", "domain_nr"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        selected_variants = [self.profile_data["best_variant"]] if self.variant_mode == "auto" else self.selected_variant_keys
        selected_impl = [str(Path(self.inspection.path) / self.variants[k].file_name) for k in selected_variants if k in self.variants]
        self._profile_cache = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "supported_parameter_axes": ["date", "config_nr", "domain_nr"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.5, max(0.2, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.1, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 3.0,
            "recommended_saturation_sensitivity": 0.7,
            "apparent_finite_space": False,
            "generation_mode": "date_config_domain_schedule",
            "adapter_type": "locky_dedicated",
            "per_variant": self.profile_data["per_variant"],
            "best_variant": self.profile_data["best_variant"],
            "selected_variant_mode": self.variant_mode,
            "selected_variants": selected_variants,
            "selected_implementation_files": selected_impl,
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "redistribution_absorption_reason": self.profile_data["redistribution_absorption_reason"],
            "config_domain_strategy": "date_then_config_then_domain_nr_deterministic",
        }
        return self._profile_cache
