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

VARIANT_FILES = {"a": "dga_a.py", "b": "dga_b.py"}


@dataclass(frozen=True)
class VariantCfg:
    key: str
    path: Path
    module: ModuleType


class QsnatchAdapter(AlgorithmAdapter):
    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.base_date = self._resolve_base_date()
        self.day_window = max(7, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 365))))
        self.profile_days = max(8, int(self.defaults.get("profile_days", self.strategy_cfg.get("profile_days", 45))))
        self.qsnatch_variant = str(self.defaults.get("qsnatch_variant", self.strategy_cfg.get("qsnatch_variant", "both"))).strip().lower()
        self.selected_variants = self._resolve_variants()
        self.next_global_index = self._resolve_resume_index()

        self._pool_cache: dict[tuple[str, int], list[str]] = {}
        self._diag_cache: dict[tuple[str, int], dict[str, int]] = {}
        self.emitted_unique: set[str] = set()

        self.profile_data = self._build_profile_data()
        self.variant_weights = self._variant_weights()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_resume_index(self) -> int:
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

    def _load_module(self, path: Path, key: str) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"qsnatch variant {key} missing file: {path}")
        spec = importlib.util.spec_from_file_location(f"dga_fraudulents_dataset_qsnatch_{key}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import qsnatch variant {key} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "dga"):
            raise RuntimeError(f"qsnatch variant {key} missing dga(date)")
        return module

    def _resolve_variants(self) -> list[VariantCfg]:
        if self.qsnatch_variant not in {"a", "b", "both"}:
            raise RuntimeError("qsnatch_variant must be one of: a, b, both")
        keys = ["a", "b"] if self.qsnatch_variant == "both" else [self.qsnatch_variant]
        out: list[VariantCfg] = []
        base = Path(self.inspection.path)
        for key in keys:
            p = base / VARIANT_FILES[key]
            out.append(VariantCfg(key=key, path=p, module=self._load_module(p, key)))
        return out

    def _date_offset_for_slot(self, slot: int) -> int:
        # Diversified traversal: avoid consecutive neighborhoods.
        # 37 and 17 are coprime with common windows and scramble date adjacency.
        return ((slot * 37) + 17) % self.day_window

    def _date_for_offset(self, day_offset: int) -> datetime:
        return self.base_date - timedelta(days=day_offset)

    @staticmethod
    def _sanitize_domain(raw: str) -> tuple[str | None, str | None]:
        if raw is None:
            return None, "none"
        domain = str(raw).strip().lower().rstrip(".")
        if not domain:
            return None, "empty"
        if ".." in domain:
            parts = [p for p in domain.split(".") if p]
            if len(parts) < 2:
                return None, "empty_label"
            domain = ".".join(parts)
            return domain, "double_dot"
        if any(p == "" for p in domain.split(".")):
            return None, "empty_label"
        return domain, None

    def _domains_for_variant_day(self, variant: VariantCfg, day_offset: int) -> tuple[list[str], dict[str, int]]:
        key = (variant.key, day_offset)
        cached = self._pool_cache.get(key)
        if cached is not None:
            return cached, dict(self._diag_cache.get(key, {}))

        when = self._date_for_offset(day_offset)
        raw_domains = [str(x) for x in variant.module.dga(when)]
        diag = {
            "raw_total": len(raw_domains),
            "double_dot": 0,
            "empty_label": 0,
            "malformed_suffix_join": 0,
            "invalid_after_sanitize": 0,
            "valid_after_sanitize": 0,
        }
        out: list[str] = []

        for d in raw_domains:
            fixed, reason = self._sanitize_domain(d)
            if reason == "double_dot":
                diag["double_dot"] += 1
                diag["malformed_suffix_join"] += 1
            elif reason == "empty_label":
                diag["empty_label"] += 1

            if not fixed:
                continue
            vr = validate_domain(fixed)
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
                diag["valid_after_sanitize"] += 1
            else:
                diag["invalid_after_sanitize"] += 1

        self._pool_cache[key] = out
        self._diag_cache[key] = diag
        return out, dict(diag)

    def _variant_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for v in self.profile_data["variants"]:
            key = v["key"]
            score = max(0.01, (1.0 - v["invalid_rate"]) * v["unique_yield"])
            weights[key] = score
        return weights

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.08
        elif tail_yield <= 0.08:
            factor = 1.18
        elif tail_yield <= 0.2:
            factor = 1.35
        elif tail_yield <= 0.4:
            factor = 1.6
        else:
            factor = 2.0
        return max(sample_unique, int(sample_unique * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        variants: list[dict[str, Any]] = []
        combo_unique: set[str] = set()
        combo_generated = 0
        combo_curve: list[dict[str, Any]] = []

        for vc in self.selected_variants:
            unique: set[str] = set()
            generated = 0
            valid = 0
            double_dot_total = 0
            malformed_total = 0
            invalid_after = 0
            curve: list[dict[str, Any]] = []
            for i in range(self.profile_days):
                day_offset = self._date_offset_for_slot(i)
                domains, diag = self._domains_for_variant_day(vc, day_offset)
                generated += int(diag["raw_total"])
                valid += len(domains)
                invalid_after += int(diag["invalid_after_sanitize"])
                double_dot_total += int(diag["double_dot"])
                malformed_total += int(diag["malformed_suffix_join"])
                unique.update(domains)
                combo_unique.update(domains)
                combo_generated += len(domains)
                curve.append(
                    {
                        "day_samples": i + 1,
                        "generated_raw": generated,
                        "valid_sanitized": valid,
                        "unique": len(unique),
                        "collision_rate": 1.0 - (len(unique) / max(valid, 1)),
                    }
                )

            unique_yield = len(unique) / max(valid, 1)
            valid_rate = valid / max(generated, 1)
            invalid_rate = 1.0 - valid_rate
            tail_yield = unique_yield
            if len(curve) >= 2:
                prev = curve[-2]
                last = curve[-1]
                tail_gen = max(1, last["valid_sanitized"] - prev["valid_sanitized"])
                tail_uni = max(0, last["unique"] - prev["unique"])
                tail_yield = tail_uni / tail_gen

            variants.append(
                {
                    "key": vc.key,
                    "implementation_file": str(vc.path),
                    "sample_days": self.profile_days,
                    "generated_raw": generated,
                    "valid_sanitized": valid,
                    "unique": len(unique),
                    "valid_rate": valid_rate,
                    "invalid_rate": invalid_rate,
                    "invalid_reasons": {
                        "double_dot": double_dot_total,
                        "malformed_suffix_join": malformed_total,
                        "invalid_after_sanitize": invalid_after,
                    },
                    "unique_yield": unique_yield,
                    "tail_unique_yield": tail_yield,
                    "collision_curve": curve,
                }
            )

        variants_sorted = sorted(
            variants,
            key=lambda v: ((1.0 - v["invalid_rate"]) * v["unique_yield"], v["valid_rate"]),
            reverse=True,
        )
        best_variant = variants_sorted[0]["key"] if variants_sorted else None
        sample_unique = len(combo_unique)
        sample_yield = sample_unique / max(combo_generated, 1)
        tail = median([v["tail_unique_yield"] for v in variants]) if variants else sample_yield
        est = self._estimate_capacity(sample_unique, tail)
        invalid_rate_combo = median([v["invalid_rate"] for v in variants]) if variants else 0.0

        for i in range(1, self.profile_days + 1):
            if combo_generated <= 0:
                break
            # Coarse combined curve.
            frac = i / max(self.profile_days, 1)
            g = int(combo_generated * frac)
            u = int(sample_unique * frac)
            combo_curve.append(
                {
                    "sample_step": i,
                    "generated": g,
                    "unique": u,
                    "collision_rate": 1.0 - (u / max(g, 1)),
                }
            )

        return {
            "variants": variants_sorted,
            "best_variant": best_variant,
            "sample_generated": combo_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail,
            "invalid_rate": invalid_rate_combo,
            "collision_curve": combo_curve,
            "estimated_max_unique_capacity": est,
            "recommended_near_capacity_ratio": 0.9,
            "multi_variant": True,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="qsnatch_dedicated")

        out: list[str] = []
        attempts = 0
        start_index = self.next_global_index
        variant_usage: dict[str, int] = {v.key: 0 for v in self.selected_variants}
        day_offsets_used: list[int] = []
        invalid_diag_totals = {
            "double_dot": 0,
            "empty_label": 0,
            "malformed_suffix_join": 0,
            "invalid_after_sanitize": 0,
            "raw_total": 0,
        }

        total_weight = sum(self.variant_weights.get(v.key, 1.0) for v in self.selected_variants)
        if total_weight <= 0:
            total_weight = float(len(self.selected_variants))

        while len(out) < batch_size:
            for vc in self.selected_variants:
                if len(out) >= batch_size:
                    break
                key = vc.key
                w = self.variant_weights.get(key, 1.0) / total_weight
                take = max(1, int(batch_size * w / max(len(self.selected_variants), 1)))
                take = min(take, batch_size - len(out))

                slot = self.next_global_index + variant_usage[key]
                day_offset = self._date_offset_for_slot(slot)
                day_offsets_used.append(day_offset)
                domains, diag = self._domains_for_variant_day(vc, day_offset)

                for dk in invalid_diag_totals:
                    invalid_diag_totals[dk] += int(diag.get(dk, 0))

                if not domains:
                    variant_usage[key] += 1
                    attempts += 1
                    continue

                dlen = len(domains)
                start = (slot * 13) % dlen  # dispersed start point to reduce correlation.
                selected: list[str] = []
                for i in range(take):
                    selected.append(domains[(start + i * 7) % dlen])
                out.extend(selected)
                variant_usage[key] += 1
                attempts += 1

                if attempts > max(batch_size * 4, 128):
                    break
            if attempts > max(batch_size * 4, 128):
                break

        out = out[:batch_size]
        self.next_global_index += max(1, sum(variant_usage.values()))
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "variant_date_diversified",
            "adapter_type": "qsnatch_dedicated",
            "selected_variant_mode": self.qsnatch_variant,
            "selected_parameter_axes": ["variant", "date", "day_offset"],
            "available_variants": {k: v for k, v in VARIANT_FILES.items()},
            "variant_weights": self.variant_weights,
            "best_variant": self.profile_data["best_variant"],
            "variant_usage_slots": variant_usage,
            "start_global_index": start_index,
            "next_global_index": self.next_global_index,
            "base_date": self.base_date.date().isoformat(),
            "day_window": self.day_window,
            "day_offsets_used": day_offsets_used[:32],
            "invalid_diagnostics": invalid_diag_totals,
            "collision_curve": self.profile_data["collision_curve"],
            "variants": self.profile_data["variants"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "multi_variant": True,
            "finite_space": True,
        }
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="qsnatch_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["variant", "date", "day_offset", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "supported_parameter_axes": ["variant", "date", "day_offset"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "invalid_rate": self.profile_data["invalid_rate"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.8, max(0.1, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"] * 0.9)),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.45,
            "apparent_finite_space": True,
            "generation_mode": "variant_date_diversified",
            "adapter_type": "qsnatch_dedicated",
            "multi_variant": True,
            "variants": self.profile_data["variants"],
            "best_variant": self.profile_data["best_variant"],
            "collision_curve": self.profile_data["collision_curve"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
        }
        self._profile_cache = p
        return p
