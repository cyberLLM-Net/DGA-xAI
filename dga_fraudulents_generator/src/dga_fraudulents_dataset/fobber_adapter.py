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


@dataclass
class FobberVariantState:
    key: str
    version: int
    r: int
    c: int
    length: int
    tld: str
    next_counter: int = 0


class FobberAdapter(AlgorithmAdapter):
    """Dedicated adapter for fobber with explicit version modes."""

    VARIANTS = {
        "v1": {"version": 1, "r": 0xC87C8A78, "c": -1719405398, "length": 17, "tld": ".net"},
        "v2": {"version": 2, "r": 0x851A3E59, "c": -1916503263, "length": 10, "tld": ".com"},
    }

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.ror32 = getattr(self.module, "ror32", None)
        if not callable(self.ror32):
            raise RuntimeError("fobber:invalid_resource_format:missing_ror32")

        self.variant_mode = str(self.defaults.get("fobber_variant", self.strategy_cfg.get("fobber_variant", "both"))).strip().lower()
        self.selected_variants = self._resolve_variants()
        self.profile_steps = max(128, int(self.defaults.get("profile_steps", self.strategy_cfg.get("profile_steps", 4096))))
        self.per_variant_cap = max(128, int(self.defaults.get("per_variant_cap", self.strategy_cfg.get("per_variant_cap", 120000))))

        self.profile_data = self._build_profile_data()
        self.variant_weights = self._variant_weights()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"fobber:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_fobber", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"fobber:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_variants(self) -> list[FobberVariantState]:
        mode = self.variant_mode
        if mode in {"both", "*", "all"}:
            keys = ["v1", "v2"]
        elif mode in {"1", "v1"}:
            keys = ["v1"]
        elif mode in {"2", "v2"}:
            keys = ["v2"]
        else:
            raise RuntimeError("fobber:unsupported_variant")

        out: list[FobberVariantState] = []
        for key in keys:
            cfg = self.VARIANTS[key]
            resume = int(
                self.strategy_cfg.get(f"resume_{key}_next_counter")
                or self.defaults.get(f"resume_{key}_next_counter")
                or 0
            )
            st = FobberVariantState(
                key=key,
                version=int(cfg["version"]),
                r=int(cfg["r"]),
                c=int(cfg["c"]),
                length=int(cfg["length"]),
                tld=str(cfg["tld"]),
                next_counter=max(0, resume),
            )
            if st.next_counter > 0:
                self._fast_forward(st, st.next_counter)
            out.append(st)
        return out

    def _step(self, st: FobberVariantState) -> str:
        domain = []
        for _ in range(st.length):
            st.r = int(self.ror32((321167 * st.r + st.c) & 0xFFFFFFFF, 16))
            domain.append(chr((st.r & 0x17FF) % 26 + ord("a")))
        return "".join(domain) + st.tld

    def _fast_forward(self, st: FobberVariantState, domain_steps: int) -> None:
        for _ in range(max(0, int(domain_steps))):
            for _ in range(st.length):
                st.r = int(self.ror32((321167 * st.r + st.c) & 0xFFFFFFFF, 16))

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.02:
            factor = 1.06
        elif tail_yield <= 0.08:
            factor = 1.16
        elif tail_yield <= 0.2:
            factor = 1.32
        elif tail_yield <= 0.4:
            factor = 1.55
        else:
            factor = 2.0
        return max(sample_unique, int(sample_unique * factor))

    def _profile_variant(self, base: FobberVariantState, steps: int) -> dict[str, Any]:
        st = FobberVariantState(**base.__dict__)
        unique: set[str] = set()
        generated = 0
        valid = 0
        invalid = 0
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(steps * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}
        for i in range(steps):
            d = self._step(st)
            generated += 1
            vr = validate_domain(d)
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
        valid_rate = valid / max(generated, 1)
        if len(curve) >= 2:
            prev, last = curve[-2], curve[-1]
            tail_gen = max(1, last["generated"] - prev["generated"])
            tail_uni = max(0, last["unique"] - prev["unique"])
            tail = tail_uni / tail_gen
        else:
            tail = unique_yield
        cap = min(self.per_variant_cap, self._estimate_capacity(len(unique), tail))
        return {
            "variant": base.key,
            "version": base.version,
            "tld": base.tld,
            "generated": generated,
            "valid": valid,
            "invalid": invalid,
            "valid_rate": valid_rate,
            "unique": len(unique),
            "unique_yield": unique_yield,
            "tail_unique_yield": tail,
            "collision_curve": curve,
            "estimated_capacity": cap,
        }

    def _build_profile_data(self) -> dict[str, Any]:
        per_variant = [self._profile_variant(v, self.profile_steps) for v in self.selected_variants]
        sample_generated = sum(v["valid"] for v in per_variant)
        sample_unique = sum(v["unique"] for v in per_variant)
        sample_unique_yield = sample_unique / max(sample_generated, 1)
        tail = median([v["tail_unique_yield"] for v in per_variant]) if per_variant else 0.0
        est_cap = min(sum(v["estimated_capacity"] for v in per_variant), self._estimate_capacity(sample_unique, tail))
        best_variant = sorted(per_variant, key=lambda x: (x["unique_yield"], x["valid_rate"]), reverse=True)[0]["variant"] if per_variant else None
        return {
            "sample_generated": sample_generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_unique_yield,
            "tail_unique_yield": tail,
            "per_variant": per_variant,
            "best_variant": best_variant,
            "estimated_max_unique_capacity": est_cap,
            "structured_finite_combination": True,
            "recommended_near_capacity_ratio": 0.9,
            "adapter_type": "fobber_dedicated",
            "generation_mode": "variant_counter_structured",
            "failure_classification": {
                "missing_required_counter": "counter_not_initialized",
                "uninitialized_local_state": "nr_not_initialized_in_upstream_dga",
                "unsupported_variant": "invalid_variant_mode",
            },
        }

    def _variant_weights(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for v in self.profile_data["per_variant"]:
            out[v["variant"]] = max(0.01, v["unique_yield"] * v["valid_rate"])
        return out

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="fobber_dedicated")

        out: list[str] = []
        attempts = 0
        variant_usage: dict[str, int] = {v.key: 0 for v in self.selected_variants}
        total_w = sum(self.variant_weights.get(v.key, 1.0) for v in self.selected_variants) or float(len(self.selected_variants))

        while len(out) < batch_size:
            for st in self.selected_variants:
                if len(out) >= batch_size:
                    break
                frac = self.variant_weights.get(st.key, 1.0) / total_w
                take = max(1, int((batch_size * frac) / max(1, len(self.selected_variants))))
                take = min(take, batch_size - len(out))
                for _ in range(take):
                    d = self._step(st)
                    vr = validate_domain(d)
                    if vr.is_valid and vr.normalized:
                        out.append(vr.normalized)
                        st.next_counter += 1
                        variant_usage[st.key] += 1
                attempts += 1
                if attempts > batch_size * 4:
                    break
            if attempts > batch_size * 4:
                break

        out = out[:batch_size]
        schedule = {
            f"{s.key}_next_counter": s.next_counter for s in self.selected_variants
        }
        remaining = max(0, self.estimated_max_unique_capacity - len(set(out)))
        eff = {
            "generation_mode": "variant_counter_structured",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_variant_mode": self.variant_mode,
            "selected_variants": [s.key for s in self.selected_variants],
            "selected_parameter_axes": ["variant", "counter"],
            "available_variants": {"v1": 1, "v2": 2},
            "best_variant": self.profile_data["best_variant"],
            "variant_usage": variant_usage,
            "counter_schedule": schedule,
            "v1_next_counter": schedule.get("v1_next_counter", 0),
            "v2_next_counter": schedule.get("v2_next_counter", 0),
            "nr_source": "adapter_initialized_counter",
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining,
            "finite_space": True,
            "structured_space": True,
            "failure_classification": self.profile_data["failure_classification"],
        }

        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="fobber_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["variant", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": False,
            "supported_parameter_axes": ["variant", "counter"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.5, max(0.08, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": True,
            "generation_mode": "variant_counter_structured",
            "adapter_type": "fobber_dedicated",
            "per_variant": self.profile_data["per_variant"],
            "best_variant": self.profile_data["best_variant"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "structured_finite_combination": True,
            "failure_classification": self.profile_data["failure_classification"],
        }
        self._profile_cache = p
        return p
