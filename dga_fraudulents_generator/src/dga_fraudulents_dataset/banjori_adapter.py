from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class BanjoriAdapter(AlgorithmAdapter):
    """Dedicated adapter for banjori mutation-based constant-suffix DGA."""

    PREFIX_LENGTH = 4
    THEORETICAL_CAPACITY = 26 ** PREFIX_LENGTH  # ~456k

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.next_domain_fn = getattr(self.module, "next_domain", None)
        if self.next_domain_fn is None or not callable(self.next_domain_fn):
            raise RuntimeError("banjori:missing_implementation:next_domain_not_found")

        module_seed = getattr(self.module, "seed", "earnestnessbiophysicalohax.com")
        self.seed_domain = str(self.defaults.get("seed_domain", self.strategy_cfg.get("seed_domain", module_seed))).strip().lower()
        self.current_domain = str(
            self.strategy_cfg.get("resume_current_domain")
            or self.defaults.get("resume_current_domain")
            or self.seed_domain
        ).strip().lower()
        raw_counter = (
            self.strategy_cfg.get("resume_next_counter")
            or self.defaults.get("resume_next_counter")
            or self.strategy_cfg.get("next_counter")
            or self.defaults.get("next_counter")
            or 0
        )
        try:
            self.next_counter = max(0, int(raw_counter))
        except Exception:
            self.next_counter = 0

        self.constant_suffix = self.seed_domain[self.PREFIX_LENGTH :]
        self.estimated_max_unique_capacity = self.THEORETICAL_CAPACITY
        self.recommended_effective_quota_cap = min(
            self.estimated_max_unique_capacity,
            max(50_000, min(200_000, int(self.estimated_max_unique_capacity * 0.4))),
        )
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"banjori:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_banjori", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"banjori:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="banjori_dedicated")

        out: list[str] = []
        attempts = 0
        start_counter = self.next_counter
        start_domain = self.current_domain

        while len(out) < batch_size:
            d = str(self.current_domain).strip().lower()
            vr = validate_domain(d)
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
            self.current_domain = str(self.next_domain_fn(self.current_domain)).strip().lower()
            self.next_counter += 1
            attempts += 1
            if attempts > batch_size * 3:
                break

        params = {
            "generation_mode": "mutation_constant_suffix",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "seed_domain": self.seed_domain,
            "current_domain": self.current_domain,
            "start_domain": start_domain,
            "start_counter": start_counter,
            "next_counter": self.next_counter,
            "constant_suffix": self.constant_suffix,
            "prefix_length": self.PREFIX_LENGTH,
            "prefix_entropy": 4.7,
            "suffix_constancy": len(self.constant_suffix) / max(len(self.seed_domain), 1),
            "theoretical_capacity_estimate": self.estimated_max_unique_capacity,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "recommended_effective_quota_cap": self.recommended_effective_quota_cap,
            "low_structural_diversity": True,
            "saturation_point": None,
        }
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="banjori_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        # keep profile side-effect free by local simulation
        sim_domain = self.seed_domain
        sample: list[str] = []
        for _ in range(max(16, min(256, sample_size * 2))):
            sample.append(sim_domain)
            sim_domain = str(self.next_domain_fn(sim_domain)).strip().lower()
        valid = [v.normalized for v in (validate_domain(d) for d in sample) if v.is_valid and v.normalized]
        unique = len(set(valid))
        uniq_ratio = unique / max(len(valid), 1)

        self._profile_cache = {
            "generates_any": bool(valid),
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": False,
            "supported_parameter_axes": ["counter"],
            "sample_generated": len(valid),
            "sample_unique": unique,
            "sample_unique_yield": uniq_ratio,
            "initial_health_score": min(1.0, max(0.05, uniq_ratio)),
            "initial_capacity_score": min(0.7, max(0.05, uniq_ratio)),
            "expected_diversity_score": min(0.35, max(0.05, uniq_ratio * 0.4)),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.3,
            "apparent_finite_space": True,
            "generation_mode": "mutation_constant_suffix",
            "adapter_type": "banjori_dedicated",
            "prefix_entropy": 4.7,
            "suffix_constancy": len(self.constant_suffix) / max(len(self.seed_domain), 1),
            "constant_suffix": self.constant_suffix,
            "effective_variable_positions": self.PREFIX_LENGTH,
            "theoretical_capacity_estimate": self.estimated_max_unique_capacity,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "recommended_effective_quota_cap": self.recommended_effective_quota_cap,
            "low_structural_diversity": True,
        }
        return self._profile_cache
