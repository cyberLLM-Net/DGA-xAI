from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .charbot_generator import DEFAULT_TLDS, generate_batch, load_base_domains
from .models import AlgorithmInspection
from .utils import stable_int_seed


@dataclass
class CharbotConfig:
    base_domains_path: Path
    num_mutated_characters: int = 2
    allowed_tlds: list[str] | None = None
    min_base_domain_length: int = 6
    random_sampling: bool = True


class CharbotAdapter(AlgorithmAdapter):
    def __init__(
        self,
        inspection: AlgorithmInspection,
        seed_strategy: str,
    ) -> None:
        self.inspection = inspection
        self.seed_strategy = inspection.parameter_strategy.get("seed_strategy", seed_strategy)
        self.seed_base = stable_int_seed(inspection.algorithm_code)
        self.round_id = 0
        self._profile_cache: dict[str, Any] | None = None

        self.config = self._resolve_config(inspection)
        self.base_domains = load_base_domains(
            self.config.base_domains_path,
            min_base_domain_length=self.config.min_base_domain_length,
        )
        if not self.base_domains:
            raise RuntimeError(f"charbot base domain list is empty: {self.config.base_domains_path}")

    def _seed_value(self, rid: int) -> int:
        if self.seed_strategy in {"fixed", "incremental"}:
            return self.seed_base + (rid if self.seed_strategy != "fixed" else 0)
        if self.seed_strategy == "sequential":
            return self.seed_base + rid
        if self.seed_strategy == "hashed_round_robin":
            digest = hashlib.sha256(f"{self.inspection.algorithm_code}:{rid}".encode("utf-8")).hexdigest()
            return int(digest[:8], 16)
        return self.seed_base + rid

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _parse_tlds(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            values = [x.strip().lstrip(".").lower() for x in value.split(",") if x.strip()]
            return values or None
        if isinstance(value, (list, tuple, set)):
            values = [str(x).strip().lstrip(".").lower() for x in value if str(x).strip()]
            return values or None
        return None

    def _resolve_config(self, inspection: AlgorithmInspection) -> CharbotConfig:
        defaults = inspection.default_params or {}
        strategy = inspection.parameter_strategy or {}

        raw_base_path = (
            defaults.get("base_domains_path")
            or defaults.get("base_domain_file")
            or strategy.get("base_domains_path")
            or strategy.get("base_domain_file")
            or "domains_examples.txt"
        )
        base_path = Path(str(raw_base_path))
        if not base_path.is_absolute():
            base_path = Path(inspection.path) / base_path

        raw_sampling = (
            defaults.get("random_sampling")
            if "random_sampling" in defaults
            else strategy.get("random_sampling")
        )
        if raw_sampling is None:
            mode = defaults.get("base_domain_sampling", strategy.get("base_domain_sampling", "random"))
            random_sampling = str(mode).strip().lower() != "deterministic"
        else:
            random_sampling = self._to_bool(raw_sampling, default=True)

        raw_mutations = (
            defaults.get("num_mutated_characters")
            or defaults.get("mutated_characters")
            or strategy.get("num_mutated_characters")
            or strategy.get("mutated_characters")
            or 2
        )

        raw_min_len = (
            defaults.get("min_base_domain_length")
            or strategy.get("min_base_domain_length")
            or 6
        )

        raw_tlds = (
            defaults.get("allowed_tlds")
            if "allowed_tlds" in defaults
            else strategy.get("allowed_tlds")
        )

        return CharbotConfig(
            base_domains_path=base_path,
            num_mutated_characters=max(1, self._to_int(raw_mutations, 2)),
            allowed_tlds=self._parse_tlds(raw_tlds),
            min_base_domain_length=max(1, self._to_int(raw_min_len, 6)),
            random_sampling=random_sampling,
        )

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(
                domains=[],
                attempts=0,
                generated=0,
                adapter_type="charbot_python",
            )

        rid = self.round_id
        self.round_id += 1

        seed = self._seed_value(rid)
        counter = rid * batch_size

        domains = generate_batch(
            base_domains=self.base_domains,
            seed=seed,
            counter=counter,
            batch_size=batch_size,
            num_mutated_characters=self.config.num_mutated_characters,
            allowed_tlds=self.config.allowed_tlds or DEFAULT_TLDS,
            min_base_domain_length=self.config.min_base_domain_length,
            random_sampling=self.config.random_sampling,
        )

        return BatchGenerationResult(
            domains=domains,
            attempts=1,
            generated=len(domains),
            adapter_type="charbot_python",
            last_effective_params={"seed": seed, "counter": counter, "batch_size": batch_size},
            supported_parameter_axes=["seed", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        n = max(1, min(sample_size, 128))
        r1 = self.generate(n)
        r2 = self.generate(n)
        combined = r1.domains + r2.domains
        unique = len(set(combined))
        unique_ratio = unique / max(len(combined), 1)

        profile = {
            "generates_any": len(combined) > 0,
            "mode": "batch",
            "seed_sensitive": set(r1.domains) != set(r2.domains),
            "date_sensitive": False,
            "supported_parameter_axes": ["counter", "seed"],
            "sample_generated": len(combined),
            "sample_unique": unique,
            "sample_unique_yield": unique_ratio,
            "initial_health_score": min(1.0, max(0.1, unique_ratio)),
            "initial_capacity_score": min(2.0, max(0.2, unique_ratio * 2.0)),
            "expected_diversity_score": min(1.0, max(0.1, unique_ratio)),
            "recommended_max_effective_quota_multiplier": 1.0 + min(0.5, unique_ratio * 0.6),
            "recommended_saturation_sensitivity": 1.0 - min(0.8, unique_ratio),
            "recommended_date_window": {"start": "n/a", "end": "n/a"},
            "apparent_finite_space": unique_ratio < 0.2,
            "adapter_type": "charbot_python",
        }
        self._profile_cache = profile
        return profile
