from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection


@dataclass(frozen=True)
class TinbaConfiguration:
    configuration_id: int
    seed: str
    initial_domain: str
    tlds: tuple[str, ...]
    num_domains: int

    @property
    def output_count(self) -> int:
        return 1 + self.num_domains * len(self.tlds)


TINBA_CONFIGURATIONS: tuple[TinbaConfiguration, ...] = (
    TinbaConfiguration(1, "oGkS3w3sGGOGG7oc", "ssrgwnrmgrxe.com", ("com",), 1000),
    TinbaConfiguration(2, "jc74FlUna852Ji9o", "blackfreeqazyio.cc", ("com", "net", "in", "ru"), 100),
    TinbaConfiguration(3, "yqokqFC2TPBFfJcG", "watchthisnow.xyz", ("pw", "us", "xyz", "club"), 100),
    TinbaConfiguration(4, "j193HsnW72Yqns7u", "j193hsne720uie8i.cc", ("com", "net", "biz", "org"), 100),
)


class TinbaAdapter(AlgorithmAdapter):
    """Dedicated adapter keeping every documented Tinba configuration coherent."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}
        self.algo_dir = Path(inspection.path)

        self.module = self._load_module(self.algo_dir / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        if not callable(self.dga_fn):
            raise RuntimeError("tinba:missing_implementation:dga")

        self.configurations = self._resolve_configurations()
        self._domains, self._domain_configuration_ids = self._materialize_domains()
        self.total_capacity = len(self._domains)
        self.next_offset = min(self._resolve_resume_offset(), self.total_capacity)
        self._profile_cache: dict[str, Any] | None = None

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"tinba:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_tinba", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"tinba:invalid_implementation:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _configured(self, key: str, fallback: Any) -> Any:
        return self.defaults.get(key, self.strategy_cfg.get(key, fallback))

    def _resolve_configurations(self) -> tuple[TinbaConfiguration, ...]:
        raw = self._configured("tinba_configurations", [cfg.configuration_id for cfg in TINBA_CONFIGURATIONS])
        if isinstance(raw, int):
            ids = [raw]
        elif isinstance(raw, str):
            ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
        elif isinstance(raw, (list, tuple, set)):
            ids = [int(value) for value in raw]
        else:
            ids = []
        by_id = {cfg.configuration_id: cfg for cfg in TINBA_CONFIGURATIONS}
        if not ids or any(config_id not in by_id for config_id in ids):
            raise RuntimeError(f"tinba:invalid_configuration_ids:{ids}")
        if len(set(ids)) != len(ids):
            raise RuntimeError("tinba:duplicate_configuration_ids")
        return tuple(by_id[config_id] for config_id in ids)

    def _resolve_resume_offset(self) -> int:
        raw = self._configured(
            "tinba_resume_next_offset",
            self._configured("tinba_next_offset", self._configured("resume_next_offset", self._configured("next_offset", 0))),
        )
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _materialize_domains(self) -> tuple[list[str], list[int]]:
        domains: list[str] = []
        configuration_ids: list[int] = []
        for cfg in self.configurations:
            generated = list(self.dga_fn(cfg.seed, cfg.initial_domain, cfg.tlds, cfg.num_domains))
            if len(generated) != cfg.output_count:
                raise RuntimeError(
                    f"tinba:unexpected_output_count:configuration={cfg.configuration_id}:"
                    f"expected={cfg.output_count}:actual={len(generated)}"
                )
            domains.extend(str(domain) for domain in generated)
            configuration_ids.extend([cfg.configuration_id] * len(generated))
        return domains, configuration_ids

    @staticmethod
    def _configuration_dict(cfg: TinbaConfiguration) -> dict[str, Any]:
        out = asdict(cfg)
        out["tlds"] = list(cfg.tlds)
        out["output_count"] = cfg.output_count
        return out

    def generate(self, batch_size: int) -> BatchGenerationResult:
        start_offset = self.next_offset
        end_offset = min(self.total_capacity, start_offset + max(0, batch_size))
        out = self._domains[start_offset:end_offset]
        used_ids = sorted(set(self._domain_configuration_ids[start_offset:end_offset]))
        self.next_offset = end_offset
        remaining = self.total_capacity - self.next_offset
        params = {
            "generation_mode": "documented_configuration_sequence",
            "implementation_file": str(self.algo_dir / "dga.py"),
            "configurations": [self._configuration_dict(cfg) for cfg in self.configurations],
            "configuration_ids_used": used_ids,
            "start_offset": start_offset,
            "next_offset": self.next_offset,
            "estimated_max_unique_capacity": self.total_capacity,
            "remaining_unique_capacity": remaining,
            "finite_space": True,
            "structured_space": True,
            "exhausted": remaining == 0,
        }
        return BatchGenerationResult(
            domains=list(out),
            attempts=len(out),
            generated=len(out),
            adapter_type="tinba_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["configuration", "sequence_offset"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        sample = self._domains[: max(1, min(sample_size, self.total_capacity))]
        unique_yield = len(set(sample)) / max(len(sample), 1)
        self._profile_cache = {
            "generates_any": bool(self._domains),
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": False,
            "supported_parameter_axes": ["configuration", "sequence_offset"],
            "sample_generated": len(sample),
            "sample_unique": len(set(sample)),
            "sample_unique_yield": unique_yield,
            "initial_health_score": max(0.1, unique_yield),
            "initial_capacity_score": max(0.1, unique_yield),
            "expected_diversity_score": max(0.1, unique_yield),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": True,
            "finite_space": True,
            "structured_finite_combination": True,
            "estimated_max_unique_capacity": self.total_capacity,
            "adapter_type": "tinba_dedicated",
            "implementation_file": str(self.algo_dir / "dga.py"),
            "configurations": [self._configuration_dict(cfg) for cfg in self.configurations],
        }
        return self._profile_cache
