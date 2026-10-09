from __future__ import annotations

import importlib.util
import sys
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection


class RamnitAdapter(AlgorithmAdapter):
    """Dedicated adapter preserving Ramnit's hexadecimal-seed semantics."""

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}
        self.algo_dir = Path(inspection.path)

        self.module = self._load_module(self.algo_dir / "dga.py")
        self.get_domains_fn = getattr(self.module, "get_domains", None)
        if not callable(self.get_domains_fn):
            raise RuntimeError("ramnit:missing_implementation:get_domains")

        self.seeds = self._resolve_seeds()
        self.tlds = self._resolve_tlds()
        self.sld_min_length = self._resolve_int("ramnit_sld_min_length", "SLD_MIN_LENGTH", 9)
        self.sld_max_length = self._resolve_int("ramnit_sld_max_length", "SLD_MAX_LENGTH", 25)
        if self.sld_min_length < 1 or self.sld_max_length < self.sld_min_length:
            raise RuntimeError("ramnit:invalid_sld_length_bounds")

        self.next_offset = self._resolve_resume_offset()
        self._streams, self._seed_offsets = self._build_streams(self.next_offset)
        self._profile_cache: dict[str, Any] | None = None

    @staticmethod
    def _load_module(path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"ramnit:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_ramnit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"ramnit:invalid_implementation:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _configured(self, key: str, fallback: Any) -> Any:
        return self.defaults.get(key, self.strategy_cfg.get(key, fallback))

    def _resolve_seeds(self) -> tuple[str, ...]:
        raw = self._configured("ramnit_seeds", getattr(self.module, "KNOWN_SEEDS", ()))
        if isinstance(raw, str):
            values = [value.strip() for value in raw.split(",") if value.strip()]
        elif isinstance(raw, (list, tuple, set)):
            values = [str(value).strip() for value in raw if str(value).strip()]
        else:
            values = []
        if not values:
            raise RuntimeError("ramnit:missing_hex_seeds")
        for value in values:
            try:
                int(value, 16)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"ramnit:invalid_hex_seed:{value}") from exc
        return tuple(values)

    def _resolve_tlds(self) -> tuple[str, ...]:
        raw = self._configured("ramnit_tlds", getattr(self.module, "TLDS", ()))
        if isinstance(raw, str):
            values = [value.strip().lstrip(".") for value in raw.split(",") if value.strip()]
        elif isinstance(raw, (list, tuple, set)):
            values = [str(value).strip().lstrip(".") for value in raw if str(value).strip()]
        else:
            values = []
        if not values:
            raise RuntimeError("ramnit:missing_tlds")
        return tuple(values)

    def _resolve_int(self, key: str, module_name: str, fallback: int) -> int:
        return int(self._configured(key, getattr(self.module, module_name, fallback)))

    def _resolve_resume_offset(self) -> int:
        raw = self._configured(
            "ramnit_resume_next_offset",
            self._configured("ramnit_next_offset", self._configured("resume_next_offset", self._configured("next_offset", 0))),
        )
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _new_stream(self, seed: str) -> Iterator[str]:
        return iter(
            self.get_domains_fn(
                seed=seed,
                number_domains=sys.maxsize,
                tlds=list(self.tlds),
                sld_min_length=self.sld_min_length,
                sld_max_length=self.sld_max_length,
            )
        )

    def _build_streams(self, offset: int) -> tuple[list[Iterator[str]], list[int]]:
        count = len(self.seeds)
        base, remainder = divmod(offset, count)
        consumed = [base + (1 if index < remainder else 0) for index in range(count)]
        streams: list[Iterator[str]] = []
        for seed, skip in zip(self.seeds, consumed):
            stream = self._new_stream(seed)
            if skip:
                next(islice(stream, skip, skip), None)
            streams.append(stream)
        return streams, consumed

    def generate(self, batch_size: int) -> BatchGenerationResult:
        start_offset = self.next_offset
        out: list[str] = []
        for _ in range(max(0, batch_size)):
            seed_index = self.next_offset % len(self.seeds)
            out.append(str(next(self._streams[seed_index])))
            self._seed_offsets[seed_index] += 1
            self.next_offset += 1

        params = {
            "generation_mode": "hex_seed_round_robin",
            "implementation_file": str(self.algo_dir / "dga.py"),
            "seeds": list(self.seeds),
            "tlds": list(self.tlds),
            "sld_min_length": self.sld_min_length,
            "sld_max_length": self.sld_max_length,
            "start_offset": start_offset,
            "next_offset": self.next_offset,
            "per_seed_next_offsets": dict(zip(self.seeds, self._seed_offsets)),
            "finite_space": False,
        }
        return BatchGenerationResult(
            domains=out,
            attempts=len(out),
            generated=len(out),
            adapter_type="ramnit_dedicated",
            last_effective_params=params,
            supported_parameter_axes=["seed", "sequence_offset", "tld"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        sample = list(
            self.get_domains_fn(
                seed=self.seeds[0],
                number_domains=max(1, sample_size),
                tlds=list(self.tlds),
                sld_min_length=self.sld_min_length,
                sld_max_length=self.sld_max_length,
            )
        )
        unique_yield = len(set(sample)) / max(len(sample), 1)
        self._profile_cache = {
            "generates_any": bool(sample),
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": False,
            "supported_parameter_axes": ["seed", "sequence_offset", "tld"],
            "sample_generated": len(sample),
            "sample_unique": len(set(sample)),
            "sample_unique_yield": unique_yield,
            "initial_health_score": max(0.1, unique_yield),
            "initial_capacity_score": max(0.2, unique_yield),
            "expected_diversity_score": max(0.1, unique_yield),
            "recommended_max_effective_quota_multiplier": 2.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": False,
            "adapter_type": "ramnit_dedicated",
            "implementation_file": str(self.algo_dir / "dga.py"),
            "seeds": list(self.seeds),
            "tlds": list(self.tlds),
            "sld_min_length": self.sld_min_length,
            "sld_max_length": self.sld_max_length,
        }
        return self._profile_cache
