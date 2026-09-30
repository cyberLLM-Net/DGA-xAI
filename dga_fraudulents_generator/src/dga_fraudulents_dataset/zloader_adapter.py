from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed

ZLOADER_KEYS = [
    "q23Cud3xsNf3",
    "41997b4a729e1a0175208305170752dd",
    "kZieCw23gffpe43Sd",
    "Ts72YjsjO5TghE6m",
    "03d5ae30a0bd934a23b6a7f0756aa504",
]


@dataclass(frozen=True)
class ZloaderParams:
    date: datetime
    key: str
    seed: int


class ZloaderAdapter(AlgorithmAdapter):
    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.seed_fn = getattr(self.module, "seeding", None)
        self.dga_fn = getattr(self.module, "dga", None)
        if self.dga_fn is None:
            raise RuntimeError("zloader dga.py missing dga(seed, nr_of_domains)")

        self.keys = self._resolve_keys()
        self.base_date = self._resolve_base_date()
        self.date_mode = str(self.defaults.get("date_mode", self.strategy_cfg.get("date_mode", "daily_forward"))).strip().lower()

        self.native_batch_generation = True
        self.next_seed_offset = self._resolve_resume_seed_offset()
        self.emitted_unique: set[str] = set()

        self.sample_offsets = max(64, int(self.defaults.get("profile_offsets", self.strategy_cfg.get("profile_offsets", 720))))
        self.sample_domains_per_offset = max(16, int(self.defaults.get("profile_domains_per_offset", self.strategy_cfg.get("profile_domains_per_offset", 64))))
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])

        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"zloader missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_zloader", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import zloader module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_keys(self) -> list[str]:
        raw = self.defaults.get("rc4_keys", self.strategy_cfg.get("rc4_keys", ZLOADER_KEYS))
        if isinstance(raw, str):
            keys = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, (list, tuple, set)):
            keys = [str(x).strip() for x in raw if str(x).strip()]
        else:
            keys = list(ZLOADER_KEYS)
        return keys or list(ZLOADER_KEYS)

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
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

    def _date_for_offset(self, offset: int) -> datetime:
        if self.date_mode == "daily_window":
            return self.base_date + timedelta(days=offset % 31)
        if self.date_mode == "fixed":
            return self.base_date
        return self.base_date + timedelta(days=offset)

    def _seed_from_date_key(self, date: datetime, key: str, offset: int) -> int:
        if self.seed_fn is not None:
            try:
                return int(self.seed_fn(date, key))
            except Exception:
                pass
        return stable_int_seed(f"zloader:{key}:{date.date().isoformat()}:{offset}")

    def _params_for_offset(self, offset: int) -> ZloaderParams:
        date = self._date_for_offset(offset)
        key = self.keys[offset % len(self.keys)]
        seed = self._seed_from_date_key(date, key, offset)
        return ZloaderParams(date=date, key=key, seed=seed)

    @staticmethod
    def _domains_from_seed(seed: int, nr_of_domains: int, start_index: int = 0) -> list[str]:
        if nr_of_domains <= 0:
            return []
        total = max(0, start_index) + nr_of_domains
        r = seed
        out: list[str] = []
        for i in range(total):
            domain = ""
            for _ in range(20):
                letter = ord("a") + (r % 25)
                domain += chr(letter)
                r = seed ^ ((r + letter) & 0xFFFFFFFF)
            domain += ".com"
            if i >= start_index:
                out.append(domain)
        return out

    @staticmethod
    def _estimate_capacity(sample_unique: int, sample_generated: int, tail_yield: float) -> int:
        if sample_generated <= 0:
            return 0
        if tail_yield <= 0.01:
            factor = 1.05
        elif tail_yield <= 0.03:
            factor = 1.12
        elif tail_yield <= 0.08:
            factor = 1.25
        elif tail_yield <= 0.18:
            factor = 1.45
        else:
            factor = 1.75
        return max(sample_unique, int(sample_unique * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        seed_only: set[str] = set()
        date_only: set[str] = set()
        combo: set[str] = set()

        generated = 0
        curve: list[dict[str, Any]] = []
        checkpoints = {max(1, int(self.sample_offsets * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)}

        # seed-only axis
        for i in range(min(128, self.sample_offsets)):
            seed_domains = self._domains_from_seed(stable_int_seed(f"zloader:seed_only:{i}"), 16)
            seed_only.update(seed_domains)

        # date-only axis (fixed key)
        key0 = self.keys[0]
        for i in range(min(128, self.sample_offsets)):
            d = self._date_for_offset(i)
            s = self._seed_from_date_key(d, key0, i)
            date_only.update(self._domains_from_seed(s, 16))

        # combo axis for runtime-like behavior
        for i in range(self.sample_offsets):
            params = self._params_for_offset(i)
            domains = self._domains_from_seed(params.seed, self.sample_domains_per_offset)
            generated += len(domains)
            before = len(unique)
            unique.update(domains)
            combo.update(domains)
            after = len(unique)

            if (i + 1) in checkpoints:
                marginal = after - before
                collision_rate = 1.0 - (after / max(generated, 1))
                curve.append(
                    {
                        "offset_samples": i + 1,
                        "generated": generated,
                        "unique": after,
                        "collision_rate": collision_rate,
                        "marginal_unique_gain": marginal,
                    }
                )

        sample_unique = len(unique)
        sample_yield = sample_unique / max(generated, 1)
        if len(curve) >= 2:
            prev = curve[-2]
            last = curve[-1]
            tail_generated = max(1, last["generated"] - prev["generated"])
            tail_unique = max(0, last["unique"] - prev["unique"])
            tail_yield = tail_unique / tail_generated
        else:
            tail_yield = sample_yield

        date_matters = len(date_only) > max(1, int(len(seed_only) * 0.55))
        estimated_cap = self._estimate_capacity(sample_unique, generated, tail_yield)
        collision_prone = sample_yield < 0.35 or tail_yield < 0.12

        return {
            "sample_offsets": self.sample_offsets,
            "sample_generated": generated,
            "sample_unique": sample_unique,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "collision_prone": collision_prone,
            "date_matters": date_matters,
            "seed_axis_unique": len(seed_only),
            "date_axis_unique": len(date_only),
            "combo_axis_unique": len(combo),
            "estimated_max_unique_capacity": estimated_cap,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="zloader_dedicated")

        start_offset = self.next_seed_offset
        params = self._params_for_offset(start_offset)
        domains = self._domains_from_seed(params.seed, batch_size)
        self.next_seed_offset += 1

        self.emitted_unique.update(domains)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "seed_date_collision_prone",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["seed", "date", "key", "nr_of_domains"],
            "date_matters": self.profile_data["date_matters"],
            "native_batch_generation": self.native_batch_generation,
            "seed_offset_start": start_offset,
            "next_seed_offset": self.next_seed_offset,
            "date": params.date.date().isoformat(),
            "rc4_key": params.key,
            "seed": params.seed,
            "nr_of_domains": batch_size,
            "tld_variants": [".com"],
            "collision_curve": self.profile_data["collision_curve"],
            "collision_prone": self.profile_data["collision_prone"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "finite_space": True,
        }

        return BatchGenerationResult(
            domains=domains,
            attempts=1,
            generated=len(domains),
            adapter_type="zloader_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["seed", "date", "key", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": bool(self.profile_data["date_matters"]),
            "supported_parameter_axes": ["seed", "date", "key", "nr_of_domains"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.2, max(0.08, self.profile_data["tail_unique_yield"] * 2.2)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.3 if self.profile_data["collision_prone"] else 0.9,
            "recommended_date_window": {"start": "n/a", "end": "n/a"},
            "apparent_finite_space": True,
            "generation_mode": "seed_date_collision_prone",
            "adapter_type": "zloader_dedicated",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "date_matters": self.profile_data["date_matters"],
            "native_batch_generation": self.native_batch_generation,
            "collision_curve": self.profile_data["collision_curve"],
            "collision_prone": self.profile_data["collision_prone"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "seed_axis_unique": self.profile_data["seed_axis_unique"],
            "date_axis_unique": self.profile_data["date_axis_unique"],
            "combo_axis_unique": self.profile_data["combo_axis_unique"],
            "tld_variants": [".com"],
        }
        self._profile_cache = p
        return p
