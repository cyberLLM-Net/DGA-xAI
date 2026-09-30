from __future__ import annotations

import importlib.util
import inspect
import io
import re
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed

DOMAIN_RE = re.compile(r"[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")


class NymaimAdapter(AlgorithmAdapter):
    """Dedicated adapter for nymaim date+counter exploration.

    The upstream implementation exposes `dga(date, nr)` and prints domains.
    This adapter provides deterministic counter progression, resume support,
    and bounded empirical capacity profiling.
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
            raise RuntimeError("nymaim dga.py missing callable dga(date, nr)")
        self.sig = inspect.signature(self.dga_fn)

        self.base_date = self._resolve_base_date()
        self.date_mode = str(self.defaults.get("date_mode", self.strategy_cfg.get("date_mode", "daily_forward"))).strip().lower()
        self.domains_per_date = max(
            64,
            int(self.defaults.get("domains_per_date", self.strategy_cfg.get("domains_per_date", 1024))),
        )
        self.date_shuffle = bool(self.defaults.get("date_shuffle", self.strategy_cfg.get("date_shuffle", False)))
        self.date_shuffle_days = max(
            16,
            int(self.defaults.get("date_shuffle_days", self.strategy_cfg.get("date_shuffle_days", 365))),
        )
        self.date_shuffle_seed = int(
            self.defaults.get(
                "date_shuffle_seed",
                self.strategy_cfg.get("date_shuffle_seed", stable_int_seed("nymaim:date_shuffle")),
            )
        )
        self._shuffled_offsets: list[int] | None = None
        if self.date_shuffle:
            self._shuffled_offsets = list(range(self.date_shuffle_days))
            self._shuffled_offsets.sort(
                key=lambda i: stable_int_seed(f"nymaim:shuffle:{self.date_shuffle_seed}:{i}")
            )

        self.native_batch_generation = True
        self.next_counter = self._resolve_resume_counter()
        self.emitted_unique: set[str] = set()

        self.profile_combined_samples = max(
            1024,
            int(self.defaults.get("profile_combined_samples", self.strategy_cfg.get("profile_combined_samples", 8192))),
        )
        self.profile_counter_samples = max(
            256,
            int(self.defaults.get("profile_counter_samples", self.strategy_cfg.get("profile_counter_samples", 2048))),
        )
        self.profile_date_samples = max(
            64,
            int(self.defaults.get("profile_date_samples", self.strategy_cfg.get("profile_date_samples", 256))),
        )
        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"nymaim missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_nymaim", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import nymaim module from {path}")
        module = importlib.util.module_from_spec(spec)
        with redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_resume_counter(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_counter")
            or self.strategy_cfg.get("next_counter")
            or self.defaults.get("resume_next_counter")
            or self.defaults.get("next_counter")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    @staticmethod
    def _extract_domains(result: Any, captured_stdout: str) -> list[str]:
        out: list[str] = []
        if isinstance(result, str):
            out.append(result)
        elif isinstance(result, (list, tuple, set)):
            out.extend([str(x) for x in result])
        elif result is not None and hasattr(result, "__iter__"):
            out.extend([str(x) for x in result])
        printed = DOMAIN_RE.findall(captured_stdout)
        if printed:
            out.extend(printed)
        return out

    def _date_for_index(self, date_index: int) -> datetime:
        if self.date_mode == "fixed":
            return self.base_date

        if self.date_shuffle and self._shuffled_offsets:
            shuffled = self._shuffled_offsets[date_index % len(self._shuffled_offsets)]
            return self.base_date + timedelta(days=shuffled)

        if self.date_mode == "daily_window":
            return self.base_date + timedelta(days=date_index % 31)

        return self.base_date + timedelta(days=date_index)

    def _generate_for_date(self, when: datetime, start_index: int, count: int) -> list[str]:
        if count <= 0:
            return []
        n = start_index + count
        sink = io.StringIO()
        with redirect_stdout(sink):
            result = self.dga_fn(when, n)
        domains = self._extract_domains(result, sink.getvalue())
        if len(domains) <= start_index:
            return []
        return domains[start_index:n]

    def _combined_sample(self, total: int) -> list[str]:
        out: list[str] = []
        counter = 0
        while len(out) < total:
            date_index = counter // self.domains_per_date
            start = counter % self.domains_per_date
            take = min(total - len(out), self.domains_per_date - start)
            when = self._date_for_index(date_index)
            out.extend(self._generate_for_date(when, start, take))
            counter += take
        return out[:total]

    @staticmethod
    def _estimate_capacity(sample_unique: int, sample_generated: int, tail_yield: float) -> int:
        if sample_generated <= 0:
            return 0
        if tail_yield <= 0.04:
            factor = 1.05
        elif tail_yield <= 0.1:
            factor = 1.15
        elif tail_yield <= 0.2:
            factor = 1.3
        elif tail_yield <= 0.35:
            factor = 1.5
        else:
            factor = 1.7
        return max(sample_unique, int(sample_unique * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        counter_only = self._generate_for_date(self.base_date, 0, self.profile_counter_samples)

        date_only: list[str] = []
        for i in range(self.profile_date_samples):
            date_only.extend(self._generate_for_date(self._date_for_index(i), 0, 1))

        combined = self._combined_sample(self.profile_combined_samples)

        generated = len(combined)
        unique_seen = len(set(combined))
        sample_yield = unique_seen / max(generated, 1)

        curve: list[dict[str, Any]] = []
        checkpoints = {
            max(1, int(generated * r)) for r in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
        }
        running: set[str] = set()
        prev_unique = 0
        for idx, domain in enumerate(combined, start=1):
            running.add(domain)
            if idx in checkpoints:
                cur_unique = len(running)
                marginal = cur_unique - prev_unique
                prev_unique = cur_unique
                curve.append(
                    {
                        "sample_counter": idx,
                        "generated": idx,
                        "unique": cur_unique,
                        "collision_rate": 1.0 - (cur_unique / max(idx, 1)),
                        "marginal_unique_gain": marginal,
                    }
                )

        if len(curve) >= 2:
            prev = curve[-2]
            last = curve[-1]
            tail_generated = max(1, last["generated"] - prev["generated"])
            tail_unique = max(0, last["unique"] - prev["unique"])
            tail_yield = tail_unique / tail_generated
        else:
            tail_yield = sample_yield

        date_axis_unique = len(set(date_only))
        counter_axis_unique = len(set(counter_only))
        date_matters = date_axis_unique > 1
        counter_matters = counter_axis_unique > 1
        # Keep nymaim in conservative medium-capacity mode by default.
        # Small probes can look deceptively diverse, but large runs saturate.
        medium_capacity = True
        collision_prone = sample_yield < 0.52 or tail_yield < 0.32
        estimated_cap = self._estimate_capacity(unique_seen, generated, tail_yield)

        return {
            "sample_generated": generated,
            "sample_unique": unique_seen,
            "sample_unique_yield": sample_yield,
            "tail_unique_yield": tail_yield,
            "collision_curve": curve,
            "date_axis_unique": date_axis_unique,
            "counter_axis_unique": counter_axis_unique,
            "date_matters": date_matters,
            "counter_matters": counter_matters,
            "medium_capacity": medium_capacity,
            "collision_prone": collision_prone,
            "estimated_max_unique_capacity": estimated_cap,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="nymaim_dedicated")

        out: list[str] = []
        start_counter = self.next_counter
        attempts = 0

        while len(out) < batch_size:
            counter = self.next_counter
            date_index = counter // self.domains_per_date
            start = counter % self.domains_per_date
            take = min(batch_size - len(out), self.domains_per_date - start)
            when = self._date_for_index(date_index)
            out.extend(self._generate_for_date(when, start, take))
            self.next_counter += take
            attempts += 1

        out = out[:batch_size]
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "date_counter_medium_capacity",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["date", "counter", "nr"],
            "native_batch_generation": self.native_batch_generation,
            "hidden_parameter_detected": False,
            "date_mode": self.date_mode,
            "date_shuffle": self.date_shuffle,
            "domains_per_date": self.domains_per_date,
            "counter_start": start_counter,
            "next_counter": self.next_counter,
            "current_date": self._date_for_index(self.next_counter // self.domains_per_date).date().isoformat(),
            "collision_curve": self.profile_data["collision_curve"],
            "collision_prone": self.profile_data["collision_prone"],
            "nymaim_medium_capacity": self.profile_data["medium_capacity"],
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
        }

        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="nymaim_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": bool(self.profile_data["date_matters"]),
            "counter_sensitive": bool(self.profile_data["counter_matters"]),
            "supported_parameter_axes": ["date", "counter", "nr"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.4, max(0.08, self.profile_data["tail_unique_yield"] * 1.9)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.45 if self.profile_data["medium_capacity"] else 0.9,
            "recommended_date_window": {"start": "n/a", "end": "n/a"},
            "apparent_finite_space": True,
            "generation_mode": "date_counter_medium_capacity",
            "adapter_type": "nymaim_dedicated",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "native_batch_generation": self.native_batch_generation,
            "collision_curve": self.profile_data["collision_curve"],
            "collision_prone": self.profile_data["collision_prone"],
            "nymaim_medium_capacity": self.profile_data["medium_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "date_axis_unique": self.profile_data["date_axis_unique"],
            "counter_axis_unique": self.profile_data["counter_axis_unique"],
            "date_matters": self.profile_data["date_matters"],
            "counter_matters": self.profile_data["counter_matters"],
            "tld_variants": [".com", ".org", ".biz", ".net", ".info"],
        }
        self._profile_cache = p
        return p
