from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed
from .validation import validate_domain


class DarkcracksAdapter(AlgorithmAdapter):
    """Type-safe dedicated adapter for darkcracks.

    darkcracks expects:
      dga(seed: str, date: datetime) -> str
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
            raise RuntimeError("darkcracks dga.py missing callable dga(seed, date)")

        self.base_date = self._resolve_base_date()
        self.day_window = max(30, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 365))))
        self.profile_slots = max(16, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 96))))

        self.base_seed = str(self.defaults.get("seed_text", self.strategy_cfg.get("seed_text", "Crackalackin'")))
        self.seed_texts = self._resolve_seed_texts()
        self.next_slot = self._resolve_resume_slot()
        self.emitted_unique: set[str] = set()

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"darkcracks missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_darkcracks", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import darkcracks module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_seed_texts(self) -> list[str]:
        raw = self.defaults.get("seed_texts", self.strategy_cfg.get("seed_texts"))
        if isinstance(raw, str):
            vals = [x.strip() for x in raw.split(",") if x.strip()]
            if vals:
                return vals
        if isinstance(raw, (list, tuple, set)):
            vals = [str(x).strip() for x in raw if str(x).strip()]
            if vals:
                return vals
        # default deterministic family around base_seed
        return [
            self.base_seed,
            f"{self.base_seed}:1",
            f"{self.base_seed}:2",
            f"{self.base_seed}:{stable_int_seed('darkcracks') & 0xFFFF:x}",
        ]

    def _resolve_resume_slot(self) -> int:
        raw = (
            self.strategy_cfg.get("resume_next_slot")
            or self.strategy_cfg.get("next_slot")
            or self.defaults.get("resume_next_slot")
            or self.defaults.get("next_slot")
            or 0
        )
        try:
            return max(0, int(raw))
        except Exception:
            return 0

    def _day_offset(self, slot: int) -> int:
        return ((slot * 17) + 5) % self.day_window

    def _date_for_slot(self, slot: int) -> datetime:
        return self.base_date + timedelta(days=self._day_offset(slot))

    def _seed_for_slot(self, slot: int) -> str:
        if self.seed_strategy == "fixed":
            return self.seed_texts[0]
        return self.seed_texts[slot % len(self.seed_texts)]

    @staticmethod
    def _classify_type_error(exc: Exception) -> str:
        txt = str(exc)
        if "encode" in txt and "has no attribute" in txt:
            return "type_mismatch_encode_expected_str"
        if "bytes-like object" in txt:
            return "type_mismatch_bytes_expected"
        return "type_mismatch"

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.02:
            factor = 1.05
        elif tail_yield <= 0.06:
            factor = 1.15
        elif tail_yield <= 0.15:
            factor = 1.3
        elif tail_yield <= 0.3:
            factor = 1.45
        else:
            factor = 1.75
        return max(sample_unique, int(sample_unique * factor))

    def _invoke_slot(self, slot: int) -> tuple[str | None, dict[str, Any], str | None]:
        date_value = self._date_for_slot(slot)
        seed_text = self._seed_for_slot(slot)
        normalized_seed = str(seed_text)
        normalized_date = date_value if isinstance(date_value, datetime) else datetime.fromisoformat(str(date_value))
        params = {
            "slot": slot,
            "seed": normalized_seed,
            "date": normalized_date.date().isoformat(),
            "types": {"seed": type(normalized_seed).__name__, "date": type(normalized_date).__name__},
        }
        try:
            out = self.dga_fn(normalized_seed, normalized_date)
            return str(out) if out is not None else None, params, None
        except Exception as exc:
            return None, params, self._classify_type_error(exc)

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        valid = 0
        invalid = 0
        type_errors: dict[str, int] = {}
        curve: list[dict[str, Any]] = []
        schedule: list[dict[str, Any]] = []

        for i in range(self.profile_slots):
            domain, params, err = self._invoke_slot(i)
            generated += 1
            schedule.append({"slot": i, "seed": params["seed"], "date": params["date"], "types": params["types"]})
            if err:
                invalid += 1
                type_errors[err] = int(type_errors.get(err, 0)) + 1
                continue
            vr = validate_domain(domain or "")
            if vr.is_valid and vr.normalized:
                valid += 1
                unique.add(vr.normalized)
            else:
                invalid += 1
            curve.append(
                {
                    "slot_samples": i + 1,
                    "generated": valid,
                    "unique": len(unique),
                    "collision_rate": 1.0 - (len(unique) / max(valid, 1)),
                }
            )

        unique_yield = len(unique) / max(valid, 1)
        valid_rate = valid / max(generated, 1)
        if len(curve) >= 2:
            prev = curve[-2]
            last = curve[-1]
            tail_gen = max(1, last["generated"] - prev["generated"])
            tail_uni = max(0, last["unique"] - prev["unique"])
            tail = tail_uni / tail_gen
        else:
            tail = unique_yield

        est = self._estimate_capacity(len(unique), tail)
        return {
            "sample_generated": valid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "valid_rate": valid_rate,
            "invalid_rate": 1.0 - valid_rate,
            "tail_unique_yield": tail,
            "collision_curve": curve,
            "type_mismatch_diagnostics": type_errors,
            "tested_parameter_schedule": schedule[:32],
            "estimated_max_unique_capacity": est,
            "recommended_near_capacity_ratio": 0.9,
            "structured_finite_combination": True,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="darkcracks_dedicated")

        out: list[str] = []
        start_slot = self.next_slot
        attempts = 0
        errors: list[str] = []
        schedules: list[dict[str, Any]] = []
        invalid_diag: dict[str, int] = {}

        while len(out) < batch_size:
            slot = self.next_slot
            domain, params, err = self._invoke_slot(slot)
            schedules.append(params)
            attempts += 1
            self.next_slot += 1
            if err:
                errors.append(
                    f"{err}:seed={params.get('seed')}:{params.get('types', {}).get('seed')},"
                    f"date={params.get('date')}:{params.get('types', {}).get('date')}"
                )
                invalid_diag[err] = int(invalid_diag.get(err, 0)) + 1
                if attempts > batch_size * 6:
                    break
                continue
            vr = validate_domain(domain or "")
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
            else:
                reason = vr.reason or "format"
                invalid_diag[reason] = int(invalid_diag.get(reason, 0)) + 1
            if attempts > batch_size * 6:
                break

        out = out[:batch_size]
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))
        eff = {
            "generation_mode": "type_normalized_seed_date",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["seed", "date", "counter"],
            "start_slot": start_slot,
            "next_slot": self.next_slot,
            "normalized_parameter_types": {"seed": "str", "date": "datetime"},
            "schedules_used": schedules[:32],
            "invalid_diagnostics": invalid_diag,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "finite_space": True,
            "structured_space": True,
        }
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            errors=errors[-5:],
            adapter_type="darkcracks_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["seed", "date", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": True,
            "supported_parameter_axes": ["seed", "date", "counter"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "valid_rate": self.profile_data["valid_rate"],
            "invalid_rate": self.profile_data["invalid_rate"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.5, max(0.08, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": True,
            "generation_mode": "type_normalized_seed_date",
            "adapter_type": "darkcracks_dedicated",
            "type_mismatch_diagnostics": self.profile_data["type_mismatch_diagnostics"],
            "tested_parameter_schedule": self.profile_data["tested_parameter_schedule"],
            "collision_curve": self.profile_data["collision_curve"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "structured_finite_combination": True,
        }
        self._profile_cache = p
        return p
