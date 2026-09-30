from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed
from .validation import validate_domain


class CorebotAdapter(AlgorithmAdapter):
    """Dedicated adapter for corebot parameter space.

    Corebot generation is driven by year/month/day plus seed controls (nr_b, r).
    """

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.init_fn = getattr(self.module, "init_rand_and_chars", None)
        self.gen_fn = getattr(self.module, "generate_domain", None)
        if self.init_fn is None or self.gen_fn is None:
            raise RuntimeError("corebot requires init_rand_and_chars() and generate_domain()")

        self.base_date = self._resolve_base_date()
        self.day_window = max(31, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 365))))
        self.domains_per_schedule = max(4, int(self.defaults.get("domains_per_schedule", self.strategy_cfg.get("domains_per_schedule", 40))))
        self.profile_slots = max(12, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 64))))

        self.nr_b_values = self._resolve_nr_b_values()
        self.r_seeds = self._resolve_r_seeds()
        self.next_schedule_offset = self._resolve_resume_offset()
        self.emitted_unique: set[str] = set()

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"corebot missing implementation file: {path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_corebot", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import corebot module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_nr_b_values(self) -> list[int]:
        raw = self.defaults.get("nr_b_values", self.strategy_cfg.get("nr_b_values", [1, 2, 3, 4]))
        if isinstance(raw, str):
            vals = []
            for x in raw.split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    vals.append(int(x))
                except Exception:
                    pass
            return vals or [1, 2, 3, 4]
        if isinstance(raw, (list, tuple, set)):
            vals = []
            for x in raw:
                try:
                    vals.append(int(x))
                except Exception:
                    continue
            return sorted(set(vals)) or [1, 2, 3, 4]
        return [1, 2, 3, 4]

    def _resolve_r_seeds(self) -> list[int]:
        raw = self.defaults.get(
            "r_seeds",
            self.strategy_cfg.get(
                "r_seeds",
                [0x1DBA8930, stable_int_seed("corebot:seed2"), stable_int_seed("corebot:seed3")],
            ),
        )
        if isinstance(raw, str):
            vals = []
            for x in raw.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                try:
                    vals.append(int(x, 16) if x.startswith("0x") else int(x))
                except Exception:
                    pass
            return vals or [0x1DBA8930]
        if isinstance(raw, (list, tuple, set)):
            vals = []
            for x in raw:
                try:
                    vals.append(int(x))
                except Exception:
                    continue
            return vals or [0x1DBA8930]
        return [0x1DBA8930]

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

    def _day_offset_for_slot(self, slot: int) -> int:
        return ((slot * 29) + 11) % self.day_window

    def _date_for_slot(self, slot: int) -> datetime:
        return self.base_date - timedelta(days=self._day_offset_for_slot(slot))

    def _params_for_slot(self, slot: int) -> tuple[datetime, int, int]:
        when = self._date_for_slot(slot)
        nb = self.nr_b_values[slot % len(self.nr_b_values)]
        r0 = self.r_seeds[(slot // len(self.nr_b_values)) % len(self.r_seeds)]
        r = (r0 + (slot * 0x9E3779B1)) & 0xFFFFFFFF
        return when, nb, r

    def _generate_schedule(self, slot: int, count: int | None = None) -> tuple[list[str], dict[str, Any]]:
        when, nb, r = self._params_for_slot(slot)
        charset, state = self.init_fn(when.year, when.month, when.day, nb, r)
        n = count if count is not None else self.domains_per_schedule

        out: list[str] = []
        valid = 0
        invalid = 0
        invalid_reasons: dict[str, int] = {}
        for _ in range(n):
            sink = io.StringIO()
            with redirect_stdout(sink):
                next_state = self.gen_fn(charset, state)
            state = int(next_state) if next_state is not None else state
            printed = [ln.strip() for ln in sink.getvalue().splitlines() if ln.strip()]
            cand = printed[-1] if printed else ""
            if not cand:
                invalid += 1
                invalid_reasons["empty"] = int(invalid_reasons.get("empty", 0)) + 1
                continue
            vr = validate_domain(cand)
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
                valid += 1
            else:
                invalid += 1
                rk = vr.reason or "format"
                invalid_reasons[rk] = int(invalid_reasons.get(rk, 0)) + 1

        meta = {
            "date": when.date().isoformat(),
            "year": when.year,
            "month": when.month,
            "day": when.day,
            "nr_b": nb,
            "r": r,
            "generated_raw": n,
            "valid": valid,
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
        }
        return out, meta

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.06
        elif tail_yield <= 0.08:
            factor = 1.16
        elif tail_yield <= 0.2:
            factor = 1.3
        elif tail_yield <= 0.4:
            factor = 1.55
        else:
            factor = 1.9
        return max(sample_unique, int(sample_unique * factor))

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        valid = 0
        invalid = 0
        curve: list[dict[str, Any]] = []
        tested: list[dict[str, Any]] = []
        for i in range(self.profile_slots):
            doms, meta = self._generate_schedule(i, count=min(self.domains_per_schedule, 24))
            generated += int(meta["generated_raw"])
            valid += int(meta["valid"])
            invalid += int(meta["invalid"])
            unique.update(doms)
            tested.append(
                {
                    "slot": i,
                    "date": meta["date"],
                    "nr_b": meta["nr_b"],
                    "r": meta["r"],
                    "valid": meta["valid"],
                    "invalid": meta["invalid"],
                }
            )
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
            "tested_parameter_schedules": tested[:32],
            "tested_nr_b_values": self.nr_b_values,
            "tested_r_seeds": self.r_seeds,
            "estimated_max_unique_capacity": est,
            "recommended_near_capacity_ratio": 0.9,
            "structured_finite_combination": True,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="corebot_dedicated")

        out: list[str] = []
        start_offset = self.next_schedule_offset
        attempts = 0
        schedules: list[dict[str, Any]] = []
        invalid_raw_total = 0
        invalid_reason_totals: dict[str, int] = {}

        while len(out) < batch_size:
            slot = self.next_schedule_offset
            take = min(self.domains_per_schedule, batch_size - len(out))
            doms, meta = self._generate_schedule(slot, count=take)
            out.extend(doms[:take])
            invalid_raw_total += int(meta["invalid"])
            for rk, rv in (meta.get("invalid_reasons") or {}).items():
                invalid_reason_totals[rk] = int(invalid_reason_totals.get(rk, 0)) + int(rv)
            schedules.append({k: meta[k] for k in ("date", "year", "month", "day", "nr_b", "r", "valid", "invalid")})
            self.next_schedule_offset += 1
            attempts += 1
            if attempts > max(8, batch_size // max(self.domains_per_schedule, 1) + 4):
                break

        out = out[:batch_size]
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "date_nr_b_r_schedule",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["year", "month", "day", "nr_b", "r"],
            "tested_nr_b_values": self.nr_b_values,
            "tested_r_seeds": self.r_seeds,
            "start_schedule_offset": start_offset,
            "next_schedule_offset": self.next_schedule_offset,
            "schedules_used": schedules,
            "invalid_diagnostics": {"invalid_generated_raw": invalid_raw_total, **invalid_reason_totals},
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "remaining_unique_capacity": remaining_est,
            "finite_space": True,
            "structured_space": True,
        }

        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            adapter_type="corebot_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["year", "month", "day", "nr_b", "r"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": True,
            "supported_parameter_axes": ["year", "month", "day", "nr_b", "r"],
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
            "recommended_saturation_sensitivity": 0.45,
            "apparent_finite_space": True,
            "generation_mode": "date_nr_b_r_schedule",
            "adapter_type": "corebot_dedicated",
            "tested_parameter_schedules": self.profile_data["tested_parameter_schedules"],
            "tested_nr_b_values": self.profile_data["tested_nr_b_values"],
            "tested_r_seeds": self.profile_data["tested_r_seeds"],
            "collision_curve": self.profile_data["collision_curve"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
        }
        self._profile_cache = p
        return p
