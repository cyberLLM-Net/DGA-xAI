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
class SeedState:
    seed: int
    r: int


class NgiowebAdapter(AlgorithmAdapter):
    """Dedicated adapter for ngioweb RNG-object based generation."""

    DEFAULT_SEEDS = [0x56EDC15, 0x5397FB1, 0x01275C63, 0x04BC65BC, 0x00375D5A]

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}

        self.module = self._load_module(Path(inspection.path) / "dga.py")
        self.dga_fn = getattr(self.module, "dga", None)
        self.rand_cls = getattr(self.module, "Rand", None)
        if self.dga_fn is None or not callable(self.dga_fn):
            raise RuntimeError("ngioweb:missing_implementation:callable_dga_not_found")
        if self.rand_cls is None or not callable(self.rand_cls):
            raise RuntimeError("ngioweb:missing_implementation:Rand_class_not_found")

        self.seed_pool = self._resolve_seed_pool()
        self.seed_states = self._resolve_seed_states()
        self.next_counter = self._resolve_resume_counter()
        self.profile_slots = max(64, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 768))))
        self._profile_cache: dict[str, Any] | None = None

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"ngioweb:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_ngioweb", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"ngioweb:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _parse_seed(self, raw: Any) -> int:
        if isinstance(raw, int):
            return raw & 0xFFFFFFFF
        txt = str(raw).strip().lower()
        if txt.startswith("0x"):
            return int(txt, 16) & 0xFFFFFFFF
        if all(ch in "0123456789abcdef" for ch in txt):
            return int(txt, 16) & 0xFFFFFFFF
        return int(txt) & 0xFFFFFFFF

    def _resolve_seed_pool(self) -> list[int]:
        raw = self.defaults.get("ngioweb_seeds", self.strategy_cfg.get("ngioweb_seeds"))
        if raw is None:
            return list(self.DEFAULT_SEEDS)
        if isinstance(raw, str):
            vals = [x.strip() for x in raw.split(",") if x.strip()]
            out = [self._parse_seed(x) for x in vals]
            return out or list(self.DEFAULT_SEEDS)
        if isinstance(raw, (list, tuple, set)):
            out = [self._parse_seed(x) for x in raw]
            return out or list(self.DEFAULT_SEEDS)
        return [self._parse_seed(raw)]

    def _resolve_seed_states(self) -> dict[int, SeedState]:
        state_map: dict[int, SeedState] = {}
        raw = self.strategy_cfg.get("resume_seed_states") or self.defaults.get("resume_seed_states") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    seed = self._parse_seed(k)
                    r_val = self._parse_seed(v)
                    state_map[seed] = SeedState(seed=seed, r=r_val)
                except Exception:
                    continue
        for s in self.seed_pool:
            if s not in state_map:
                state_map[s] = SeedState(seed=s, r=s)
        return state_map

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
    def _classify_rng_error(exc: Exception) -> str:
        txt = str(exc).lower()
        if "no attribute 'rand'" in txt or "has no attribute 'rand'" in txt:
            return "expected_random_object"
        if "rand" in txt and "attribute" in txt:
            return "rng_type_mismatch"
        return "invalid_seed_normalization"

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.2
        elif tail_yield <= 0.08:
            factor = 1.45
        elif tail_yield <= 0.2:
            factor = 1.9
        elif tail_yield <= 0.4:
            factor = 2.6
        else:
            factor = 3.4
        return max(sample_unique, int(sample_unique * factor))

    def _seed_state_list(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for seed in sorted(self.seed_states):
            st = self.seed_states[seed]
            out[f"0x{seed:08x}"] = f"0x{st.r:08x}"
        return out

    def _invoke_with_state(self, seed: int, r_state: int) -> tuple[str | None, dict[str, Any], str | None, int]:
        raw_params = {"seed": seed}
        normalized: dict[str, Any] = {}
        try:
            rng = self.rand_cls(seed)
            if hasattr(rng, "r"):
                setattr(rng, "r", r_state & 0xFFFFFFFF)
            normalized = {
                "rng_type": type(rng).__name__,
                "seed_int": int(seed),
                "seed_hex": f"0x{seed:08x}",
                "rng_state_before": f"0x{(getattr(rng, 'r', seed) & 0xFFFFFFFF):08x}",
            }
            out = self.dga_fn(rng)
            after = int(getattr(rng, "r", r_state)) & 0xFFFFFFFF
            normalized["rng_state_after"] = f"0x{after:08x}"
            return str(out) if out is not None else None, {
                "raw": raw_params,
                "normalized": normalized,
                "types_before": {"seed": type(seed).__name__},
                "types_after": {"rng": type(rng).__name__},
            }, None, after
        except Exception as exc:
            return None, {
                "raw": raw_params,
                "normalized": normalized,
                "types_before": {"seed": type(seed).__name__},
                "types_after": {"rng": normalized.get("rng_type", "unknown")},
            }, self._classify_rng_error(exc), r_state

    def _build_profile_data(self) -> dict[str, Any]:
        unique: set[str] = set()
        generated = 0
        valid = 0
        invalid = 0
        errs: dict[str, int] = {}
        curve: list[dict[str, Any]] = []
        schedule: list[dict[str, Any]] = []

        tmp = {k: SeedState(seed=v.seed, r=v.r) for k, v in self.seed_states.items()}
        for i in range(self.profile_slots):
            seed = self.seed_pool[i % len(self.seed_pool)]
            st = tmp[seed]
            dom, diag, err, next_r = self._invoke_with_state(seed, st.r)
            st.r = next_r
            generated += 1
            if i < 32:
                schedule.append(diag)
            if err:
                invalid += 1
                errs[err] = int(errs.get(err, 0)) + 1
                continue
            vr = validate_domain(dom or "")
            if vr.is_valid and vr.normalized:
                valid += 1
                unique.add(vr.normalized)
            else:
                invalid += 1
            if (i + 1) % max(8, self.profile_slots // 8) == 0:
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
        est = self._estimate_capacity(len(unique), tail)
        return {
            "sample_generated": valid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "valid_rate": valid_rate,
            "invalid_rate": 1.0 - valid_rate,
            "tail_unique_yield": tail,
            "collision_curve": curve,
            "type_mismatch_diagnostics": errs,
            "tested_parameter_schedule": schedule,
            "estimated_max_unique_capacity": est,
            "recommended_near_capacity_ratio": 0.92,
            "structured_finite_combination": False,
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="ngioweb_dedicated")

        out: list[str] = []
        attempts = 0
        errors: list[str] = []
        schedules: list[dict[str, Any]] = []
        invalid_diag: dict[str, int] = {}
        start_counter = self.next_counter

        while len(out) < batch_size:
            idx = self.next_counter
            seed = self.seed_pool[idx % len(self.seed_pool)]
            st = self.seed_states[seed]
            dom, diag, err, next_r = self._invoke_with_state(seed, st.r)
            st.r = next_r
            self.next_counter += 1
            attempts += 1
            if len(schedules) < 32:
                schedules.append(diag)
            if err:
                errors.append(
                    f"{err}:seed={diag.get('raw', {}).get('seed')}:{diag.get('types_before', {}).get('seed')}"
                )
                invalid_diag[err] = int(invalid_diag.get(err, 0)) + 1
                if attempts > batch_size * 8:
                    break
                continue
            vr = validate_domain(dom or "")
            if vr.is_valid and vr.normalized:
                out.append(vr.normalized)
            else:
                reason = vr.reason or "format"
                invalid_diag[reason] = int(invalid_diag.get(reason, 0)) + 1
            if attempts > batch_size * 8:
                break

        out = out[:batch_size]
        eff = {
            "generation_mode": "rng_object_seed_normalized",
            "implementation_file": str(Path(self.inspection.path) / "dga.py"),
            "selected_parameter_axes": ["seed", "counter", "rng_state"],
            "seed_pool": [f"0x{s:08x}" for s in self.seed_pool],
            "start_counter": start_counter,
            "next_counter": self.next_counter,
            "raw_seed_type": "int",
            "normalized_parameter_types": {"rng": "Rand"},
            "seed_states": self._seed_state_list(),
            "schedule_used": schedules,
            "invalid_diagnostics": invalid_diag,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "failure_classification": {
                "rng_type_mismatch": "rng_type_mismatch",
                "expected_random_object": "expected_random_object",
                "invalid_seed_normalization": "invalid_seed_normalization",
            },
        }
        return BatchGenerationResult(
            domains=out,
            attempts=attempts,
            generated=len(out),
            errors=errors[-5:],
            adapter_type="ngioweb_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["seed", "counter", "rng_state"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": True,
            "date_sensitive": False,
            "supported_parameter_axes": ["seed", "counter", "rng_state"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "valid_rate": self.profile_data["valid_rate"],
            "invalid_rate": self.profile_data["invalid_rate"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(2.0, max(0.1, self.profile_data["tail_unique_yield"] * 2.2)),
            "expected_diversity_score": min(1.0, max(0.08, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 2.5,
            "recommended_saturation_sensitivity": 0.4,
            "apparent_finite_space": False,
            "generation_mode": "rng_object_seed_normalized",
            "adapter_type": "ngioweb_dedicated",
            "collision_curve": self.profile_data["collision_curve"],
            "tested_parameter_schedule": self.profile_data["tested_parameter_schedule"],
            "type_mismatch_diagnostics": self.profile_data["type_mismatch_diagnostics"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "structured_finite_combination": False,
            "failure_classification": {
                "rng_type_mismatch": "rng_type_mismatch",
                "expected_random_object": "expected_random_object",
                "invalid_seed_normalization": "invalid_seed_normalization",
            },
        }
        self._profile_cache = p
        return p
