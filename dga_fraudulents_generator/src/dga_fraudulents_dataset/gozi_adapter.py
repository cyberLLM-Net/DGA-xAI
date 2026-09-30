from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .validation import validate_domain


class GoziAdapter(AlgorithmAdapter):
    """Dedicated adapter for gozi with explicit lexical resource handling."""

    EPOCH = datetime.strptime("2015-01-01", "%Y-%m-%d")

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str, date_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy
        self.date_strategy = date_strategy
        self.defaults = inspection.default_params or {}
        self.strategy_cfg = inspection.parameter_strategy or {}
        self.algo_dir = Path(inspection.path)

        self.module = self._load_module(self.algo_dir / "dga.py")
        self.seeds_cfg = self._resolve_seeds_cfg()
        self.base_date = self._resolve_base_date()
        self.day_window = max(60, int(self.defaults.get("day_window", self.strategy_cfg.get("day_window", 720))))
        self.profile_slots = max(24, int(self.defaults.get("profile_slots", self.strategy_cfg.get("profile_slots", 120))))

        self.wordlists_requested = self._resolve_wordlists_requested()
        self.word_resources, self.wordlist_errors = self._resolve_resources()
        self.wordlists_accepted = sorted(self.word_resources.keys())
        self.next_slot = self._resolve_resume_slot()

        if not self.wordlists_accepted:
            diagnostics = ", ".join(f"{k}:{v}" for k, v in sorted(self.wordlist_errors.items()))
            raise RuntimeError(f"gozi:missing_resource_file:no_accepted_wordlists:{diagnostics}")

        self.profile_data = self._build_profile_data()
        self.estimated_max_unique_capacity = int(self.profile_data["estimated_max_unique_capacity"])
        self.emitted_unique: set[str] = set()
        self._profile_cache: dict[str, Any] | None = None

    def _load_module(self, path: Path) -> ModuleType:
        if not path.exists():
            raise RuntimeError(f"gozi:missing_implementation:{path}")
        spec = importlib.util.spec_from_file_location("dga_fraudulents_dataset_gozi", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"gozi:invalid_resource_format:cannot_import:{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_seeds_cfg(self) -> dict[str, dict[str, Any]]:
        seeds = getattr(self.module, "seeds", None)
        if not isinstance(seeds, dict) or not seeds:
            raise RuntimeError("gozi:invalid_resource_format:missing_seeds_dict")
        out: dict[str, dict[str, Any]] = {}
        for k, v in seeds.items():
            try:
                key = str(k)
                out[key] = {"div": int(v["div"]), "tld": str(v["tld"]), "nr": int(v.get("nr", 12))}
            except Exception:
                continue
        if not out:
            raise RuntimeError("gozi:invalid_resource_format:invalid_seeds_entries")
        return out

    def _resolve_base_date(self) -> datetime:
        raw = self.defaults.get("base_date", self.strategy_cfg.get("base_date", "2020-01-01"))
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return datetime(2020, 1, 1)

    def _resolve_wordlists_requested(self) -> list[str]:
        raw = self.defaults.get("gozi_wordlists", self.strategy_cfg.get("gozi_wordlists", "all"))
        if isinstance(raw, str):
            txt = raw.strip().lower()
            if txt in {"all", "*"}:
                return sorted(self.seeds_cfg.keys())
            return [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, (list, tuple, set)):
            vals = [str(x).strip() for x in raw if str(x).strip()]
            return vals or sorted(self.seeds_cfg.keys())
        return sorted(self.seeds_cfg.keys())

    def _resolve_resources(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        resources: dict[str, list[str]] = {}
        errors: dict[str, str] = {}
        for name in self.wordlists_requested:
            if name not in self.seeds_cfg:
                errors[name] = "invalid_resource_format:unknown_wordlist"
                continue
            p = self.algo_dir / name
            if not p.exists():
                errors[name] = f"missing_resource_file:{p}"
                continue
            try:
                words = [w.strip() for w in p.read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()]
            except Exception as exc:
                errors[name] = f"invalid_resource_format:{exc}"
                continue
            if not words:
                errors[name] = "invalid_resource_format:empty_wordlist"
                continue
            resources[name] = words
        return resources, errors

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

    def _resource_deps(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in sorted(self.wordlists_requested):
            out.append(
                {
                    "resource_name": name,
                    "resource_path": str(self.algo_dir / name),
                    "resolved": name in self.word_resources,
                    "error": self.wordlist_errors.get(name),
                    "expected_type": "path_or_wordlist_name",
                    "adapter_passed_type": "str",
                }
            )
        return out

    def _lcg(self, value: int) -> int:
        return (1664525 * value + 1013904223) & 0xFFFFFFFF

    def _generate_for_wordlist_date(self, when: datetime, wordlist_name: str) -> list[str]:
        cfg = self.seeds_cfg[wordlist_name]
        words = self.word_resources[wordlist_name]

        days_passed = (when - self.EPOCH).days // int(cfg["div"])
        seed = ((1 << 16) + days_passed - 306607824) & 0xFFFFFFFF
        r = seed

        out: list[str] = []
        for _ in range(12):
            r = self._lcg(r)
            v = self._lcg(r)
            r = v
            length = v % 12 + 12
            domain = ""
            guard = 0
            while len(domain) < length and guard < 128:
                guard += 1
                r = self._lcg(r)
                idx = r % len(words)
                word = words[idx]
                r = self._lcg(r)
                l = len(word)
                if (r % 3) == 0:
                    l >>= 1
                if l > 0 and len(domain) + l <= 24:
                    domain += word[:l]
            out.append(f"{domain}{cfg['tld']}")
        return out

    def _slot_to_schedule(self, slot: int) -> tuple[str, datetime]:
        wl = self.wordlists_accepted[slot % len(self.wordlists_accepted)]
        day_offset = ((slot * 23) + 7) % self.day_window
        when = self.base_date + timedelta(days=day_offset)
        return wl, when

    @staticmethod
    def _estimate_capacity(sample_unique: int, tail_yield: float) -> int:
        if tail_yield <= 0.03:
            factor = 1.08
        elif tail_yield <= 0.08:
            factor = 1.2
        elif tail_yield <= 0.18:
            factor = 1.35
        elif tail_yield <= 0.35:
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
        schedules: list[dict[str, Any]] = []
        per_wordlist: dict[str, dict[str, Any]] = {w: {"generated": 0, "valid": 0, "unique": set()} for w in self.wordlists_accepted}

        for i in range(self.profile_slots):
            wl, when = self._slot_to_schedule(i)
            domains = self._generate_for_wordlist_date(when, wl)
            schedules.append({"slot": i, "wordlist": wl, "date": when.date().isoformat()})
            for d in domains:
                generated += 1
                per_wordlist[wl]["generated"] += 1
                vr = validate_domain(d)
                if vr.is_valid and vr.normalized:
                    valid += 1
                    per_wordlist[wl]["valid"] += 1
                    unique.add(vr.normalized)
                    per_wordlist[wl]["unique"].add(vr.normalized)
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

        per_wl_out = []
        for w, data in sorted(per_wordlist.items()):
            valid_w = int(data["valid"])
            unique_w = len(data["unique"])
            per_wl_out.append(
                {
                    "wordlist": w,
                    "resource_path": str(self.algo_dir / w),
                    "generated": int(data["generated"]),
                    "valid": valid_w,
                    "unique": unique_w,
                    "unique_yield": unique_w / max(valid_w, 1),
                }
            )

        return {
            "sample_generated": valid,
            "sample_unique": len(unique),
            "sample_unique_yield": unique_yield,
            "valid_rate": valid_rate,
            "invalid_rate": 1.0 - valid_rate,
            "tail_unique_yield": tail,
            "collision_curve": curve,
            "per_wordlist": per_wl_out,
            "resource_dependencies": self._resource_deps(),
            "resource_diagnostics": self.wordlist_errors,
            "estimated_max_unique_capacity": est,
            "recommended_near_capacity_ratio": 0.9,
            "structured_finite_combination": True,
            "generation_mode": "resource_wordlist_date",
            "adapter_type": "gozi_dedicated",
        }

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="gozi_dedicated")

        out: list[str] = []
        start_slot = self.next_slot
        attempts = 0
        schedule_used: list[dict[str, Any]] = []
        errors: list[str] = []

        while len(out) < batch_size:
            slot = self.next_slot
            wl, when = self._slot_to_schedule(slot)
            schedule_used.append({"slot": slot, "wordlist": wl, "date": when.date().isoformat(), "wordlist_type": type(wl).__name__})
            try:
                domains = self._generate_for_wordlist_date(when, wl)
            except Exception as exc:
                errors.append(f"resource_type_mismatch:{wl}:{type(wl).__name__}:{exc}")
                self.next_slot += 1
                attempts += 1
                if attempts > batch_size * 3:
                    break
                continue
            for d in domains:
                vr = validate_domain(d)
                if vr.is_valid and vr.normalized:
                    out.append(vr.normalized)
                    if len(out) >= batch_size:
                        break
            self.next_slot += 1
            attempts += 1
            if attempts > batch_size * 3:
                break

        out = out[:batch_size]
        self.emitted_unique.update(out)
        remaining_est = max(0, self.estimated_max_unique_capacity - len(self.emitted_unique))

        eff = {
            "generation_mode": "resource_wordlist_date",
            "implementation_file": str(self.algo_dir / "dga.py"),
            "resource_dependencies": self._resource_deps(),
            "start_slot": start_slot,
            "next_slot": self.next_slot,
            "schedule_used": schedule_used[:32],
            "normalized_parameter_types": {"date": "datetime", "wordlist": "str"},
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
            adapter_type="gozi_dedicated",
            last_effective_params=eff,
            supported_parameter_axes=["date", "wordlist", "counter"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache
        p = {
            "generates_any": self.profile_data["sample_generated"] > 0,
            "mode": "batch",
            "seed_sensitive": False,
            "date_sensitive": True,
            "supported_parameter_axes": ["date", "wordlist", "counter"],
            "sample_generated": self.profile_data["sample_generated"],
            "sample_unique": self.profile_data["sample_unique"],
            "sample_unique_yield": self.profile_data["sample_unique_yield"],
            "valid_rate": self.profile_data["valid_rate"],
            "invalid_rate": self.profile_data["invalid_rate"],
            "tail_unique_yield": self.profile_data["tail_unique_yield"],
            "initial_health_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "initial_capacity_score": min(1.6, max(0.08, self.profile_data["tail_unique_yield"] * 2.0)),
            "expected_diversity_score": min(1.0, max(0.05, self.profile_data["sample_unique_yield"])),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.5,
            "apparent_finite_space": True,
            "generation_mode": "resource_wordlist_date",
            "adapter_type": "gozi_dedicated",
            "resource_dependencies": self.profile_data["resource_dependencies"],
            "resource_diagnostics": self.profile_data["resource_diagnostics"],
            "per_wordlist": self.profile_data["per_wordlist"],
            "collision_curve": self.profile_data["collision_curve"],
            "estimated_max_unique_capacity": self.profile_data["estimated_max_unique_capacity"],
            "observed_unique_capacity_estimate": self.profile_data["sample_unique"],
            "recommended_near_capacity_ratio": self.profile_data["recommended_near_capacity_ratio"],
            "structured_finite_combination": True,
        }
        self._profile_cache = p
        return p
