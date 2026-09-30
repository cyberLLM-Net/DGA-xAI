from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapter_base import AlgorithmAdapter, BatchGenerationResult
from .models import AlgorithmInspection
from .utils import stable_int_seed

ORCHARD_PATTERN = '{"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa":{"final_balance":FB,"n_tx":NTX,"total_received":FB}}'
ORCHARD_TLDS: tuple[str, ...] = (".com", ".net", ".org", ".duckdns.org")


@dataclass(frozen=True)
class OrchardDomainEntry:
    record_index: int
    base_label: str
    domain: str


class OrchardAdapter(AlgorithmAdapter):
    """Finite-space, dataset-backed adapter for orchard.

    The original orchard implementation derives domains from deterministic seeds
    built from blockchain record balances (db.json), then applies MD5 and fixed TLDs.
    """

    def __init__(self, inspection: AlgorithmInspection, seed_strategy: str) -> None:
        self.inspection = inspection
        self.seed_strategy = seed_strategy

        self.db_path = self._resolve_db_path()
        self.records = self._load_records(self.db_path)
        self.tld_variants = list(ORCHARD_TLDS)

        self.shuffle_records, self.shuffle_seed = self._resolve_shuffle_config()
        self.record_order = list(range(len(self.records)))
        if self.shuffle_records and self.record_order:
            random.Random(self.shuffle_seed).shuffle(self.record_order)

        self.entries = self._build_entries()
        self.estimated_max_unique_capacity = len(self.entries)
        self.distinct_base_labels = len({e.base_label for e in self.entries})

        self.next_domain_index = self._resolve_resume_domain_index()
        self._profile_cache: dict[str, Any] | None = None

    def _resolve_db_path(self) -> Path:
        defaults = self.inspection.default_params or {}
        strategy = self.inspection.parameter_strategy or {}
        raw = (
            defaults.get("db_path")
            or defaults.get("db_json")
            or strategy.get("db_path")
            or strategy.get("db_json")
            or "db.json"
        )
        p = Path(str(raw))
        if not p.is_absolute():
            p = Path(self.inspection.path) / p
        return p

    def _load_records(self, db_path: Path) -> list[dict[str, Any]]:
        if not db_path.exists():
            raise RuntimeError(f"orchard missing required resource db.json at {db_path}")
        try:
            payload = json.loads(db_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            raise RuntimeError(f"orchard failed to parse db.json at {db_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"orchard db.json must be a JSON object at {db_path}")
        records: list[dict[str, Any]] = []
        for tx_hash, tx in payload.items():
            if not isinstance(tx, dict):
                continue
            bal = tx.get("balance")
            try:
                bal_i = int(bal)
            except Exception:
                continue
            records.append({"tx_hash": str(tx_hash), "balance": bal_i, "tx": tx})
        if not records:
            raise RuntimeError(f"orchard db.json has no usable transaction records at {db_path}")
        records.sort(key=lambda x: x["balance"], reverse=True)
        return records

    def _resolve_shuffle_config(self) -> tuple[bool, int]:
        defaults = self.inspection.default_params or {}
        strategy = self.inspection.parameter_strategy or {}

        shuffle_val = defaults.get("shuffle_records", strategy.get("shuffle_records", False))
        shuffle_records = bool(shuffle_val)
        if isinstance(shuffle_val, str):
            shuffle_records = shuffle_val.strip().lower() in {"1", "true", "yes", "y", "on"}

        base = defaults.get("shuffle_seed", strategy.get("shuffle_seed"))
        if base is None:
            base = stable_int_seed(f"{self.inspection.algorithm_code}:orchard")
        try:
            shuffle_seed = int(base)
        except Exception:
            shuffle_seed = stable_int_seed(f"{self.inspection.algorithm_code}:orchard:fallback")

        return shuffle_records, shuffle_seed

    def _resolve_resume_domain_index(self) -> int:
        strategy = self.inspection.parameter_strategy or {}
        defaults = self.inspection.default_params or {}
        raw = (
            strategy.get("resume_next_domain_index")
            or strategy.get("next_domain_index")
            or defaults.get("resume_next_domain_index")
            or defaults.get("next_domain_index")
            or 0
        )
        try:
            idx = int(raw)
        except Exception:
            idx = 0
        return max(0, min(idx, self.estimated_max_unique_capacity))

    def _seed_for_record(self, sorted_index: int, balance: int) -> str:
        ntx = len(self.records) + 1
        return ORCHARD_PATTERN.replace("FB", str(balance)).replace("NTX", str(ntx - sorted_index - 1))

    def _derive_base_labels(self, seed: str) -> list[str]:
        md5 = hashlib.md5(seed.encode("ascii", errors="ignore")).hexdigest()
        return [md5[i : i + 8] for i in range(0, len(md5), 8)]

    def _build_entries(self) -> list[OrchardDomainEntry]:
        out: list[OrchardDomainEntry] = []
        seen_domains: set[str] = set()

        for idx in self.record_order:
            rec = self.records[idx]
            seed = self._seed_for_record(idx, rec["balance"])
            labels = self._derive_base_labels(seed)
            for label in labels:
                for tld in self.tld_variants:
                    domain = f"{label}{tld}"
                    if domain in seen_domains:
                        continue
                    seen_domains.add(domain)
                    out.append(
                        OrchardDomainEntry(
                            record_index=idx,
                            base_label=label,
                            domain=domain,
                        )
                    )
        return out

    def _record_index_for_domain_cursor(self, cursor: int) -> int:
        if not self.entries:
            return 0
        if cursor < 0:
            return self.entries[0].record_index
        if cursor >= len(self.entries):
            return len(self.records)
        return self.entries[cursor].record_index

    def _resource_deps(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "file",
                "path": str(self.db_path),
                "required": True,
                "description": "orchard blockchain transaction dataset",
            }
        ]

    def generate(self, batch_size: int) -> BatchGenerationResult:
        if batch_size <= 0:
            return BatchGenerationResult(domains=[], attempts=0, generated=0, adapter_type="orchard_dataset")

        start = self.next_domain_index
        end = min(start + batch_size, self.estimated_max_unique_capacity)
        domains = [entry.domain for entry in self.entries[start:end]]
        self.next_domain_index = end

        current_record_index = self._record_index_for_domain_cursor(start)
        next_record_index = self._record_index_for_domain_cursor(self.next_domain_index)
        remaining = max(0, self.estimated_max_unique_capacity - self.next_domain_index)

        params = {
            "generation_mode": "dataset_backed",
            "db_json_path": str(self.db_path),
            "source_records": len(self.records),
            "distinct_derived_base_labels": self.distinct_base_labels,
            "tld_variants": self.tld_variants,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "current_record_index": current_record_index,
            "next_record_index": next_record_index,
            "next_domain_index": self.next_domain_index,
            "remaining_unique_capacity": remaining,
            "shuffle_records": self.shuffle_records,
            "shuffle_seed": self.shuffle_seed,
            "resource_dependencies": self._resource_deps(),
            "finite_space": True,
        }

        return BatchGenerationResult(
            domains=domains,
            attempts=1,
            generated=len(domains),
            adapter_type="orchard_dataset",
            last_effective_params=params,
            supported_parameter_axes=["record_index"],
        )

    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        if self._profile_cache is not None:
            return self._profile_cache

        n = max(1, min(sample_size, self.estimated_max_unique_capacity))
        sample = [e.domain for e in self.entries[:n]]
        unique = len(set(sample))
        uniq_ratio = unique / max(len(sample), 1)

        self._profile_cache = {
            "generates_any": self.estimated_max_unique_capacity > 0,
            "mode": "batch",
            "seed_sensitive": self.shuffle_records,
            "date_sensitive": False,
            "supported_parameter_axes": ["record_index"],
            "sample_generated": len(sample),
            "sample_unique": unique,
            "sample_unique_yield": uniq_ratio,
            "initial_health_score": min(1.0, max(0.1, uniq_ratio)),
            "initial_capacity_score": min(2.0, max(0.2, uniq_ratio * 2.0)),
            "expected_diversity_score": min(1.0, max(0.1, uniq_ratio)),
            "recommended_max_effective_quota_multiplier": 1.0,
            "recommended_saturation_sensitivity": 0.9,
            "recommended_date_window": {"start": "n/a", "end": "n/a"},
            "apparent_finite_space": True,
            "generation_mode": "dataset_backed",
            "adapter_type": "orchard_dataset",
            "source_records": len(self.records),
            "distinct_derived_base_labels": self.distinct_base_labels,
            "tld_variants": self.tld_variants,
            "estimated_max_unique_capacity": self.estimated_max_unique_capacity,
            "resource_dependencies": self._resource_deps(),
            "derivation_logic": {
                "record_seed": "ORCHARD_PATTERN(balance, ntx_rank)",
                "label_derivation": "md5(seed) split into 4 chunks of 8 hex chars",
                "domain": "label + fixed_tld",
            },
        }
        return self._profile_cache
