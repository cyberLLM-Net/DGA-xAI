from __future__ import annotations

import csv
from pathlib import Path

from .dedup import SQLiteDedupStore


def export_csv(store: SQLiteDedupStore, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "algorithm_code"])
        for rows in store.stream_domains():
            w.writerows(rows)
            total += len(rows)
    return total
