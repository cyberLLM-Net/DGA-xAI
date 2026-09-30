from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def open_csv_writer(path: Path) -> tuple[object, csv.writer]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(["domain"])
    return handle, writer


def write_domains(writer: csv.writer, domains: Iterable[str]) -> int:
    count = 0
    for domain in domains:
        writer.writerow([domain])
        count += 1
    return count
