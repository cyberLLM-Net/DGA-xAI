from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InsertStats:
    attempted: int
    inserted: int


class SQLiteDedupStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self.conn.execute("PRAGMA cache_size=-200000;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS domains (
                domain TEXT PRIMARY KEY,
                algorithm_code TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self.conn.commit()

    def insert_many(self, rows: list[tuple[str, str]]) -> InsertStats:
        attempted = len(rows)
        before = self.conn.total_changes
        self.conn.executemany(
            "INSERT OR IGNORE INTO domains(domain, algorithm_code) VALUES (?, ?)",
            rows,
        )
        inserted = self.conn.total_changes - before
        return InsertStats(attempted=attempted, inserted=inserted)

    def commit(self) -> None:
        self.conn.commit()

    def count_unique(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM domains")
        return int(cur.fetchone()[0])

    def count_by_algorithm(self) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT algorithm_code, COUNT(*) FROM domains GROUP BY algorithm_code"
        )
        return {row[0]: int(row[1]) for row in cur.fetchall()}

    def stream_domains(self, batch_size: int = 100_000):
        cur = self.conn.cursor()
        cur.execute("SELECT domain, algorithm_code FROM domains ORDER BY rowid")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield rows

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
