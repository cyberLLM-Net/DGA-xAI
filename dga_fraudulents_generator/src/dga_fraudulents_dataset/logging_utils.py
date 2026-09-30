from __future__ import annotations

import json
import logging
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str, log_file: Path) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(level.upper())
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level.upper())
    fh.setFormatter(JsonFormatter())

    root.addHandler(ch)
    root.addHandler(fh)
