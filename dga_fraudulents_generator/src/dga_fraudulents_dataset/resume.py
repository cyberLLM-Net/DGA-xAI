from __future__ import annotations

from pathlib import Path

from .models import GenerationRuntimeState
from .utils import read_json, write_json


def save_state(path: Path, state: GenerationRuntimeState) -> None:
    write_json(path, state.to_dict())


def load_state(path: Path) -> GenerationRuntimeState:
    return GenerationRuntimeState.from_dict(read_json(path))
