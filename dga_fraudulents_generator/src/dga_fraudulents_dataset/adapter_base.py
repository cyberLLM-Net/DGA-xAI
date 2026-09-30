from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchGenerationResult:
    domains: list[str] = field(default_factory=list)
    attempts: int = 0
    generated: int = 0
    timed_out: bool = False
    errors: list[str] = field(default_factory=list)
    adapter_type: str = "unknown"
    last_effective_params: dict[str, Any] = field(default_factory=dict)
    supported_parameter_axes: list[str] = field(default_factory=list)
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    batch_aborted: bool = False
    abort_reason: str | None = None
    timeout_events: int = 0
    subprocess_calls: int = 0


class AlgorithmAdapter(ABC):
    @abstractmethod
    def generate(self, batch_size: int) -> BatchGenerationResult:
        raise NotImplementedError

    @abstractmethod
    def profile(self, sample_size: int = 64) -> dict[str, Any]:
        raise NotImplementedError
