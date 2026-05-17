"""Solved density container."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SystemSolution:
    system: object
    operator: object | None
    densities: Mapping[str, object]
    constants: Mapping[str, complex] = field(default_factory=dict)
    residual: NDArray[np.generic] = field(default_factory=lambda: np.zeros(0))
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def evaluate(self, targets, *, field: str = "u", config=None):
        from .evaluation import evaluate_solution

        return evaluate_solution(self, targets, field=field, config=config)
