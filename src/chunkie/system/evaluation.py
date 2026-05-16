"""Solution field evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FieldResult:
    points: object
    values: NDArray[np.generic]
    field: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def evaluate_solution(solution, targets, *, field: str, config=None) -> FieldResult:
    raise NotImplementedError("Solution evaluation follows dense system assembly")
