"""Assembled system matrix wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SystemConfig


@dataclass(frozen=True)
class SystemMatrix:
    data: NDArray[np.generic]
    config: SystemConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def to_dense(self) -> NDArray[np.generic]:
        return np.asarray(self.data)

    def matvec(self, vector: NDArray[np.generic]) -> NDArray[np.generic]:
        return self.data @ vector

    def solve(self, rhs: NDArray[np.generic]) -> NDArray[np.generic]:
        return np.linalg.solve(self.data, rhs)
