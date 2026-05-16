"""Dense system assembly."""

from __future__ import annotations

import numpy as np

from .config import SystemConfig
from .matrix import SystemMatrix


def assemble_system_matrix(system, *, config: SystemConfig) -> SystemMatrix:
    raise NotImplementedError("System assembly follows after geometry, kernels, and quadrature baselines")


def identity_system_matrix(size: int, *, config: SystemConfig | None = None) -> SystemMatrix:
    return SystemMatrix(np.eye(size), SystemConfig() if config is None else config)
