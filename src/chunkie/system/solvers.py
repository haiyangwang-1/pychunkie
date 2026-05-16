"""System solve policy."""

from __future__ import annotations

import numpy as np

from .config import SystemConfig
from .solution import SystemSolution


def solve_system(system, *, config: SystemConfig) -> SystemSolution:
    matrix = system.assemble(config=config)
    rhs = np.zeros(matrix.shape[0])
    vector = matrix.solve(rhs)
    return SystemSolution(system=system, operator=matrix, densities={}, constants={}, residual=matrix.matvec(vector) - rhs)
