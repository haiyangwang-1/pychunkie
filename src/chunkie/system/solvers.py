"""System solve policy."""

from __future__ import annotations

from .assembly import rhs_vector
from .config import SystemConfig
from .density import Density
from .solution import SystemSolution


def solve_system(system, *, config: SystemConfig) -> SystemSolution:
    matrix = system.assemble(config=config)
    rhs = rhs_vector(system)
    vector = matrix.solve(rhs)
    unknown = system.unknowns[0]
    density = Density.from_vector(
        unknown.name,
        unknown.geometry,
        vector,
        component_count=unknown.component_count,
    )
    return SystemSolution(
        system=system,
        operator=matrix,
        densities={unknown.name: density},
        constants={},
        residual=matrix.matvec(vector) - rhs,
    )
