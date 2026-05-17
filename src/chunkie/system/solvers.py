"""System solve policy."""

from __future__ import annotations

from .assembly import rhs_vector
from .backends.flam import factor_system
from .config import SystemConfig
from .density import Density
from .solution import SystemSolution


def solve_system(system, *, config: SystemConfig) -> SystemSolution:
    matrix = system.assemble(config=config)
    rhs = rhs_vector(system)
    unknown = system.unknowns[0]
    if config.solve_method == "flam":
        factor = factor_system(
            matrix,
            _solver_points(unknown),
            occupancy=config.flam_occupancy,
            tolerance=config.flam_tolerance,
        )
        vector = factor.solve(rhs)
    else:
        vector = matrix.solve(rhs)
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


def _solver_points(unknown) -> object:
    if unknown.component_count != 1:
        raise NotImplementedError("FLAM solve integration currently supports scalar unknowns")
    if not hasattr(unknown.geometry, "pointinfo"):
        raise TypeError("FLAM solve integration requires geometry pointinfo")
    return unknown.geometry.pointinfo.flat_positions
