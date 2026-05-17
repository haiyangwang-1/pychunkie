"""System solve policy."""

from __future__ import annotations

from .assembly import rhs_vector, unknown_column_slices
from .backends.flam import factor_system
from .config import SystemConfig
from .density import Density
from .solution import SystemSolution


def solve_system(system, *, config: SystemConfig) -> SystemSolution:
    matrix = system.assemble(config=config)
    rhs = rhs_vector(system)
    if config.solve_method == "flam":
        if len(system.unknowns) != 1:
            raise NotImplementedError("FLAM solve integration currently supports one unknown density")
        unknown = system.unknowns[0]
        factor = factor_system(
            matrix,
            _solver_points(unknown),
            occupancy=config.flam_occupancy,
            tolerance=config.flam_tolerance,
        )
        vector = factor.solve(rhs)
    else:
        vector = matrix.solve(rhs)
    column_slices = unknown_column_slices(system.unknowns)
    densities = {
        unknown.name: Density.from_vector(
            unknown.name,
            unknown.geometry,
            vector[column_slices[unknown.name]],
            component_count=unknown.component_count,
        )
        for unknown in system.unknowns
    }
    return SystemSolution(
        system=system,
        operator=matrix,
        densities=densities,
        constants={},
        residual=matrix.matvec(vector) - rhs,
    )


def _solver_points(unknown) -> object:
    if unknown.component_count != 1:
        raise NotImplementedError("FLAM solve integration currently supports scalar unknowns")
    if not hasattr(unknown.geometry, "pointinfo"):
        raise TypeError("FLAM solve integration requires geometry pointinfo")
    return unknown.geometry.pointinfo.flat_positions
