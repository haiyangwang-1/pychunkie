"""System solve policy."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .assembly import rhs_vector, unknown_column_slices
from .backends.flam import factor_system
from .config import SystemConfig
from .density import Density
from .solution import SystemSolution


def solve_system(system, *, config: SystemConfig) -> SystemSolution:
    matrix = system.assemble(config=config)
    rhs = rhs_vector(system)
    diagnostics: dict[str, object] = {"solve_method": config.solve_method}
    if config.solve_method == "flam":
        if len(system.unknowns) != 1:
            raise NotImplementedError(
                "FLAM solve integration currently supports one unknown density"
            )
        unknown = system.unknowns[0]
        factor = factor_system(
            matrix,
            _solver_points(unknown),
            occupancy=config.flam_occupancy,
            tolerance=config.flam_tolerance,
        )
        vector = factor.solve(rhs)
        diagnostics["backend"] = "flam"
    elif config.solve_method == "gmres":
        vector, gmres_info, iteration_count = _gmres_solve(matrix, rhs, config)
        if gmres_info != 0:
            raise RuntimeError(f"GMRES failed to converge, info={gmres_info}")
        diagnostics["backend"] = "scipy.sparse.linalg.gmres"
        diagnostics["iterations"] = iteration_count
        diagnostics["gmres_info"] = gmres_info
    else:
        vector = matrix.solve(rhs)
        diagnostics["backend"] = "numpy.linalg.solve"
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
        diagnostics=diagnostics,
    )


def _gmres_solve(matrix, rhs, config: SystemConfig):
    iteration_count = 0

    def callback(_residual) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator(
        matrix.shape,
        matvec=lambda vector: matrix.matvec(vector),
        dtype=np.asarray(matrix.to_dense()).dtype,
    )
    # GMRES is introduced first as a dense-reference solve policy. Matrix-free
    # and FMM-backed operators can reuse the same solver boundary once their
    # correction and RCIP paths are complete.
    vector, info = gmres(
        operator,
        rhs,
        rtol=config.tolerance,
        atol=0.0,
        maxiter=config.max_iterations,
        callback=callback,
        callback_type="pr_norm",
    )
    return vector, int(info), iteration_count


def _solver_points(unknown) -> object:
    if unknown.component_count != 1:
        raise NotImplementedError("FLAM solve integration currently supports scalar unknowns")
    if not hasattr(unknown.geometry, "pointinfo"):
        raise TypeError("FLAM solve integration requires geometry pointinfo")
    return unknown.geometry.pointinfo.flat_positions
