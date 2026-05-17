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
        factor = factor_system(
            matrix,
            _solver_points(system.unknowns),
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


def _solver_points(unknowns) -> object:
    point_blocks = []
    for unknown in unknowns:
        if unknown.component_count != 1:
            raise NotImplementedError("FLAM solve integration currently supports scalar unknowns")
        if not hasattr(unknown.geometry, "pointinfo"):
            raise TypeError("FLAM solve integration requires geometry pointinfo")
        # FLAM sees the same dense solver-vector order as the matrix columns.
        # Multiple scalar unknowns therefore concatenate repeated geometry
        # point clouds in unknown-block order.
        point_blocks.append(unknown.geometry.pointinfo.flat_positions)
    if len(point_blocks) == 1:
        return point_blocks[0]

    base_dim = max(block.shape[0] for block in point_blocks)
    spans = [
        float(np.max(block, initial=0.0) - np.min(block, initial=0.0))
        for block in point_blocks
        if block.size
    ]
    block_spacing = (max(spans) if spans else 1.0) + 1.0
    lifted_blocks = []
    for block_id, block in enumerate(point_blocks):
        lifted = np.zeros((base_dim + 1, block.shape[1]), dtype=float)
        lifted[: block.shape[0]] = block
        # Repeated unknowns on the same geometry would otherwise give FLAM
        # duplicate points. The extra coordinate is a backend-only block axis.
        lifted[-1] = block_id * block_spacing
        lifted_blocks.append(lifted)
    return np.concatenate(lifted_blocks, axis=1)
