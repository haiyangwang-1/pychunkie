"""Neumann Laplace BVPs on a smooth unit circle.

Run from the repository root:

    uv run python examples/smooth_laplace_neumann.py

For a single-layer representation, the Neumann equation is

    (jump * I + S') sigma = du/dn,

with jump = +1/2 for the interior side and -1/2 for the exterior side.
"""

from __future__ import annotations

import numpy as np
from _smooth_laplace_common import (
    EXTERIOR_TARGETS,
    INTERIOR_TARGETS,
    boundary_nodes,
    boundary_normals,
    boundary_weights,
    exterior_solution,
    interior_solution,
    make_circle,
    single_layer_values,
)

from chunkie import chunkermat, kernel


def smooth_kprime_matrix(boundary) -> np.ndarray:
    kprime = chunkermat(boundary, kernel("lap", "sp"))
    weights = boundary_weights(boundary)
    curvature = boundary.signed_curvature().T.reshape(-1)
    kprime[np.diag_indices_from(kprime)] = -curvature * weights / (4.0 * np.pi)
    return kprime


def solve_neumann(boundary, kprime: np.ndarray, normal_values: np.ndarray, side: str) -> np.ndarray:
    jump = 0.5 if side == "interior" else -0.5
    system = jump * np.eye(boundary.npt) + kprime
    return np.linalg.solve(system, normal_values)


def main() -> None:
    boundary = make_circle()
    nodes = boundary_nodes(boundary)
    normals = boundary_normals(boundary)
    kprime = smooth_kprime_matrix(boundary)

    sigma_int = solve_neumann(boundary, kprime, normals[0], "interior")
    interior_vals = single_layer_values(boundary, sigma_int, 0.0, INTERIOR_TARGETS)
    interior_vals += INTERIOR_TARGETS[0, 0] - interior_vals[0]

    sigma_ext = solve_neumann(boundary, kprime, -nodes[0], "exterior")
    exterior_vals = single_layer_values(boundary, sigma_ext, 0.0, EXTERIOR_TARGETS)

    print(f"unit circle: {boundary.nch} chunks, {boundary.npt} nodes")
    print(
        f"interior Neumann max error: {np.max(np.abs(interior_vals - interior_solution(INTERIOR_TARGETS))):.3e}"
    )
    print(
        f"exterior Neumann max error: {np.max(np.abs(exterior_vals - exterior_solution(EXTERIOR_TARGETS))):.3e}"
    )


if __name__ == "__main__":
    main()
