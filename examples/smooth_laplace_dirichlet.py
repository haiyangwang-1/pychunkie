"""Dirichlet Laplace BVPs on a smooth unit circle.

Run from the repository root:

    uv run python examples/smooth_laplace_dirichlet.py

The example uses a single-layer representation,

    S sigma + c = g,    integral sigma ds = 0,

which fixes the 2D Laplace single-layer compatibility constraint.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkermat, kernel

from _smooth_laplace_common import (
    EXTERIOR_TARGETS,
    INTERIOR_TARGETS,
    boundary_nodes,
    exterior_solution,
    interior_solution,
    make_circle,
    single_layer_values,
)


def solve_dirichlet(s_mat: np.ndarray, weights: np.ndarray, boundary_values: np.ndarray):
    ones = np.ones((s_mat.shape[0], 1))
    system = np.block([[s_mat, ones], [weights[None, :], np.zeros((1, 1))]])
    rhs = np.concatenate((boundary_values, [0.0]))
    sol = np.linalg.solve(system, rhs)
    return sol[:-1], float(sol[-1])


def run_case(title: str, chnkr, s_mat: np.ndarray, weights: np.ndarray, targets: np.ndarray, truth_fn) -> None:
    sigma, const = solve_dirichlet(s_mat, weights, truth_fn(boundary_nodes(chnkr)))
    vals = single_layer_values(chnkr, sigma, const, targets)
    error = np.max(np.abs(vals - truth_fn(targets)))
    print(f"{title} max error: {error:.3e}")


def main() -> None:
    chnkr = make_circle()
    weights = chnkr.wts.reshape(-1, order="F")
    s_mat = chunkermat(chnkr, kernel("lap", "s"))

    print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
    run_case("interior Dirichlet", chnkr, s_mat, weights, INTERIOR_TARGETS, interior_solution)
    run_case("exterior Dirichlet", chnkr, s_mat, weights, EXTERIOR_TARGETS, exterior_solution)


if __name__ == "__main__":
    main()
