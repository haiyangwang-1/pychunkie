"""Smooth Laplace Dirichlet/Neumann BVP demo on the unit circle.

Run from the repository root:

    uv run python examples/smooth_laplace_bvp.py

The manufactured harmonic solutions are u=x inside the disk and u=x/r^2
outside the disk. The demo forms the BIE systems explicitly so the jump terms
and 2D Laplace compatibility constraint are visible.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, kernel


def circle(t: np.ndarray):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def flat_nodes(chnkr):
    return chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F")


def single_layer_dirichlet(chnkr, boundary_values):
    """Solve S sigma + c = g with zero net charge for 2D Laplace."""

    lap_s = kernel("lap", "s")
    s_mat = chunkermat(chnkr, lap_s)
    weights = chnkr.wts.reshape(-1, order="F")
    ones = np.ones((chnkr.npt, 1))
    system = np.block([[s_mat, ones], [weights[None, :], np.zeros((1, 1))]])
    rhs = np.concatenate((boundary_values, [0.0]))
    sol = np.linalg.solve(system, rhs)
    return sol[:-1], sol[-1]


def smooth_kprime_matrix(chnkr):
    """Principal-value K' matrix for smooth curves.

    ``chunkermat(kernel("lap", "sp"))`` has undefined pointwise diagonal
    entries because the direct formula is evaluated at identical nodes. The
    smooth diagonal limit is ``-curvature / (4*pi)`` times the quadrature
    weight.
    """

    kprime = chunkermat(chnkr, kernel("lap", "sp"))
    weights = chnkr.wts.reshape(-1, order="F")
    curvature = chnkr.signed_curvature().reshape(-1, order="F")
    kprime[np.diag_indices_from(kprime)] = -curvature * weights / (4.0 * np.pi)
    return kprime


def single_layer_neumann(chnkr, normal_values, side):
    """Solve the single-layer Neumann equation on the requested side."""

    jump = 0.5 if side == "interior" else -0.5
    system = jump * np.eye(chnkr.npt) + smooth_kprime_matrix(chnkr)
    return np.linalg.solve(system, normal_values)


def exterior_solution_x_over_r2(targets):
    r2 = np.sum(targets**2, axis=0)
    return targets[0] / r2


def main() -> None:
    chnkr, _ = chunkerfunc(circle, {"nchmin": 10, "eps": 1e-10}, {"k": 16})
    boundary = flat_nodes(chnkr)
    lap_s = kernel("lap", "s")

    interior_targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
    exterior_targets = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])

    sigma_d, const_d = single_layer_dirichlet(chnkr, boundary[0])
    interior_d = chunkerkerneval(chnkr, lap_s, sigma_d, interior_targets, {"forceadap": True}).reshape(-1) + const_d
    exterior_d = chunkerkerneval(chnkr, lap_s, sigma_d, exterior_targets, {"forceadap": True}).reshape(-1) + const_d

    interior_d_err = np.max(np.abs(interior_d - interior_targets[0]))
    exterior_d_err = np.max(np.abs(exterior_d - exterior_solution_x_over_r2(exterior_targets)))

    normals = chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F")
    sigma_n_int = single_layer_neumann(chnkr, normals[0], "interior")
    interior_n = chunkerkerneval(chnkr, lap_s, sigma_n_int, interior_targets, {"forceadap": True}).reshape(-1)
    interior_n += interior_targets[0, 0] - interior_n[0]

    sigma_n_ext = single_layer_neumann(chnkr, -boundary[0], "exterior")
    exterior_n = chunkerkerneval(chnkr, lap_s, sigma_n_ext, exterior_targets, {"forceadap": True}).reshape(-1)

    interior_n_err = np.max(np.abs(interior_n - interior_targets[0]))
    exterior_n_err = np.max(np.abs(exterior_n - exterior_solution_x_over_r2(exterior_targets)))

    print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
    print(f"interior Dirichlet max error: {interior_d_err:.3e}")
    print(f"exterior Dirichlet max error: {exterior_d_err:.3e}")
    print(f"interior Neumann max error, after constant fix: {interior_n_err:.3e}")
    print(f"exterior Neumann max error: {exterior_n_err:.3e}")


if __name__ == "__main__":
    main()
