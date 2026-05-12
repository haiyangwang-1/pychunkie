"""Non-smooth Laplace BVP demo on a square with dyadic corner refinement.

Run from the repository root:

    uv run python examples/nonsmooth_laplace_polygon.py

The same manufactured interior solution, u=x, is used for Dirichlet and
Neumann data. The square is kept geometrically non-smooth; dyadic panels
resolve the corner neighborhoods instead of rounding them.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerkerneval, chunkermat, chunkerpoly, kernel


def flat_nodes(chnkr):
    return chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F")


def single_layer_dirichlet(chnkr, boundary_values):
    lap_s = kernel("lap", "s")
    s_mat = chunkermat(chnkr, lap_s)
    weights = chnkr.wts.reshape(-1, order="F")
    system = np.block(
        [
            [s_mat, np.ones((chnkr.npt, 1))],
            [weights[None, :], np.zeros((1, 1))],
        ]
    )
    sol = np.linalg.solve(system, np.concatenate((boundary_values, [0.0])))
    return sol[:-1], sol[-1]


def polygon_kprime_matrix(chnkr):
    """Principal-value K' matrix for straight-panel polygon nodes.

    Nodes do not lie exactly on the corners, and straight panels have zero
    smooth curvature, so only the undefined self entries need replacement.
    """

    kprime = chunkermat(chnkr, kernel("lap", "sp"))
    kprime[np.diag_indices_from(kprime)] = 0.0
    return kprime


def main() -> None:
    verts = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
        ]
    )
    chnkr = chunkerpoly(
        verts,
        {"ifclosed": True, "dyadic": True, "depth": 3, "widths": 0.25},
        {"k": 12, "nchmax": 2000},
    )

    boundary = flat_nodes(chnkr)
    normals = chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F")
    targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
    truth = targets[0]
    lap_s = kernel("lap", "s")

    sigma_d, const_d = single_layer_dirichlet(chnkr, boundary[0])
    vals_d = chunkerkerneval(chnkr, lap_s, sigma_d, targets, {"forceadap": True}).reshape(-1) + const_d

    kprime = polygon_kprime_matrix(chnkr)
    sigma_n = np.linalg.lstsq(0.5 * np.eye(chnkr.npt) + kprime, normals[0], rcond=None)[0]
    vals_n = chunkerkerneval(chnkr, lap_s, sigma_n, targets, {"forceadap": True}).reshape(-1)
    vals_n += truth[0] - vals_n[0]

    print(f"dyadic square: {chnkr.nch} chunks, {chnkr.npt} nodes, area {chnkr.area():.6f}")
    print(f"interior Dirichlet max error: {np.max(np.abs(vals_d - truth)):.3e}")
    print(f"interior Neumann max error, after constant fix: {np.max(np.abs(vals_n - truth)):.3e}")


if __name__ == "__main__":
    main()
