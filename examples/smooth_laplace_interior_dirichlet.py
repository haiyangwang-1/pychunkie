"""Interior Laplace Dirichlet solve on a smooth unit circle."""

import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel

boundary = chunkerfunc(ellipse, min_chunks=10, tol=1e-10, order=16)[0]
nodes = PointInfo.from_any(boundary).r
weights = boundary.quadrature_weights.T.reshape(-1)
targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])

lap_s = kernel("lap", "s")
s_mat = chunkermat(boundary, lap_s)
system = np.block(
    [
        [s_mat, np.ones((boundary.npt, 1))],
        [weights[None, :], np.zeros((1, 1))],
    ]
)

rhs = np.concatenate((nodes[0], [0.0]))
sol = np.linalg.solve(system, rhs)
sigma = sol[:-1]
const = float(sol[-1])
values = chunkerkerneval(boundary, lap_s, sigma, targets, force_adaptive=True).reshape(-1) + const

print(f"unit circle: {boundary.nch} chunks, {boundary.npt} nodes")
print(
    f"interior Dirichlet boundary residual: {np.max(np.abs(s_mat @ sigma + const - nodes[0])):.3e}"
)
print(f"interior Dirichlet target max error: {np.max(np.abs(values - targets[0])):.3e}")
