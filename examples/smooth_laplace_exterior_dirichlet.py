"""Exterior Laplace Dirichlet solve on a smooth unit circle."""

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 10, "eps": 1e-10}, {"k": 16})[0]
boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
weights = chnkr.wts.reshape(-1, order="F")
targets = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])

lap_s = kernel("lap", "s")
s_mat = chunkermat(chnkr, lap_s)
system = np.block(
    [
        [s_mat, np.ones((chnkr.npt, 1))],
        [weights[None, :], np.zeros((1, 1))],
    ]
)

boundary_truth = boundary[0] / np.sum(boundary**2, axis=0)
target_truth = targets[0] / np.sum(targets**2, axis=0)
rhs = np.concatenate((boundary_truth, [0.0]))
sol = np.linalg.solve(system, rhs)
sigma = sol[:-1]
const = float(sol[-1])
values = chunkerkerneval(chnkr, lap_s, sigma, targets, {"forceadap": True}).reshape(-1) + const

print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"exterior Dirichlet boundary residual: {np.max(np.abs(s_mat @ sigma + const - boundary_truth)):.3e}")
print(f"exterior Dirichlet target max error: {np.max(np.abs(values - target_truth)):.3e}")
