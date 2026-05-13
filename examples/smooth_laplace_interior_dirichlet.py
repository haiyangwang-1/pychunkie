"""Interior Laplace Dirichlet solve on a smooth unit circle."""

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 10, "eps": 1e-10}, {"k": 16})[0]
boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
weights = chnkr.wts.reshape(-1, order="F")
targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])

lap_s = kernel("lap", "s")
s_mat = chunkermat(chnkr, lap_s)
system = np.block(
    [
        [s_mat, np.ones((chnkr.npt, 1))],
        [weights[None, :], np.zeros((1, 1))],
    ]
)

rhs = np.concatenate((boundary[0], [0.0]))
sol = np.linalg.solve(system, rhs)
sigma = sol[:-1]
const = float(sol[-1])
values = chunkerkerneval(chnkr, lap_s, sigma, targets, {"forceadap": True}).reshape(-1) + const

print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"interior Dirichlet boundary residual: {np.max(np.abs(s_mat @ sigma + const - boundary[0])):.3e}")
print(f"interior Dirichlet target max error: {np.max(np.abs(values - targets[0])):.3e}")
