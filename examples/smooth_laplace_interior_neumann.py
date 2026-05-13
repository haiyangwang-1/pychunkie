"""Interior Laplace Neumann solve on a smooth unit circle."""

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 10, "eps": 1e-10}, {"k": 16})[0]
normals = chnkr.n.reshape(2, chnkr.npt, order="F")
targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])

lap_s = kernel("lap", "s")
kprime = chunkermat(chnkr, kernel("lap", "sp"))
weights = chnkr.wts.reshape(-1, order="F")
curvature = chnkr.signed_curvature().reshape(-1, order="F")
kprime[np.diag_indices_from(kprime)] = -curvature * weights / (4.0 * np.pi)

system = 0.5 * np.eye(chnkr.npt) + kprime
normal_data = normals[0]
sigma = np.linalg.solve(system, normal_data)
values = chunkerkerneval(chnkr, lap_s, sigma, targets, {"forceadap": True}).reshape(-1)
values += targets[0, 0] - values[0]

print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"interior Neumann boundary residual: {np.max(np.abs(system @ sigma - normal_data)):.3e}")
print(f"interior Neumann target max error: {np.max(np.abs(values - targets[0])):.3e}")
