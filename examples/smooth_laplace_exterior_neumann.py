"""Exterior Laplace Neumann solve on a smooth unit circle."""

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 10, "eps": 1e-10}, {"k": 16})[0]
boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
targets = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])

lap_s = kernel("lap", "s")
kprime = chunkermat(chnkr, kernel("lap", "sp"))
weights = chnkr.wts.reshape(-1, order="F")
curvature = chnkr.signed_curvature().reshape(-1, order="F")
kprime[np.diag_indices_from(kprime)] = -curvature * weights / (4.0 * np.pi)

system = -0.5 * np.eye(chnkr.npt) + kprime
normal_data = -boundary[0]
sigma = np.linalg.solve(system, normal_data)
values = chunkerkerneval(chnkr, lap_s, sigma, targets, {"forceadap": True}).reshape(-1)
target_truth = targets[0] / np.sum(targets**2, axis=0)

print(f"unit circle: {chnkr.nch} chunks, {chnkr.npt} nodes")
print(f"exterior Neumann boundary residual: {np.max(np.abs(system @ sigma - normal_data)):.3e}")
print(f"exterior Neumann target max error: {np.max(np.abs(values - target_truth)):.3e}")
