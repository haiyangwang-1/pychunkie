"""Exterior Laplace Neumann solve on a smooth unit circle."""

import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, chunkermat, ellipse, kernel

boundary = chunkerfunc(ellipse, min_chunks=10, tol=1e-10, order=16)[0]
nodes = PointInfo.from_any(boundary).r
targets = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])

lap_s = kernel("lap", "s")
kprime = chunkermat(boundary, kernel("lap", "sp"))
weights = boundary.quadrature_weights.T.reshape(-1)
curvature = boundary.signed_curvature().T.reshape(-1)
kprime[np.diag_indices_from(kprime)] = -curvature * weights / (4.0 * np.pi)

system = -0.5 * np.eye(boundary.npt) + kprime
normal_data = -nodes[0]
sigma = np.linalg.solve(system, normal_data)
values = chunkerkerneval(boundary, lap_s, sigma, targets, force_adaptive=True).reshape(-1)
target_truth = targets[0] / np.sum(targets**2, axis=0)

print(f"unit circle: {boundary.nch} chunks, {boundary.npt} nodes")
print(f"exterior Neumann boundary residual: {np.max(np.abs(system @ sigma - normal_data)):.3e}")
print(f"exterior Neumann target max error: {np.max(np.abs(values - target_truth)):.3e}")
