"""FMM target evaluation for a Laplace single-layer potential."""

import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, ellipse, kernel


chnkr = chunkerfunc(ellipse, {"nchmin": 6, "eps": 1e-8}, {"k": 8})[0]
nodes = chnkr.r.reshape(2, chnkr.npt, order="F")
targets = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])
density = np.cos(nodes[0])
lap_s = kernel("lap", "s")

direct = chunkerkerneval(chnkr, lap_s, density, targets).reshape(-1, order="F")
fmm = chunkerkerneval(
    chnkr, lap_s, density, targets, {"acceleration": "fmm", "eps": 1e-11}
).reshape(-1, order="F")
error = np.linalg.norm(fmm - direct) / max(np.linalg.norm(direct), 1.0)

print(f"Laplace single layer FMM relative error: {error:.3e}")
