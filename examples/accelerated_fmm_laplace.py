"""FMM target evaluation for a Laplace single-layer potential."""

import numpy as np
from _accelerated_common import TARGETS, boundary_nodes, make_circle, relerr

from chunkie import chunkerkerneval, kernel

boundary = make_circle()
nodes = boundary_nodes(boundary)
density = np.cos(nodes[0])
lap_s = kernel("lap", "s")

direct = chunkerkerneval(boundary, lap_s, density, TARGETS).reshape(-1)
fmm = chunkerkerneval(boundary, lap_s, density, TARGETS, acceleration="fmm", tol=1e-11).reshape(-1)
error = relerr(fmm, direct)

print(f"Laplace single layer FMM relative error: {error:.3e}")
