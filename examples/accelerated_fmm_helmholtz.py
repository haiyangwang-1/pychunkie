"""FMM target evaluation for a Helmholtz single-layer potential."""

import numpy as np
from _accelerated_common import TARGETS, boundary_nodes, make_circle, relerr

from chunkie import chunkerkerneval, kernel

boundary = make_circle()
nodes = boundary_nodes(boundary)
density = np.cos(nodes[0])
helm_s = kernel("helm", "s", 1.4 + 0.1j)

direct = chunkerkerneval(boundary, helm_s, density, TARGETS).reshape(-1)
fmm = chunkerkerneval(boundary, helm_s, density, TARGETS, acceleration="fmm", tol=1e-11).reshape(-1)
error = relerr(fmm, direct)

print(f"Helmholtz single layer FMM relative error: {error:.3e}")
