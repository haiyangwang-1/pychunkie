"""FMM target evaluation for a Stokes single-layer velocity."""

import numpy as np
from _accelerated_common import TARGETS, boundary_nodes, component_vector, make_circle, relerr

from chunkie import chunkerkerneval, kernel

boundary = make_circle()
nodes = boundary_nodes(boundary)
density = component_vector(np.vstack((np.cos(nodes[0]), np.sin(nodes[1]))))
stok_s = kernel("stok", "s", 1.0)

direct = component_vector(chunkerkerneval(boundary, stok_s, density, TARGETS))
fmm = chunkerkerneval(boundary, stok_s, density, TARGETS, acceleration="fmm", tol=1e-11)
error = relerr(component_vector(fmm), direct)

print(f"Stokes velocity FMM relative error: {error:.3e}")
