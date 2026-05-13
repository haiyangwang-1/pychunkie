"""RCIP corner-compression diagnostic on a square chunkgraph."""

import numpy as np

from chunkie import chunkgraph, kernel
from chunkie.quadrature import rcip


verts = np.array(
    [
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ]
)
edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
nsub = 3

cg = chunkgraph(verts, edges, pref={"k": 8, "nchmax": 1000}, cparams={"nchmin": 2})
result = rcip.chunkgraph_rcip(cg, kernel("lap", "d"), 1, opts={"nsub": nsub, "rcip_savedepth": nsub})
deviations = np.linalg.norm(np.stack(result.R) - np.eye(result.R[0].shape[0]), axis=(1, 2))

print(f"corner blocks: {result.vertices.size}")
print(f"RCIP block size: {result.R[0].shape[0]}")
print(f"max ||R-I||_F: {max(deviations):.3e}")
