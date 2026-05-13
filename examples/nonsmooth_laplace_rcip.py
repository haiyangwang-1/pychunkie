"""Default chunkermat RCIP corner-compression diagnostic on a square."""

import numpy as np

from chunkie import chunkermat, chunkgraph, kernel


verts = np.array(
    [
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ]
)
edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
nsub = 20

cg = chunkgraph(verts, edges, pref={"k": 8, "nchmax": 1000}, cparams={"nchmin": 2})
mat = chunkermat(cg, kernel("lap", "d"), {"nsub": nsub, "rcip_savedepth": nsub})
context = mat.rcip
if context is None:
    raise RuntimeError("chunkermat did not attach RCIP metadata")
deviations = np.linalg.norm(np.stack([saved.R[-1] for saved in context.saved]) - np.eye(context.saved[0].R[-1].shape[0]), axis=(1, 2))

print(f"corner blocks: {len(context.saved)}")
print(f"RCIP block size: {context.saved[0].R[-1].shape[0]}")
print(f"max ||R-I||_F: {max(deviations):.3e}")
