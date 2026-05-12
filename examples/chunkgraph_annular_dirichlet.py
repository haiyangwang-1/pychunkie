"""Dirichlet solve on the annular region of a square-annulus chunkgraph."""

import numpy as np

from chunkie import chunkerkerneval, chunkermat, chunkgraph, chunkgraphinregion, kernel


verts = np.array(
    [
        [-2.0, 2.0, 2.0, -2.0, -0.6, 0.6, 0.6, -0.6],
        [-2.0, -2.0, 2.0, 2.0, -0.6, -0.6, 0.6, 0.6],
    ]
)
edges = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
    ]
)
targets = np.array(
    [
        [0.0, 1.2, -1.5, 2.5, 0.0],
        [1.2, 0.4, -0.3, 0.0, 0.0],
    ]
)

cg = chunkgraph(verts, edges, pref={"k": 12, "nchmax": 2000}, cparams={"nchmin": 8})
lap_s = kernel("lap", "s")
boundary = cg.r.reshape(2, cg.npt, order="F")
sigma = np.linalg.solve(chunkermat(cg, lap_s), boundary[0])

region_ids = chunkgraphinregion(cg, targets)
annular_targets = targets[:, region_ids == 2]
values = chunkerkerneval(cg, lap_s, sigma, annular_targets, {"forceadap": True}).reshape(-1)
error = np.max(np.abs(values - annular_targets[0]))

print(f"chunkgraph: {len(cg.echnks)} edges, {cg.npt} nodes")
print(f"annular target count: {annular_targets.shape[1]}")
print(f"annular-region Dirichlet max error: {error:.3e}")
