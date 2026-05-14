"""Classify targets in a square-annulus chunkgraph."""

import numpy as np

from chunkie import ChunkGraph, chunkgraphinregion, find_edge_regions

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

cg = ChunkGraph(verts, edges, pref={"k": 12, "nchmax": 2000}, cparams={"nchmin": 8})
region_ids = chunkgraphinregion(cg, targets)
edge_regions = find_edge_regions(cg)

print(f"chunkgraph: {len(cg.echnks)} edges, {cg.npt} nodes")
print(f"regions at sample targets: {region_ids.tolist()}")
print(f"edge regions on positive side: {edge_regions[0].tolist()}")
print(f"edge regions on negative side: {edge_regions[1].tolist()}")
