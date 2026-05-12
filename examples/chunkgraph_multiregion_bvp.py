"""Chunkgraph demo for a multiply connected, multi-region Laplace BVP.

Run from the repository root:

    uv run python examples/chunkgraph_multiregion_bvp.py

The graph has an outer square and an inner square inclusion. The annular
region between them is solved with a single-layer Dirichlet representation for
the manufactured solution u=x.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerkerneval, chunkermat, chunkgraph, chunkgraphinregion, find_edge_regions, kernel


def main() -> None:
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
    cg = chunkgraph(verts, edges, pref={"k": 12, "nchmax": 1000}, cparams={"nchmin": 1})
    lap_s = kernel("lap", "s")

    boundary = cg.r.reshape(2, cg.npt, order="F")
    sigma = np.linalg.solve(chunkermat(cg, lap_s), boundary[0])

    targets = np.array(
        [
            [0.0, 1.2, -1.5, 2.5, 0.0],
            [1.2, 0.4, -0.3, 0.0, 0.0],
        ]
    )
    region_ids = chunkgraphinregion(cg, targets)
    vals = chunkerkerneval(cg, lap_s, sigma, targets[:, region_ids == 2], {"forceadap": True}).reshape(-1)
    truth = targets[0, region_ids == 2]

    edge_regions = find_edge_regions(cg)

    print(f"chunkgraph: {len(cg.echnks)} edges, {cg.npt} nodes")
    print(f"regions at sample targets: {region_ids.tolist()}")
    print(f"edge regions on positive side: {edge_regions[0].tolist()}")
    print(f"edge regions on negative side: {edge_regions[1].tolist()}")
    print(f"annular-region Dirichlet max error: {np.max(np.abs(vals - truth)):.3e}")


if __name__ == "__main__":
    main()
