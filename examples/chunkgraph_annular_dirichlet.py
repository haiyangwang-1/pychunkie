"""Dirichlet solve on the annular region of a square-annulus chunkgraph.

Run from the repository root:

    uv run python examples/chunkgraph_annular_dirichlet.py

The manufactured solution is u=x. Only targets classified in the annular
region are evaluated.
"""

from __future__ import annotations

import numpy as np

from chunkie import chunkerkerneval, chunkermat, chunkgraphinregion, kernel

from _chunkgraph_square_annulus_common import SAMPLE_TARGETS, boundary_nodes, make_square_annulus


def main() -> None:
    cg = make_square_annulus()
    lap_s = kernel("lap", "s")

    boundary = boundary_nodes(cg)
    sigma = np.linalg.solve(chunkermat(cg, lap_s), boundary[0])

    region_ids = chunkgraphinregion(cg, SAMPLE_TARGETS)
    annular_targets = SAMPLE_TARGETS[:, region_ids == 2]
    values = chunkerkerneval(cg, lap_s, sigma, annular_targets, {"forceadap": True}).reshape(-1)
    error = np.max(np.abs(values - annular_targets[0]))

    print(f"chunkgraph: {len(cg.echnks)} edges, {cg.npt} nodes")
    print(f"annular target count: {annular_targets.shape[1]}")
    print(f"annular-region Dirichlet max error: {error:.3e}")


if __name__ == "__main__":
    main()
