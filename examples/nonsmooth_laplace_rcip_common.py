"""Shared square geometry for non-smooth Laplace RCIP examples."""

from __future__ import annotations

import numpy as np

from chunkie import ChunkGraph

SQUARE_VERTS = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
SQUARE_EDGES = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])


def square_graph(depth: int = 2, k: int = 12):
    """Build the coarse square chunkgraph used by the RCIP examples."""

    return ChunkGraph(
        SQUARE_VERTS,
        SQUARE_EDGES,
        pref={"k": k, "nchmax": 2000},
        cparams={"nchmin": 2 ** int(depth)},
    )
