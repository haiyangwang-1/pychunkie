"""Shared square-annulus chunkgraph setup for examples."""

from __future__ import annotations

import numpy as np

from chunkie import ChunkGraph, PointInfo

SAMPLE_TARGETS = np.array(
    [
        [0.0, 1.2, -1.5, 2.5, 0.0],
        [1.2, 0.4, -0.3, 0.0, 0.0],
    ]
)


def make_square_annulus():
    """Build an outer square with an inner square inclusion."""

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
    return ChunkGraph(verts, edges, pref={"k": 12, "nchmax": 2000}, cparams={"nchmin": 8})


def boundary_nodes(cg) -> np.ndarray:
    return PointInfo.from_any(cg).r
