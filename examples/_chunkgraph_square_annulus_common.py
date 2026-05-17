"""Shared square-annulus ChunkGraph setup for examples."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from chunkie.geometry import ChunkGraph, GraphRegion, RegionCycle, SignedEdge

SAMPLE_TARGETS = np.array(
    [
        [0.0, 1.2, -1.5, 2.5, 0.0],
        [1.2, 0.4, -0.3, 0.0, 0.0],
    ],
)

SQUARE_ANNULUS_VERTICES = np.array(
    [
        [-2.0, 2.0, 2.0, -2.0, -0.6, 0.6, 0.6, -0.6],
        [-2.0, -2.0, 2.0, 2.0, -0.6, -0.6, 0.6, 0.6],
    ],
)
SQUARE_ANNULUS_EDGES = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 0, 5, 6, 7, 4],
    ],
)


def make_square_annulus(*, quadrature_order: int = 16) -> ChunkGraph:
    """Build an outer square with an inner square hole."""

    graph = ChunkGraph.from_vertices(
        SQUARE_ANNULUS_VERTICES,
        SQUARE_ANNULUS_EDGES,
        quadrature_order=quadrature_order,
    )
    # The outer cycle is counter-clockwise. The inner geometry is also stored
    # counter-clockwise as the boundary of the hole, so the annular material
    # region uses the reversed inner cycle to keep the material on the left.
    outer = RegionCycle(tuple(SignedEdge(edge_id, 1) for edge_id in range(4)))
    inner_ccw = RegionCycle(tuple(SignedEdge(edge_id, 1) for edge_id in range(4, 8)))
    inner_cw = RegionCycle(tuple(SignedEdge(edge_id, -1) for edge_id in range(4, 8)))
    graph.regions = [
        GraphRegion(0, (), bounded=False, label="exterior"),
        GraphRegion(1, (outer, inner_cw), bounded=True, label="annulus"),
        GraphRegion(2, (inner_ccw,), bounded=True, label="hole"),
    ]
    for edge_id in range(4, 8):
        graph.edges[edge_id] = replace(graph.edges[edge_id], left_region=2, right_region=1)
    return graph
