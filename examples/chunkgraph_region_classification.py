"""Classify targets in a square-annulus ChunkGraph."""

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
    graph = ChunkGraph.from_vertices(
        SQUARE_ANNULUS_VERTICES,
        SQUARE_ANNULUS_EDGES,
        quadrature_order=quadrature_order,
    )
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


def main() -> None:
    graph = make_square_annulus()
    region_ids = graph.classify_points(SAMPLE_TARGETS)
    left_regions = [edge.left_region for edge in graph.edges]
    right_regions = [edge.right_region for edge in graph.edges]

    print(f"chunkgraph: {len(graph.edges)} edges, {graph.point_count} nodes")
    print(f"regions at sample targets: {region_ids.tolist()}")
    print(f"edge left regions: {left_regions}")
    print(f"edge right regions: {right_regions}")


if __name__ == "__main__":
    main()
