"""Dirichlet solve on the annular region of a square-annulus ChunkGraph."""

from dataclasses import replace

import numpy as np

from chunkie.geometry import ChunkGraph, GraphRegion, RegionCycle, SignedEdge
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)

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
    boundary = graph.boundary(1, side="interior")
    density = DensitySpace("sigma", boundary)
    double = LayerPotential("double", boundary, kernel("laplace", selector="d"), "sigma")
    trace = BoundaryTrace(double, boundary, "interior", jump=JumpTerm(-0.5, "sigma"))
    system = IntegralSystem(
        "chunkgraph_annular_dirichlet",
        graph,
        (density,),
        (BoundaryEquation("dirichlet", boundary, (trace,), boundary.pointinfo.positions[0]),),
        fields={"u": (double,)},
    )
    solution = system.solve()

    region_ids = graph.classify_points(SAMPLE_TARGETS)
    annular_targets = SAMPLE_TARGETS[:, region_ids == 1]
    values = solution.evaluate(annular_targets).values[0].real
    error = abs(values - annular_targets[0]).max()

    print(f"chunkgraph: {len(graph.edges)} edges, {boundary.point_count} annular boundary nodes")
    print(f"annular target count: {annular_targets.shape[1]}")
    print(f"annular-region Dirichlet max error: {error:.3e}")


if __name__ == "__main__":
    main()
