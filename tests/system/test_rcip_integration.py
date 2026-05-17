import numpy as np

from chunkie.geometry import ChunkGraph
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
    SystemConfig,
    build_rcip_state,
)


def _square_graph(quadrature_order=4):
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    return ChunkGraph.from_vertices(vertices, edges, quadrature_order=quadrature_order)


def test_build_rcip_state_discovers_chunkgraph_corners_and_local_layouts():
    graph = _square_graph(quadrature_order=4)
    boundary = graph.boundary(1)
    density = DensitySpace("sigma", boundary)
    system = IntegralSystem("square", graph, (density,), ())

    state = build_rcip_state(system, config=SystemConfig(rcip_subdivisions=3))

    assert state.metadata["active"] is True
    assert state.metadata["subdivisions"] == 3
    assert len(state.corners) == 4
    for corner in state.corners:
        assert len(corner.edge_ids) == 2
        assert corner.boundary_part.point_count == 2 * graph.quadrature_order
        assert corner.local_geometry.panel_count == 2 * 3
        assert corner.prolongation.shape == (4 * graph.quadrature_order, 2 * graph.quadrature_order)
        assert corner.weighted_prolongation.shape == corner.prolongation.shape
        np.testing.assert_allclose(corner.compressed_inverse, np.eye(2 * graph.quadrature_order))


def test_dense_assembly_records_rcip_state_without_changing_reference_matrix():
    graph = _square_graph(quadrature_order=4)
    boundary = graph.boundary(1, side="left")
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", boundary)
    layer = LayerPotential("single", boundary, single, "sigma")
    system = IntegralSystem(
        "square",
        graph,
        (density,),
        (
            BoundaryEquation(
                "dirichlet",
                boundary,
                (BoundaryTrace(layer, boundary, "left"),),
                np.zeros(boundary.point_count),
            ),
        ),
    )

    with_rcip = system.assemble(config=SystemConfig(rcip_subdivisions=2))
    without_rcip = system.assemble(config=SystemConfig(use_rcip=False))

    assert "rcip" in with_rcip.diagnostics
    assert len(with_rcip.diagnostics["rcip"].corners) == 4
    assert "rcip" not in without_rcip.diagnostics
    np.testing.assert_allclose(with_rcip.to_dense(), without_rcip.to_dense())


def test_rcip_state_attaches_finite_local_trace_operators():
    graph = _square_graph(quadrature_order=4)
    boundary = graph.boundary(1, side="left")
    double = kernel("laplace", selector="d")
    density = DensitySpace("sigma", boundary)
    layer = LayerPotential("double", boundary, double, "sigma")
    system = IntegralSystem(
        "square",
        graph,
        (density,),
        (
            BoundaryEquation(
                "dirichlet",
                boundary,
                (BoundaryTrace(layer, boundary, "interior", jump=JumpTerm(-0.5, "sigma")),),
                boundary.pointinfo.positions[0],
            ),
        ),
    )

    state = build_rcip_state(system, config=SystemConfig(rcip_subdivisions=3))

    for corner in state.corners:
        assert corner.local_operator is not None
        assert corner.local_operator.shape == (
            corner.local_geometry.point_count,
            corner.local_geometry.point_count,
        )
        assert np.all(np.isfinite(corner.local_operator))
