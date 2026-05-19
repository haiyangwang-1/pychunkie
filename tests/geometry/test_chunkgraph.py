import numpy as np

from chunkie.geometry import ChunkGraph, GraphRegion, RegionCycle, SignedEdge


def test_chunkgraph_from_vertices_builds_directed_square_edges():
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])

    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=6)

    assert len(graph.vertices) == 4
    assert len(graph.edges) == 4
    assert graph.point_count == 4 * 6
    assert graph.panel_count == 4
    for edge_id, edge in enumerate(graph.edges):
        start = vertices[:, edges[0, edge_id]]
        end = vertices[:, edges[1, edge_id]]
        tangent = end - start
        half_tangent = tangent / 2.0
        reference = (graph.edge(edge_id).chunker._legendre_nodes + 1.0) / 2.0
        expected_positions = start[:, None] + tangent[:, None] * reference[None, :]
        expected_normals = np.repeat([[half_tangent[1]], [-half_tangent[0]]], 6, axis=1) / np.linalg.norm(
            half_tangent,
        )

        np.testing.assert_allclose(edge.chunker.positions[:, :, 0], expected_positions)
        np.testing.assert_allclose(edge.chunker.derivatives[:, :, 0], np.repeat(half_tangent[:, None], 6, axis=1))
        np.testing.assert_allclose(edge.chunker.second_derivatives[:, :, 0], 0.0)
        np.testing.assert_allclose(edge.chunker.normals[:, :, 0], expected_normals)
        assert edge.left_region == 1
        assert edge.right_region == 0


def test_chunkgraph_multi_edge_boundary_and_region_classification():
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=5)

    boundary = graph.boundary(1, side="left")
    regions = graph.classify_points(np.array([[0.5, 1.5, 0.25], [0.5, 0.5, 1.25]]))

    assert boundary.edges == (0, 1, 2, 3)
    assert boundary.point_indices.size == graph.point_count
    np.testing.assert_array_equal(boundary.orientation, [1, 1, 1, 1])
    np.testing.assert_allclose(graph.pointinfo.flat_positions[:, boundary.point_indices], boundary.points.flat_positions)
    np.testing.assert_array_equal(regions, [1, 0, 0])
    np.testing.assert_allclose(boundary.signed_curvature, 0.0)


def test_chunkgraph_boundary_part_selects_edge_subset_in_global_order():
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=5)

    part = graph.boundary_part([1, 2], side="left")
    merged = graph.pointinfo

    assert part.edges == (1, 2)
    assert part.side == "left"
    np.testing.assert_array_equal(part.orientation, [1, 1])
    np.testing.assert_array_equal(part.point_indices, np.arange(5, 15))
    np.testing.assert_allclose(part.points.flat_positions, merged.flat_positions[:, part.point_indices])
    np.testing.assert_allclose(part.points.flat_normals, merged.flat_normals[:, part.point_indices])
    np.testing.assert_allclose(part.points.flat_weights, merged.flat_weights[part.point_indices])


def test_chunkgraph_boundary_part_reverses_edge_orientation_tensors():
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=5)
    edge = graph.edge(1).chunker

    part = graph.boundary_part([1], side="right", orientation=-1)

    np.testing.assert_array_equal(part.orientation, [-1])
    np.testing.assert_array_equal(part.point_indices, np.arange(5, 10)[::-1])
    np.testing.assert_allclose(part.points.positions, edge.positions[:, ::-1, ::-1])
    np.testing.assert_allclose(part.points.derivatives, -edge.derivatives[:, ::-1, ::-1])
    np.testing.assert_allclose(part.points.second_derivatives, edge.second_derivatives[:, ::-1, ::-1])
    np.testing.assert_allclose(part.points.normals, -edge.normals[:, ::-1, ::-1])
    np.testing.assert_allclose(part.points.weights, edge.weights[::-1, ::-1])


def test_chunkgraph_nested_region_classification_uses_oriented_hole_cycles():
    vertices = np.array(
        [
            [0.0, 3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 1.0],
            [0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 2.0, 2.0],
        ],
    )
    edges = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 0, 5, 6, 7, 4],
        ],
    )
    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=4)
    outer = RegionCycle(tuple(SignedEdge(edge_id, 1) for edge_id in range(4)))
    inner_ccw = RegionCycle(tuple(SignedEdge(edge_id, 1) for edge_id in range(4, 8)))
    inner_cw = RegionCycle(tuple(SignedEdge(edge_id, -1) for edge_id in range(4, 8)))
    graph.regions = [
        GraphRegion(0, (), bounded=False, label="exterior"),
        GraphRegion(1, (outer, inner_cw), bounded=True, label="annulus"),
        GraphRegion(2, (inner_ccw,), bounded=True, label="hole"),
    ]

    points = np.array(
        [
            [0.5, 1.5, 3.5],
            [0.5, 1.5, 1.5],
        ],
    )

    np.testing.assert_array_equal(graph.classify_points(points), [1, 2, 0])
