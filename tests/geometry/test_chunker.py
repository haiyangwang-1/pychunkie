import numpy as np
import pytest

from chunkie.geometry import ChunkGraph, chunker_from_polygon, circle, ellipse, flagnear


def test_circle_uses_panel_major_storage_and_exterior_normals():
    boundary = circle(radius=2.0, quadrature_order=8, panel_count=4)

    assert boundary.positions.shape == (2, 8, 4)
    assert boundary.weights.shape == (8, 4)
    assert boundary.point_count == 32
    with pytest.raises(AttributeError):
        _ = boundary.nodes
    with pytest.raises(AttributeError):
        _ = boundary.reference_weights
    with pytest.raises(AttributeError):
        boundary.nodes = np.zeros(8)
    with pytest.raises(AttributeError):
        boundary.reference_weights = np.zeros(8)

    radial = boundary.positions / np.linalg.norm(boundary.positions, axis=0, keepdims=True)
    np.testing.assert_allclose(boundary.normals, radial, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(boundary.weights), 4.0 * np.pi, rtol=1.0e-12)


def test_panel_major_point_ids_use_direct_formula():
    boundary = ellipse(axes=(2.0, 1.0), quadrature_order=5, panel_count=3)
    point_id = 2 * boundary.quadrature_order + 4
    panel = point_id // boundary.quadrature_order
    local = point_id % boundary.quadrature_order

    assert point_id == 14
    assert panel == 2
    assert local == 4


def test_polygon_constructor_and_near_flags_are_active():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=4)
    flags = flagnear(boundary, np.array([[0.5], [0.05]]), near_factor=1.0)

    assert boundary.panel_count == 4
    assert flags.shape == (1, 4)
    assert np.any(flags)


def test_chunkgraph_exposes_merged_points_and_boundary_parts():
    boundary = circle(quadrature_order=6, panel_count=5)
    graph = ChunkGraph.from_chunker(boundary)
    part = graph.boundary(1)

    assert graph.point_count == boundary.point_count
    np.testing.assert_allclose(graph.pointinfo.flat_positions, boundary.pointinfo.flat_positions)
    assert part.edges == (0,)
    assert part.point_indices.size == boundary.point_count
    assert part.side == "interior"


def test_chunkgraph_classifies_single_closed_boundary_regions():
    graph = ChunkGraph.from_chunker(circle(quadrature_order=8, panel_count=24))
    points = np.array([[0.0, 2.0], [0.0, 0.0]])

    regions = graph.classify_points(points)

    np.testing.assert_array_equal(regions, np.array([1, 0]))
