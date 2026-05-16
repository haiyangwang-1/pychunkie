import numpy as np

from chunkie.geometry import chunker_from_polygon, circle, ellipse, flagnear


def test_circle_uses_panel_major_storage_and_exterior_normals():
    boundary = circle(radius=2.0, quadrature_order=8, panel_count=4)

    assert boundary.positions.shape == (2, 8, 4)
    assert boundary.weights.shape == (8, 4)
    assert boundary.point_count == 32

    radial = boundary.positions / np.linalg.norm(boundary.positions, axis=0, keepdims=True)
    np.testing.assert_allclose(boundary.normals, radial, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(boundary.weights), 4.0 * np.pi, rtol=1.0e-12)


def test_point_map_is_panel_major():
    boundary = ellipse(axes=(2.0, 1.0), quadrature_order=5, panel_count=3)
    point_id = boundary.point_map.to_point_id(2, 4)
    panel, local = boundary.point_map.from_point_id(point_id)

    assert int(point_id) == 14
    assert int(panel) == 2
    assert int(local) == 4


def test_polygon_constructor_and_near_flags_are_active():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=4)
    flags = flagnear(boundary, np.array([[0.5], [0.05]]), near_factor=1.0)

    assert boundary.panel_count == 4
    assert flags.shape == (1, 4)
    assert np.any(flags)
