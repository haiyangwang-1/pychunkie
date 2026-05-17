import numpy as np

from chunkie.geometry import chunker_from_polygon, circle, refine


def test_refine_splits_circle_panels_with_reference_chain_rule():
    boundary = circle(radius=1.0, quadrature_order=18, panel_count=4)

    refined = refine(boundary, levels=1)

    assert refined.panel_count == 8
    np.testing.assert_array_equal(refined.adjacency[:, 0], [7, 1])
    np.testing.assert_array_equal(refined.adjacency[:, -1], [6, 0])
    np.testing.assert_allclose(refined.area, boundary.area, atol=1.0e-12)
    np.testing.assert_allclose(refined.length, boundary.length, atol=1.0e-12)

    breaks = np.linspace(0.0, 2.0 * np.pi, refined.panel_count + 1)
    for panel_id, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
        half_width = 0.5 * (right - left)
        theta = 0.5 * (left + right) + half_width * refined.nodes
        expected_positions = np.vstack((np.cos(theta), np.sin(theta)))
        expected_derivatives = half_width * np.vstack((-np.sin(theta), np.cos(theta)))
        expected_second_derivatives = -(half_width**2) * np.vstack((np.cos(theta), np.sin(theta)))

        np.testing.assert_allclose(refined.positions[:, :, panel_id], expected_positions, atol=1.0e-12)
        np.testing.assert_allclose(refined.derivatives[:, :, panel_id], expected_derivatives, atol=1.0e-12)
        np.testing.assert_allclose(refined.second_derivatives[:, :, panel_id], expected_second_derivatives, atol=1.0e-12)
        np.testing.assert_allclose(refined.normals[:, :, panel_id], expected_positions, atol=1.0e-12)
        np.testing.assert_allclose(refined.weights[:, panel_id], half_width * refined.reference_weights, atol=1.0e-13)


def test_refine_keeps_open_line_adjacency_and_weights_consistent():
    boundary = chunker_from_polygon(
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        closed=False,
        quadrature_order=8,
    )

    refined = refine(boundary, levels=2)

    assert refined.panel_count == 4
    assert not refined.closed
    np.testing.assert_array_equal(refined.adjacency[:, 0], [-1, 1])
    np.testing.assert_array_equal(refined.adjacency[:, -1], [2, -1])
    np.testing.assert_allclose(refined.panel_lengths, 0.25, atol=1.0e-14)
    np.testing.assert_allclose(refined.length, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(refined.positions[1], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(refined.derivatives[0], 0.125, atol=1.0e-14)
    np.testing.assert_allclose(refined.derivatives[1], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(refined.second_derivatives, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(refined.normals[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(refined.normals[1], -1.0, atol=1.0e-14)
    expected_weights = np.repeat(0.125 * refined.reference_weights[:, None], refined.panel_count, axis=1)
    np.testing.assert_allclose(refined.weights, expected_weights, atol=1.0e-14)
