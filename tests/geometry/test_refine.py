import numpy as np

from chunkie.geometry import change_quadrature_order, chunker_from_polygon, circle, refine
from chunkie.geometry.constructors import chunker_from_curve


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
        theta = 0.5 * (left + right) + half_width * refined._legendre_nodes
        expected_positions = np.vstack((np.cos(theta), np.sin(theta)))
        expected_derivatives = half_width * np.vstack((-np.sin(theta), np.cos(theta)))
        expected_second_derivatives = -(half_width**2) * np.vstack((np.cos(theta), np.sin(theta)))

        np.testing.assert_allclose(
            refined.positions[:, :, panel_id], expected_positions, atol=1.0e-12
        )
        np.testing.assert_allclose(
            refined.derivatives[:, :, panel_id], expected_derivatives, atol=1.0e-12
        )
        np.testing.assert_allclose(
            refined.second_derivatives[:, :, panel_id], expected_second_derivatives, atol=1.0e-12
        )
        np.testing.assert_allclose(
            refined.normals[:, :, panel_id], expected_positions, atol=1.0e-12
        )
        np.testing.assert_allclose(
            refined.weights[:, panel_id],
            half_width * refined._legendre_weights,
            atol=1.0e-13,
        )


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
    expected_weights = np.repeat(
        0.125 * refined._legendre_weights[:, None], refined.panel_count, axis=1
    )
    np.testing.assert_allclose(refined.weights, expected_weights, atol=1.0e-14)


def test_refine_splits_selected_panels_with_direct_point_order():
    boundary = chunker_from_polygon(
        np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]]),
        closed=False,
        quadrature_order=8,
    )

    refined = refine(boundary, splitchunks=[1], lvlr="n", stype="t")

    assert refined.panel_count == 4
    np.testing.assert_allclose(refined.panel_lengths, [1.0, 0.5, 0.5, 1.0], atol=1.0e-14)
    np.testing.assert_array_equal(refined.adjacency[:, 0], [-1, 1])
    np.testing.assert_array_equal(refined.adjacency[:, -1], [2, -1])


def test_refine_level_restriction_splits_large_neighbor_panels():
    boundary = chunker_from_polygon(
        np.array([[0.0, 4.0, 5.0], [0.0, 0.0, 0.0]]),
        closed=False,
        quadrature_order=8,
    )

    refined = refine(boundary, splitchunks=[1], lvlr="a", lvlrfac=2.1, stype="t")
    adjacent_ratios = np.maximum(
        refined.panel_lengths[:-1] / refined.panel_lengths[1:],
        refined.panel_lengths[1:] / refined.panel_lengths[:-1],
    )

    assert refined.panel_count > 3
    assert np.max(adjacent_ratios) <= 2.1


def test_refine_maxchunklen_and_nover_split_panels():
    boundary = chunker_from_polygon(
        np.array([[0.0, 3.0], [0.0, 0.0]]),
        closed=False,
        quadrature_order=8,
    )

    max_length_refined = refine(boundary, maxchunklen=0.75, lvlr="n", stype="t")
    oversampled = refine(boundary, nover=2, lvlr="n", stype="t")

    assert np.max(max_length_refined.panel_lengths) <= 0.75 + 1.0e-14
    assert oversampled.panel_count == 4
    np.testing.assert_allclose(oversampled.panel_lengths, 0.75, atol=1.0e-14)


def test_refine_arclength_split_balances_child_panel_lengths():
    def curve(t):
        positions = np.vstack((t, t**2))
        derivatives = np.vstack((np.ones_like(t), 2.0 * t))
        second = np.vstack((np.zeros_like(t), 2.0 * np.ones_like(t)))
        return positions, derivatives, second

    boundary = chunker_from_curve(
        curve,
        closed=False,
        quadrature_order=16,
        min_panel_count=1,
        max_panel_count=1,
    )

    arclength_split = refine(boundary, splitchunks=[0], lvlr="n", stype="a")
    parameter_split = refine(boundary, splitchunks=[0], lvlr="n", stype="t")

    np.testing.assert_allclose(
        arclength_split.panel_lengths[0],
        arclength_split.panel_lengths[1],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert abs(parameter_split.panel_lengths[0] - parameter_split.panel_lengths[1]) > 1.0e-2


def test_change_quadrature_order_interpolates_geometry_and_panel_values():
    boundary = circle(radius=1.0, quadrature_order=16, panel_count=4)
    values = (1.0 + boundary._legendre_nodes - 2.0 * boundary._legendre_nodes**3)[
        None,
        :,
        None,
    ]
    values = np.repeat(values, boundary.panel_count, axis=2)

    upsampled, upsampled_values = change_quadrature_order(boundary, 24, values)

    assert upsampled.quadrature_order == 24
    assert upsampled.panel_count == boundary.panel_count
    breaks = np.linspace(0.0, 2.0 * np.pi, boundary.panel_count + 1)
    for panel_id, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
        half_width = 0.5 * (right - left)
        theta = 0.5 * (left + right) + half_width * upsampled._legendre_nodes
        expected_positions = np.vstack((np.cos(theta), np.sin(theta)))
        expected_derivatives = half_width * np.vstack((-np.sin(theta), np.cos(theta)))
        expected_second = -(half_width**2) * expected_positions

        np.testing.assert_allclose(
            upsampled.positions[:, :, panel_id], expected_positions, atol=1.0e-11
        )
        np.testing.assert_allclose(
            upsampled.derivatives[:, :, panel_id], expected_derivatives, atol=1.0e-11
        )
        np.testing.assert_allclose(
            upsampled.second_derivatives[:, :, panel_id], expected_second, atol=1.0e-11
        )
    np.testing.assert_allclose(upsampled.normals, upsampled.positions, atol=1.0e-11)
    np.testing.assert_allclose(upsampled.area, boundary.area, atol=1.0e-12)
    expected_values = (
        1.0 + upsampled._legendre_nodes[:, None] - 2.0 * upsampled._legendre_nodes[:, None] ** 3
    )
    expected_values = np.repeat(expected_values, upsampled.panel_count, axis=1)
    np.testing.assert_allclose(upsampled_values[0], expected_values, atol=1.0e-13)
