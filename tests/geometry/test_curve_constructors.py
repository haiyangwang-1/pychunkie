import numpy as np

from chunkie.geometry import chunker_from_curve


def _circle_curve(theta, *, radius=1.0, center=(0.0, 0.0)):
    theta = np.asarray(theta)
    center_array = np.asarray(center, dtype=float).reshape(2, 1)
    positions = center_array + radius * np.vstack((np.cos(theta), np.sin(theta)))
    derivatives = radius * np.vstack((-np.sin(theta), np.cos(theta)))
    second_derivatives = -radius * np.vstack((np.cos(theta), np.sin(theta)))
    return positions, derivatives, second_derivatives


def test_chunker_from_curve_builds_closed_circle_panels():
    radius = 2.5
    boundary = chunker_from_curve(
        lambda theta: _circle_curve(theta, radius=radius),
        quadrature_order=12,
        min_panel_count=4,
    )
    breaks = np.linspace(0.0, 2.0 * np.pi, 5)

    assert boundary.closed
    assert boundary.panel_count == 4
    assert boundary.orientation == "ccw"
    np.testing.assert_array_equal(boundary.adjacency[:, 0], [3, 1])
    np.testing.assert_array_equal(boundary.adjacency[:, -1], [2, 0])
    np.testing.assert_allclose(boundary.area, np.pi * radius**2, atol=1.0e-12)
    np.testing.assert_allclose(boundary.length, 2.0 * np.pi * radius, atol=1.0e-12)

    for panel_id, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
        half_width = 0.5 * (right - left)
        theta = 0.5 * (left + right) + half_width * boundary._legendre_nodes
        expected_positions, global_derivatives, global_second = _circle_curve(theta, radius=radius)
        np.testing.assert_allclose(boundary.positions[:, :, panel_id], expected_positions, atol=1.0e-12)
        np.testing.assert_allclose(boundary.derivatives[:, :, panel_id], half_width * global_derivatives)
        np.testing.assert_allclose(boundary.second_derivatives[:, :, panel_id], half_width**2 * global_second)
        np.testing.assert_allclose(boundary.normals[:, :, panel_id], expected_positions / radius)
        np.testing.assert_allclose(
            boundary.weights[:, panel_id],
            radius * half_width * boundary._legendre_weights,
        )


def test_chunker_from_curve_builds_open_line_with_free_ends():
    def line(t):
        t = np.asarray(t)
        positions = np.vstack((2.0 * t, np.zeros_like(t)))
        derivatives = np.repeat([[2.0], [0.0]], t.size, axis=1)
        second = np.zeros_like(positions)
        return positions, derivatives, second

    boundary = chunker_from_curve(line, closed=False, quadrature_order=8, min_panel_count=2)

    assert not boundary.closed
    assert boundary.orientation == "open"
    np.testing.assert_array_equal(boundary.adjacency[:, 0], [-1, 1])
    np.testing.assert_array_equal(boundary.adjacency[:, 1], [0, -1])
    np.testing.assert_allclose(boundary.length, 2.0, atol=1.0e-13)
    np.testing.assert_allclose(boundary.panel_lengths, [1.0, 1.0], atol=1.0e-13)
    np.testing.assert_allclose(boundary.derivatives[0], 0.5)
    np.testing.assert_allclose(boundary.derivatives[1], 0.0)
    np.testing.assert_allclose(boundary.second_derivatives, 0.0)
    np.testing.assert_allclose(boundary.normals[0], 0.0)
    np.testing.assert_allclose(boundary.normals[1], -1.0)


def test_chunker_from_curve_differentiates_position_only_callback():
    def position_only(theta):
        theta = np.asarray(theta)
        return np.vstack((np.cos(theta), np.sin(theta)))

    boundary = chunker_from_curve(position_only, quadrature_order=14, min_panel_count=4)
    breaks = np.linspace(0.0, 2.0 * np.pi, 5)

    for panel_id, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
        half_width = 0.5 * (right - left)
        theta = 0.5 * (left + right) + half_width * boundary._legendre_nodes
        expected_derivatives = half_width * np.vstack((-np.sin(theta), np.cos(theta)))
        expected_second = -(half_width**2) * np.vstack((np.cos(theta), np.sin(theta)))

        np.testing.assert_allclose(boundary.derivatives[:, :, panel_id], expected_derivatives, atol=2.0e-10)
        np.testing.assert_allclose(boundary.second_derivatives[:, :, panel_id], expected_second, atol=5.0e-4)
    np.testing.assert_allclose(np.linalg.norm(boundary.normals, axis=0), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(boundary.positions * boundary.normals, axis=0), 1.0, atol=1.0e-10)


def test_chunker_from_curve_adaptively_refines_unresolved_open_curve():
    frequency = 24.0 * np.pi

    def wavy(t):
        t = np.asarray(t)
        positions = np.vstack((t, 0.05 * np.sin(frequency * t)))
        derivatives = np.vstack((np.ones_like(t), 0.05 * frequency * np.cos(frequency * t)))
        second = np.vstack((np.zeros_like(t), -0.05 * frequency**2 * np.sin(frequency * t)))
        return positions, derivatives, second

    coarse = chunker_from_curve(
        wavy,
        closed=False,
        quadrature_order=8,
        min_panel_count=1,
        max_panel_count=1,
    )
    refined = chunker_from_curve(
        wavy,
        closed=False,
        quadrature_order=8,
        min_panel_count=1,
        max_panel_count=256,
        tolerance=1.0e-6,
    )
    intervals = refined.metadata["parameter_intervals"]

    assert coarse.panel_count == 1
    assert refined.panel_count > coarse.panel_count
    np.testing.assert_allclose(intervals[0, 0], 0.0)
    np.testing.assert_allclose(intervals[1, -1], 1.0)
    np.testing.assert_allclose(intervals[1, :-1], intervals[0, 1:])
    np.testing.assert_allclose(refined.panel_lengths, np.sum(refined.weights, axis=0), atol=1.0e-14)

    nodes, weights = np.polynomial.legendre.leggauss(2000)
    parameters = 0.5 * (nodes + 1.0)
    reference_length = 0.5 * np.sum(
        weights * np.sqrt(1.0 + (0.05 * frequency * np.cos(frequency * parameters)) ** 2),
    )
    np.testing.assert_allclose(refined.length, reference_length, rtol=5.0e-8, atol=5.0e-9)
