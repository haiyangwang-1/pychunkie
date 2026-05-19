import numpy as np

from chunkie.geometry import affine, circle, reflect, rotate, scale, translate


def test_circle_metrics_match_analytic_geometry():
    boundary = circle(radius=2.5, quadrature_order=18, panel_count=6)
    tangents = boundary.tangents
    expected_panel_length = 2.0 * np.pi * 2.5 / 6

    np.testing.assert_allclose(
        boundary.weights,
        np.linalg.norm(boundary.derivatives, axis=0) * boundary._legendre_weights[:, None],
        atol=1.0e-13,
    )

    np.testing.assert_allclose(
        boundary.panel_lengths, np.full(6, expected_panel_length), atol=1.0e-13
    )
    np.testing.assert_allclose(boundary.length, 2.0 * np.pi * 2.5, atol=1.0e-13)
    np.testing.assert_allclose(boundary.area, np.pi * 2.5**2, atol=1.0e-13)
    np.testing.assert_allclose(boundary.arclength_density, 2.5 * np.pi / 6.0, atol=1.0e-13)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=0), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(np.sum(tangents * boundary.normals, axis=0), 0.0, atol=1.0e-13)
    np.testing.assert_allclose(boundary.signed_curvature, 1.0 / 2.5, atol=1.0e-13)


def test_panel_endpoint_and_bounds_diagnostics():
    boundary = circle(radius=1.0, quadrature_order=18, panel_count=4)
    endpoints = boundary.panel_endpoints
    endpoint_tangents = boundary.panel_endpoint_tangents
    expected_angles = np.linspace(0.0, 2.0 * np.pi, 5)
    expected_starts = np.vstack((np.cos(expected_angles[:-1]), np.sin(expected_angles[:-1])))
    expected_ends = np.vstack((np.cos(expected_angles[1:]), np.sin(expected_angles[1:])))
    expected_start_tangents = np.vstack(
        (-np.sin(expected_angles[:-1]), np.cos(expected_angles[:-1]))
    )
    expected_end_tangents = np.vstack((-np.sin(expected_angles[1:]), np.cos(expected_angles[1:])))
    bounds_min, bounds_max = boundary.bounds

    np.testing.assert_allclose(endpoints[:, 0, :], expected_starts, atol=1.0e-12)
    np.testing.assert_allclose(endpoints[:, 1, :], expected_ends, atol=1.0e-12)
    np.testing.assert_allclose(endpoint_tangents[:, 0, :], expected_start_tangents, atol=1.0e-12)
    np.testing.assert_allclose(endpoint_tangents[:, 1, :], expected_end_tangents, atol=1.0e-12)
    np.testing.assert_allclose(bounds_min, [-1.0, -1.0], atol=1.0e-2)
    np.testing.assert_allclose(bounds_max, [1.0, 1.0], atol=1.0e-2)


def test_translation_and_uniform_scaling_preserve_layout_conventions():
    boundary = circle(radius=1.25, quadrature_order=12, panel_count=5)
    moved = translate(boundary, [1.0, -2.0])
    scaled = scale(boundary, 3.0, center=[0.25, -0.5])
    inverted = scale(boundary, -2.0)

    np.testing.assert_allclose(
        moved.positions, boundary.positions + np.array([1.0, -2.0])[:, None, None]
    )
    np.testing.assert_allclose(moved.derivatives, boundary.derivatives)
    np.testing.assert_allclose(moved.second_derivatives, boundary.second_derivatives)
    np.testing.assert_allclose(moved.normals, boundary.normals)
    np.testing.assert_allclose(moved.weights, boundary.weights)
    np.testing.assert_allclose(moved.area, boundary.area + 0.0, atol=1.0e-13)

    center = np.array([0.25, -0.5])[:, None, None]
    np.testing.assert_allclose(scaled.positions, center + 3.0 * (boundary.positions - center))
    np.testing.assert_allclose(scaled.derivatives, 3.0 * boundary.derivatives)
    np.testing.assert_allclose(scaled.second_derivatives, 3.0 * boundary.second_derivatives)
    np.testing.assert_allclose(scaled.normals, boundary.normals)
    np.testing.assert_allclose(scaled.weights, 3.0 * boundary.weights)
    np.testing.assert_allclose(scaled.area, 9.0 * boundary.area, atol=1.0e-12)

    np.testing.assert_allclose(inverted.positions, -2.0 * boundary.positions)
    np.testing.assert_allclose(inverted.derivatives, -2.0 * boundary.derivatives)
    np.testing.assert_allclose(inverted.normals, -boundary.normals)
    np.testing.assert_allclose(inverted.weights, 2.0 * boundary.weights)
    np.testing.assert_allclose(inverted.area, 4.0 * boundary.area, atol=1.0e-13)


def test_affine_rotation_and_reflection_update_normals_weights_and_area():
    boundary = circle(radius=1.0, quadrature_order=10, panel_count=4)
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]])
    transformed = affine(boundary, matrix, offset=[0.5, -0.25])
    expected_positions = (
        np.einsum("ij,jsS->isS", matrix, boundary.positions) + np.array([0.5, -0.25])[:, None, None]
    )
    expected_derivatives = np.einsum("ij,jsS->isS", matrix, boundary.derivatives)
    expected_speed = np.linalg.norm(expected_derivatives, axis=0)
    expected_normals = np.stack(
        (expected_derivatives[1] / expected_speed, -expected_derivatives[0] / expected_speed)
    )

    np.testing.assert_allclose(transformed.positions, expected_positions)
    np.testing.assert_allclose(transformed.derivatives, expected_derivatives)
    np.testing.assert_allclose(
        transformed.second_derivatives,
        np.einsum("ij,jsS->isS", matrix, boundary.second_derivatives),
    )
    np.testing.assert_allclose(transformed.normals, expected_normals)
    np.testing.assert_allclose(
        transformed.weights, expected_speed * boundary._legendre_weights[:, None]
    )
    np.testing.assert_allclose(
        transformed.area, np.linalg.det(matrix) * boundary.area, atol=1.0e-12
    )

    angle = np.pi / 3.0
    source_center = np.array([0.25, -0.5])
    target_center = np.array([1.0, 2.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = rotate(boundary, angle, center=source_center, target_center=target_center)
    np.testing.assert_allclose(
        rotated.positions,
        np.einsum("ij,jsS->isS", rotation, boundary.positions - source_center[:, None, None])
        + target_center[:, None, None],
    )
    np.testing.assert_allclose(rotated.weights, boundary.weights)
    np.testing.assert_allclose(rotated.area, boundary.area, atol=1.0e-13)

    reflection_angle = np.pi / 4.0
    reflection = np.array(
        [
            [np.cos(2.0 * reflection_angle), np.sin(2.0 * reflection_angle)],
            [np.sin(2.0 * reflection_angle), -np.cos(2.0 * reflection_angle)],
        ],
    )
    reflected = reflect(
        boundary, reflection_angle, center=source_center, target_center=target_center
    )
    np.testing.assert_allclose(
        reflected.positions,
        np.einsum("ij,jsS->isS", reflection, boundary.positions - source_center[:, None, None])
        + target_center[:, None, None],
    )
    np.testing.assert_allclose(reflected.weights, boundary.weights)
    np.testing.assert_allclose(reflected.area, -boundary.area, atol=1.0e-13)
    assert reflected.orientation == "cw"
