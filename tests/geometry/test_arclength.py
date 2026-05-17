import numpy as np

from chunkie.geometry import (
    arclength_parameterization,
    circle,
    evaluate_arclength,
    resample_by_arclength,
)


def test_arclength_parameterization_evaluates_original_circle_nodes():
    boundary = circle(radius=1.25, center=(2.0, -0.5), quadrature_order=16, panel_count=5)
    parameterization = arclength_parameterization(boundary)
    arclengths = parameterization.node_arclengths.T.reshape(-1)

    positions, tangents, second_arclength_derivatives = evaluate_arclength(parameterization, arclengths)

    np.testing.assert_allclose(positions, boundary.pointinfo.flat_positions, atol=1.0e-11)
    np.testing.assert_allclose(tangents, boundary.tangents.swapaxes(1, 2).reshape(2, -1), atol=1.0e-11)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=0), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.sum(tangents * second_arclength_derivatives, axis=0), 0.0, atol=1.0e-11)


def test_arclength_derivatives_match_circle_formulas():
    radius = 2.0
    boundary = circle(radius=radius, quadrature_order=18, panel_count=6)
    parameterization = arclength_parameterization(boundary)
    arclengths = np.linspace(0.1, boundary.length - 0.1, 30)

    positions, tangents, second_arclength_derivatives = evaluate_arclength(parameterization, arclengths)

    theta = arclengths / radius
    expected_positions = radius * np.vstack((np.cos(theta), np.sin(theta)))
    expected_tangents = np.vstack((-np.sin(theta), np.cos(theta)))
    np.testing.assert_allclose(positions, expected_positions, atol=1.0e-10)
    np.testing.assert_allclose(tangents, expected_tangents, atol=1.0e-10)
    np.testing.assert_allclose(second_arclength_derivatives, -expected_positions / radius**2, atol=1.0e-10)


def test_arclength_resample_makes_panel_speed_constant():
    boundary = circle(radius=1.5, quadrature_order=16, panel_count=5)

    resampled = resample_by_arclength(boundary, panel_count=7)

    expected_speed = np.full((resampled.quadrature_order, resampled.panel_count), resampled.length / 14.0)
    np.testing.assert_allclose(resampled.area, boundary.area, atol=1.0e-10)
    np.testing.assert_allclose(resampled.length, boundary.length, atol=1.0e-10)
    np.testing.assert_allclose(resampled.arclength_density, expected_speed, atol=1.0e-12)
    np.testing.assert_allclose(np.sqrt(np.sum(resampled.positions**2, axis=0)), 1.5, atol=1.0e-10)
    np.testing.assert_allclose(np.sum(resampled.positions * resampled.derivatives, axis=0), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(resampled.normals, resampled.positions / 1.5, atol=1.0e-10)
    np.testing.assert_allclose(resampled.signed_curvature, 1.0 / 1.5, atol=1.0e-9)
