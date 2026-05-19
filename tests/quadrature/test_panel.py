import numpy as np

from chunkie.geometry import chunker_from_polygon, circle
from chunkie.kernels import kernel
from chunkie.quadrature import (
    adaptive_panel_matrix,
    apply_panel_potential,
    dense_panel_matrix,
    dense_panel_operator_matrix,
    operator_matrix_from_weighted_kernel,
)


def test_dense_panel_quadrature_applies_weights_at_quadrature_boundary():
    boundary = circle(quadrature_order=6, panel_count=4)
    target = np.array([[2.0], [0.0]])
    laplace_s = kernel("laplace", selector="s")
    density = np.ones((1, boundary.quadrature_order, boundary.panel_count))

    matrix = dense_panel_matrix(boundary.pointinfo, target, laplace_s)
    values = apply_panel_potential(boundary.pointinfo, target, laplace_s, density)

    assert matrix.shape == (1, 1, 1, boundary.point_count)
    np.testing.assert_allclose(values[0], np.sum(matrix[0, 0], axis=1))


def test_dense_panel_operator_matrix_uses_component_major_solver_layout():
    boundary = circle(quadrature_order=5, panel_count=3)
    target = np.array([[2.0, -2.0], [0.0, 0.0]])
    stokes_s = kernel("stokes", selector="s", viscosity=1.5)
    density = np.arange(2 * boundary.quadrature_order * boundary.panel_count, dtype=float).reshape(
        2,
        boundary.quadrature_order,
        boundary.panel_count,
    )

    matrix = dense_panel_operator_matrix(boundary.pointinfo, target, stokes_s)
    vector = density.swapaxes(1, 2).reshape(-1)
    applied = matrix @ vector
    field = apply_panel_potential(boundary.pointinfo, target, stokes_s, density)

    assert matrix.shape == (2 * target.shape[1], 2 * boundary.point_count)
    np.testing.assert_allclose(applied.reshape(2, target.shape[1]), field)


def test_operator_matrix_materializes_component_major_kernel_tensor():
    weighted_kernel = np.empty((2, 2, 1, 3))
    weighted_kernel[0, 0, 0] = [1.0, 2.0, 3.0]
    weighted_kernel[0, 1, 0] = [4.0, 5.0, 6.0]
    weighted_kernel[1, 0, 0] = [7.0, 8.0, 9.0]
    weighted_kernel[1, 1, 0] = [10.0, 11.0, 12.0]

    matrix = operator_matrix_from_weighted_kernel(weighted_kernel)

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ],
        ),
    )


def test_adaptive_panel_matrix_integrates_close_straight_panel_log():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=8)
    panel = boundary.panel(0)
    target = np.array([[0.5], [1.0e-4]])
    laplace_s = kernel("laplace", selector="s")

    matrix = adaptive_panel_matrix(panel, target, laplace_s, tolerance=1.0e-11)
    value = np.sum(matrix[0, 0, 0])

    half_length = 0.5
    height = target[1, 0]
    exact_log_integral = (
        2.0 * half_length * np.log(half_length**2 + height**2)
        - 4.0 * half_length
        + 4.0 * height * np.arctan(half_length / height)
    )
    exact = -exact_log_integral / (4.0 * np.pi)

    assert matrix.shape == (1, 1, 1, panel.nodes.size)
    np.testing.assert_allclose(value, exact, rtol=2.0e-10, atol=2.0e-12)


def test_apply_panel_potential_replaces_close_panel_contribution():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=8)
    target = np.array([[0.5], [1.0e-4]])
    laplace_s = kernel("laplace", selector="s")
    density = np.zeros((1, boundary.quadrature_order, boundary.panel_count))
    density[0, :, 0] = 1.0

    direct = apply_panel_potential(boundary.pointinfo, target, laplace_s, density)
    corrected = apply_panel_potential(
        boundary.pointinfo,
        target,
        laplace_s,
        density,
        close_correction=True,
        near_factor=0.25,
        tolerance=1.0e-11,
    )

    half_length = 0.5
    height = target[1, 0]
    exact_log_integral = (
        2.0 * half_length * np.log(half_length**2 + height**2)
        - 4.0 * half_length
        + 4.0 * height * np.arctan(half_length / height)
    )
    exact = -exact_log_integral / (4.0 * np.pi)

    assert abs(corrected[0, 0] - exact) < abs(direct[0, 0] - exact)
    np.testing.assert_allclose(corrected[0, 0], exact, rtol=2.0e-10, atol=2.0e-12)
