import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import (
    apply_panel_potential,
    dense_panel_matrix,
    dense_panel_operator_matrix,
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
