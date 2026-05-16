import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential, dense_panel_matrix


def test_dense_panel_quadrature_applies_weights_at_quadrature_boundary():
    boundary = circle(quadrature_order=6, panel_count=4)
    target = np.array([[2.0], [0.0]])
    laplace_s = kernel("laplace", selector="s")
    density = np.ones((1, boundary.quadrature_order, boundary.panel_count))

    matrix = dense_panel_matrix(boundary.pointinfo, target, laplace_s)
    values = apply_panel_potential(boundary.pointinfo, target, laplace_s, density)

    assert matrix.shape == (1, 1, 1, boundary.point_count)
    np.testing.assert_allclose(values[0], np.sum(matrix[0, 0], axis=1))
