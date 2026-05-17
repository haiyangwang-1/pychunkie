import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.quadrature import apply_panel_potential
from chunkie.system.backends.fmm2d import apply_fmm


def test_laplace_single_layer_fmm_matches_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    density = np.cos(boundary.positions[0])[None, :, :]
    laplace_s = kernel("laplace", selector="s")

    dense = apply_panel_potential(boundary.pointinfo, targets, laplace_s, density)
    fmm = apply_fmm(boundary.pointinfo, targets, laplace_s, density)

    np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def test_laplace_double_layer_fmm_matches_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    density = np.sin(boundary.positions[1])[None, :, :]
    laplace_d = kernel("laplace", selector="d")

    dense = apply_panel_potential(boundary.pointinfo, targets, laplace_d, density)
    fmm = apply_fmm(boundary.pointinfo, targets, laplace_d, density)

    np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)
