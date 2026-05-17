from types import SimpleNamespace

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


def test_laplace_derived_fmm_selectors_match_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = _target_points_with_normals()
    density = np.cos(boundary.positions[0])[None, :, :]

    for selector in ("sg", "sp", "dg", "dp"):
        laplace = kernel("laplace", selector=selector)
        dense = apply_panel_potential(boundary.pointinfo, targets, laplace, density)
        fmm = apply_fmm(boundary.pointinfo, targets, laplace, density)

        np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def test_helmholtz_single_layer_fmm_matches_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    density = np.cos(boundary.positions[0])[None, :, :]
    helmholtz_s = kernel("helmholtz", selector="s", wavenumber=1.4 + 0.1j)

    dense = apply_panel_potential(boundary.pointinfo, targets, helmholtz_s, density)
    fmm = apply_fmm(boundary.pointinfo, targets, helmholtz_s, density)

    np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def test_helmholtz_double_layer_fmm_matches_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    density = np.sin(boundary.positions[1])[None, :, :]
    helmholtz_d = kernel("helmholtz", selector="d", wavenumber=1.4 + 0.1j)

    dense = apply_panel_potential(boundary.pointinfo, targets, helmholtz_d, density)
    fmm = apply_fmm(boundary.pointinfo, targets, helmholtz_d, density)

    np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def test_helmholtz_derived_fmm_selectors_match_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = _target_points_with_normals()
    density = np.sin(boundary.positions[1])[None, :, :]

    for selector in ("sg", "sp", "dg", "dp"):
        helmholtz = kernel("helmholtz", selector=selector, wavenumber=1.4 + 0.1j)
        dense = apply_panel_potential(boundary.pointinfo, targets, helmholtz, density)
        fmm = apply_fmm(boundary.pointinfo, targets, helmholtz, density)

        np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def test_stokes_single_layer_fmm_matches_dense_evaluation():
    boundary = circle(quadrature_order=12, panel_count=20)
    targets = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    density = np.stack((np.cos(boundary.positions[0]), np.sin(boundary.positions[1])), axis=0)
    stokes_s = kernel("stokes", selector="s", viscosity=1.0)

    dense = apply_panel_potential(boundary.pointinfo, targets, stokes_s, density)
    fmm = apply_fmm(boundary.pointinfo, targets, stokes_s, density)

    np.testing.assert_allclose(fmm, dense, rtol=2.0e-11, atol=2.0e-12)


def _target_points_with_normals():
    positions = np.array([[1.8, -1.6, 0.2], [0.1, -0.3, 1.7]])
    normals = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]])
    normals = normals / np.linalg.norm(normals, axis=0, keepdims=True)
    return SimpleNamespace(positions=positions, normals=normals)
