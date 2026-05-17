import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import jv

from chunkie.geometry import chunker_from_polygon
from chunkie.kernels import kernel
from chunkie.quadrature import (
    build_helsing_ojala_panel_matrix,
    helsing_ojala_log_singular_matrix,
    helsing_ojala_weights,
)


def test_helsing_ojala_log_weights_match_oversampled_legendre_moments():
    source_count = 16
    nodes, weights = leggauss(source_count)
    source = nodes.astype(complex)
    normal = -1j * np.ones(source_count)
    source_wxp = weights.astype(complex)
    target = np.array([0.0 + 0.2j])

    special = helsing_ojala_weights(target, source, normal, source_wxp, -1.0, 1.0, "i", nout=4)

    reference_nodes, reference_weights = leggauss(800)
    reference_z = reference_nodes.astype(complex)
    for degree in (0, 1, 3, 7):
        node_values = nodes**degree
        reference_values = reference_nodes**degree
        expected = (
            np.sum(-np.log(np.abs(reference_z - target[0])) / (2.0 * np.pi) * reference_values * reference_weights),
            np.sum(1j / (2.0 * np.pi) * reference_values / (reference_z - target[0]) * reference_weights),
            np.sum(1j / (2.0 * np.pi) * reference_values / (reference_z - target[0]) ** 2 * reference_weights),
            np.sum(1j / (2.0 * np.pi) * reference_values / (reference_z - target[0]) ** 3 * reference_weights),
        )
        for weights0, expected0 in zip(special, expected, strict=True):
            actual = weights0 @ node_values
            np.testing.assert_allclose(actual, expected0, rtol=1.0e-12, atol=1.0e-12)


def test_helsing_ojala_panel_matrix_integrates_close_laplace_single_layer():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=12)
    panel = boundary.panel(0)
    target = np.array([[0.5], [1.0e-4]])
    laplace_s = kernel("laplace", selector="s")

    matrix = build_helsing_ojala_panel_matrix(panel, target, laplace_s, side="i")
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
    np.testing.assert_allclose(value, exact, rtol=1.0e-12, atol=1.0e-12)


def test_helsing_ojala_log_singular_matrix_consumes_smooth_amplitudes():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=10)
    panel = boundary.panel(0)
    target = np.array([[0.5], [2.0e-3]])
    wavenumber = 1.7

    helmholtz_s = kernel("helmholtz", selector="s", wavenumber=wavenumber)
    laplace_s = kernel("laplace", selector="s")
    singular = helsing_ojala_log_singular_matrix(panel, target, helmholtz_s, side="i")
    laplace_weights = build_helsing_ojala_panel_matrix(panel, target, laplace_s, side="i")

    source_positions = panel.positions
    distances = np.linalg.norm(target[:, :, None] - source_positions[:, None, :], axis=0)
    expected = jv(0, wavenumber * distances)[None, None, :, :] * laplace_weights

    assert singular.shape == (1, 1, 1, panel.nodes.size)
    np.testing.assert_allclose(singular, expected, rtol=1.0e-13, atol=1.0e-13)
