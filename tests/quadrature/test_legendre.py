import numpy as np

from chunkie.quadrature import legendre as lege


def test_legendre_exps_round_trips_values_and_coefficients():
    nodes, weights, coefficient_transform, value_transform = lege.exps(12)
    coefficients = np.arange(1, 13, dtype=float)
    values = value_transform @ coefficients

    np.testing.assert_allclose(coefficient_transform @ values, coefficients, atol=1.0e-13)
    np.testing.assert_allclose(np.sum(weights), 2.0, atol=1.0e-14)
    np.testing.assert_allclose(nodes, -nodes[::-1], atol=1.0e-14)


def test_legendre_polynomials_and_derivatives_match_low_order_formulas():
    points = np.array([-0.75, -0.1, 0.2, 0.9])
    values, derivatives = lege.pols(points, 4)

    np.testing.assert_allclose(values[0], 1.0)
    np.testing.assert_allclose(values[1], points)
    np.testing.assert_allclose(values[2], 0.5 * (3.0 * points**2 - 1.0))
    np.testing.assert_allclose(values[3], 0.5 * (5.0 * points**3 - 3.0 * points))
    np.testing.assert_allclose(values[4], (35.0 * points**4 - 30.0 * points**2 + 3.0) / 8.0)
    np.testing.assert_allclose(derivatives[3], 0.5 * (15.0 * points**2 - 3.0))


def test_legendre_derivative_matrix_differentiates_node_values():
    _, _, _, value_transform = lege.exps(10)
    differentiation = lege.dermat(10)
    coefficients = np.zeros(10)
    coefficients[4] = 1.0

    expected = value_transform[:, :-1] @ lege.derpol(coefficients[:, None]).ravel()
    np.testing.assert_allclose(differentiation @ (value_transform @ coefficients), expected, atol=1.0e-12)
    np.testing.assert_allclose(differentiation @ np.ones(10), 0.0, atol=1.0e-12)


def test_legendre_interpolation_and_expansion_evaluation():
    nodes, _, _, value_transform = lege.exps(8)
    coefficients = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0, 0.25, 0.75])
    node_values = value_transform @ coefficients
    targets = np.array([-0.95, -0.2, 0.0, 0.6])

    interpolation = lege.matrin(8, targets)[0]
    target_values = lege.exev(targets, coefficients)

    np.testing.assert_allclose(interpolation @ node_values, target_values, atol=1.0e-12)
    np.testing.assert_allclose(lege.matrin(8, nodes)[0] @ node_values, node_values, atol=1.0e-12)


def test_legendre_integration_matrix_integrates_from_left_endpoint():
    integrated_coefficients = lege.intpol(np.array([2.0]))
    values, _ = lege.pols(np.array([-1.0, 0.0, 1.0]), 1)
    np.testing.assert_allclose(np.moveaxis(values, 0, -1) @ integrated_coefficients, [0.0, 2.0, 4.0])

    nodes, *_ = lege.exps(6)
    integration_matrix = lege.intmat(6)[0]
    np.testing.assert_allclose(integration_matrix @ np.ones(6), nodes + 1.0, atol=1.0e-12)


def test_legendre_barycentric_weights_interpolate_polynomials():
    nodes, *_ = lege.exps(7)
    weights = lege.barywts(7, points=nodes)
    node_values = nodes**4 - 0.25 * nodes**2 + 0.5 * nodes - 2.0
    targets = np.array([-0.8, -0.15, 0.6])

    numerator = np.sum(
        weights[None, :] * node_values[None, :] / (targets[:, None] - nodes[None, :]),
        axis=1,
    )
    denominator = np.sum(weights[None, :] / (targets[:, None] - nodes[None, :]), axis=1)

    assert weights.shape == (7,)
    assert np.all(np.sign(weights[:-1]) != np.sign(weights[1:]))
    np.testing.assert_allclose(
        numerator / denominator,
        targets**4 - 0.25 * targets**2 + 0.5 * targets - 2.0,
    )


def test_legendre_bernstein_polsum_taylor_and_adaptive_gauss():
    ellipse = lege.bernstein_ellipse(8, 2.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    np.testing.assert_allclose(ellipse, 0.5 * (2.0 * np.exp(1j * theta) + 0.5 * np.exp(-1j * theta)))

    points = np.array([-0.7, 0.0, 0.8])
    value, derivative, total = lege.polsum(points, 5)
    values, derivatives = lege.pols(points, 5)
    expected_total = np.sum(values**2 * (np.arange(6)[:, None] + 0.5), axis=0)
    np.testing.assert_allclose(value, values[5], atol=1.0e-14)
    np.testing.assert_allclose(derivative, derivatives[5], atol=1.0e-14)
    np.testing.assert_allclose(total, expected_total, atol=1.0e-14)

    step = np.array([0.02, -0.03, 0.015])
    polynomial, polynomial_derivative = lege.pol(points, 6)
    moved, moved_derivative = lege.tayl(polynomial, polynomial_derivative, points, step, degree=6, taylor_order=8)
    expected_polynomial, expected_derivative = lege.pol(points + step, 6)
    np.testing.assert_allclose(moved, expected_polynomial, atol=1.0e-13)
    np.testing.assert_allclose(moved_derivative, expected_derivative, atol=1.0e-12)

    integral, max_recursion, interval_count, error_code = lege.adapgauss(lambda x: x**4, -1.0, 2.0)
    np.testing.assert_allclose(integral, 33.0 / 5.0, atol=1.0e-12)
    assert max_recursion >= 1
    assert interval_count >= 1
    assert error_code == 0

    vector_integral, *_ = lege.adapgauss(lambda x: np.column_stack((x, x**2)), 0.0, 1.0)
    np.testing.assert_allclose(vector_integral, [0.5, 1.0 / 3.0], atol=1.0e-13)
