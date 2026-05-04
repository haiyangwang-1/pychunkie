import numpy as np

from chunkie import lege


def test_exps_round_trips_values_and_coefficients():
    x, w, u, v = lege.exps(12)
    coeffs = np.arange(1, 13, dtype=float)
    vals = v @ coeffs

    np.testing.assert_allclose(u @ vals, coeffs, atol=1e-13)
    np.testing.assert_allclose(np.sum(w), 2.0, atol=1e-14)
    np.testing.assert_allclose(x, -x[::-1], atol=1e-14)


def test_pols_matches_known_low_order_polynomials():
    xs = np.array([-0.75, -0.1, 0.2, 0.9])
    vals, ders = lege.pols(xs, 4)

    np.testing.assert_allclose(vals[0], 1.0)
    np.testing.assert_allclose(vals[1], xs)
    np.testing.assert_allclose(vals[2], 0.5 * (3 * xs**2 - 1))
    np.testing.assert_allclose(vals[3], 0.5 * (5 * xs**3 - 3 * xs))
    np.testing.assert_allclose(vals[4], (35 * xs**4 - 30 * xs**2 + 3) / 8)
    np.testing.assert_allclose(ders[3], 0.5 * (15 * xs**2 - 3))


def test_dermat_differentiates_node_values():
    x, _, _, v = lege.exps(10)
    dmat = lege.dermat(10)
    coeffs = np.zeros(10)
    coeffs[4] = 1.0

    expected = v[:, :-1] @ lege.derpol(coeffs[:, None]).ravel()
    np.testing.assert_allclose(dmat @ (v @ coeffs), expected, atol=1e-12)
    np.testing.assert_allclose(dmat @ np.ones_like(x), 0.0, atol=1e-12)


def test_matrin_interpolates_legendre_node_values():
    n = 8
    x, _, _, v = lege.exps(n)
    coeffs = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0, 0.25, 0.75])
    node_values = v @ coeffs

    targets = np.array([-0.95, -0.2, 0.0, 0.6])
    mat, *_ = lege.matrin(n, targets)
    target_values = np.moveaxis(lege.pols(targets, n - 1)[0], 0, -1) @ coeffs

    np.testing.assert_allclose(mat @ node_values, target_values, atol=1e-12)
    np.testing.assert_allclose(lege.matrin(n, x)[0] @ node_values, node_values, atol=1e-12)


def test_intpol_and_intmat_integrate_constants_from_left_endpoint():
    coeffs = np.array([2.0])
    icoeffs = lege.intpol(coeffs)
    vals, _ = lege.pols(np.array([-1.0, 0.0, 1.0]), 1)
    np.testing.assert_allclose(np.moveaxis(vals, 0, -1) @ icoeffs, [0.0, 2.0, 4.0])

    x, _, _, v = lege.exps(6)
    aint, _, _ = lege.intmat(6)
    np.testing.assert_allclose(aint @ np.ones(6), x + 1.0, atol=1e-12)


def test_barywts_reproduce_lagrange_basis_sign_pattern():
    x, *_ = lege.exps(7)
    w = lege.barywts(7, x)
    assert w.shape == (7,)
    assert np.all(np.sign(w[:-1]) != np.sign(w[1:]))
