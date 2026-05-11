import numpy as np

from chunkie import lege


def test_exps_round_trips_values_and_coefficients():
    x, w, u, v = lege.exps(12)
    coeffs = np.arange(1, 13, dtype=float)
    vals = v @ coeffs

    np.testing.assert_allclose(u @ vals, coeffs, atol=1e-13)
    np.testing.assert_allclose(np.sum(w), 2.0, atol=1e-14)
    np.testing.assert_allclose(x, -x[::-1], atol=1e-14)


def test_rts_aliases_match_exps_nodes_weights():
    x, w, *_ = lege.exps(9)
    xr, wr = lege.rts(9)
    xs, ws = lege.rts_stab(9)

    np.testing.assert_allclose(xr, x)
    np.testing.assert_allclose(wr, w)
    np.testing.assert_allclose(xs, x)
    np.testing.assert_allclose(ws, w)


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


def test_exev_evaluates_single_and_multiple_expansions():
    xs = np.array([-0.5, 0.0, 0.5])
    coeff = np.array([1.0, 2.0, 3.0])
    expected = 1.0 + 2.0 * xs + 3.0 * 0.5 * (3.0 * xs**2 - 1.0)

    np.testing.assert_allclose(lege.exev(xs, coeff), expected)
    vals = lege.exev(xs, np.column_stack((coeff, 2.0 * coeff)))
    np.testing.assert_allclose(vals[:, 0], expected)
    np.testing.assert_allclose(vals[:, 1], 2.0 * expected)


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


def test_bernstein_ellipse_matches_conformal_map():
    z = lege.bernstein_ellipse(8, 2.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    expected = 0.5 * (2.0 * np.exp(1j * theta) + 0.5 * np.exp(-1j * theta))

    np.testing.assert_allclose(z, expected)
    np.testing.assert_allclose(z.real.max(), 1.25)
    np.testing.assert_allclose(z.real.min(), -1.25)


def test_polsum_matches_pols_and_recurrence_total():
    xs = np.array([-0.7, 0.0, 0.8])
    val, der, total = lege.polsum(xs, 5)
    vals, ders = lege.pols(xs, 5)
    expected_total = np.sum(vals**2 * (np.arange(6)[:, None] + 0.5), axis=0)

    np.testing.assert_allclose(val, vals[5], atol=1e-14)
    np.testing.assert_allclose(der, ders[5], atol=1e-14)
    np.testing.assert_allclose(total, expected_total, atol=1e-14)


def test_tayl_matches_direct_legendre_step():
    x = np.array([-0.4, 0.15, 0.55])
    h = np.array([0.02, -0.03, 0.015])
    pol0, der0 = lege.pol(x, 6)
    pol1, der1 = lege.tayl(pol0, der0, x, h, 6, 8)
    expected_pol, expected_der = lege.pol(x + h, 6)

    np.testing.assert_allclose(pol1, expected_pol, atol=1e-13)
    np.testing.assert_allclose(der1, expected_der, atol=1e-12)
    zpol, zder = lege.tayl(pol0, der0, x, np.zeros_like(h), 6, 8)
    np.testing.assert_allclose(zpol, pol0)
    np.testing.assert_allclose(zder, der0)


def test_adapgauss_integrates_scalar_and_vector_functions():
    val, maxrec, numint, ier = lege.adapgauss(lambda x: x**4, -1.0, 2.0)
    np.testing.assert_allclose(val, 33.0 / 5.0, atol=1e-12)
    assert maxrec >= 1
    assert numint >= 1
    assert ier == 0

    vec, *_ = lege.adapgauss(lambda x: np.column_stack((x, x**2)), 0.0, 1.0)
    np.testing.assert_allclose(vec, [0.5, 1.0 / 3.0], atol=1e-13)
