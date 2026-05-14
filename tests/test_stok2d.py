import numpy as np

from chunkie import PointInfo, kernel
from chunkie.kernels import stokes as stok2d


def _interleave_2x2(kxx, kxy, kyx, kyy):
    nt, ns = kxx.shape
    out = np.zeros((2 * nt, 2 * ns), dtype=np.result_type(kxx, kxy, kyx, kyy))
    out[0::2, 0::2] = kxx
    out[0::2, 1::2] = kxy
    out[1::2, 0::2] = kyx
    out[1::2, 1::2] = kyy
    return out


def _interleave_1x2(kx, ky):
    nt, ns = kx.shape
    out = np.zeros((nt, 2 * ns), dtype=np.result_type(kx, ky))
    out[:, 0::2] = kx
    out[:, 1::2] = ky
    return out


def test_stokes_kernel_shapes():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(
        r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]),
        n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
    )

    mu = 1.1
    rx = targ.r[0, :, None] - src.r[0, None, :]
    ry = targ.r[1, :, None] - src.r[1, None, :]
    r2 = rx**2 + ry**2
    r = np.sqrt(r2)
    expected_s = _interleave_2x2(
        (rx**2 / r2 + np.log(1.0 / r)) / (4.0 * np.pi * mu),
        (rx * ry / r2) / (4.0 * np.pi * mu),
        (rx * ry / r2) / (4.0 * np.pi * mu),
        (ry**2 / r2 + np.log(1.0 / r)) / (4.0 * np.pi * mu),
    )
    expected_spres = _interleave_1x2(rx / (2.0 * np.pi * r2), ry / (2.0 * np.pi * r2))

    single = stok2d.kernel(mu, src, targ, "s")
    spres = stok2d.kernel(mu, src, targ, "spres")
    np.testing.assert_allclose(single, expected_s)
    np.testing.assert_allclose(spres, expected_spres)

    eps = 1.0e-6
    txp = PointInfo(r=targ.r + np.array([[eps], [0.0]]), n=targ.n)
    txm = PointInfo(r=targ.r - np.array([[eps], [0.0]]), n=targ.n)
    typ = PointInfo(r=targ.r + np.array([[0.0], [eps]]), n=targ.n)
    tym = PointInfo(r=targ.r - np.array([[0.0], [eps]]), n=targ.n)
    d_dx = (stok2d.kernel(mu, src, txp, "s") - stok2d.kernel(mu, src, txm, "s")) / (2.0 * eps)
    d_dy = (stok2d.kernel(mu, src, typ, "s") - stok2d.kernel(mu, src, tym, "s")) / (2.0 * eps)
    expected_sgrad = np.zeros((4 * targ.r.shape[1], 2 * src.r.shape[1]))
    expected_sgrad[0::4] = d_dx[0::2]
    expected_sgrad[1::4] = d_dy[0::2]
    expected_sgrad[2::4] = d_dx[1::2]
    expected_sgrad[3::4] = d_dy[1::2]
    np.testing.assert_allclose(
        stok2d.kernel(mu, src, targ, "sgrad"), expected_sgrad, rtol=1e-5, atol=1e-7
    )
    assert kernel("stok", "dtrac", 1.1).opdims == (2, 2)
    assert kernel("stok", "cpres", 1.1).opdims == (1, 2)
    assert kernel("stok", "cgrad", 1.1).opdims == (4, 2)


def test_stokes_dtrac_matches_pressure_gradient_stress_identity():
    rng = np.random.default_rng(8675309)
    src = PointInfo(r=rng.random((2, 1)), n=rng.random((2, 1)))
    targ = PointInfo(r=rng.random((2, 1)), n=rng.random((2, 1)))
    strengths = rng.random(2)
    mu = 1.1

    traction = stok2d.kernel(mu, src, targ, "dtrac") @ strengths
    grad = stok2d.kernel(mu, src, targ, "dgrad") @ strengths
    pressure = stok2d.kernel(mu, src, targ, "dpres") @ strengths

    du = grad.reshape(2, 2)
    strain = du + du.T
    expected = np.array(
        [
            -pressure[0] * targ.n[0, 0]
            + mu * (strain[0, 0] * targ.n[0, 0] + strain[0, 1] * targ.n[1, 0]),
            -pressure[0] * targ.n[1, 0]
            + mu * (strain[1, 0] * targ.n[0, 0] + strain[1, 1] * targ.n[1, 0]),
        ]
    )

    np.testing.assert_allclose(traction, expected)


def test_stokes_combined_variants_match_linear_combinations():
    src = PointInfo(r=np.array([[0.0, 0.7], [0.0, -0.2]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[1.1, -0.4], [0.8, 0.5]]), n=np.array([[0.6, 0.8], [0.8, -0.6]]))
    coefs = np.array([1.5, -0.25])
    mu = 1.3

    np.testing.assert_allclose(
        stok2d.kernel(mu, src, targ, "cvel", coefs),
        coefs[0] * stok2d.kernel(mu, src, targ, "d") + coefs[1] * stok2d.kernel(mu, src, targ, "s"),
    )
    np.testing.assert_allclose(
        stok2d.kernel(mu, src, targ, "cpres", coefs),
        coefs[0] * stok2d.kernel(mu, src, targ, "dpres")
        + coefs[1] * stok2d.kernel(mu, src, targ, "spres"),
    )
    np.testing.assert_allclose(
        stok2d.kernel(mu, src, targ, "ctrac", coefs),
        coefs[0] * stok2d.kernel(mu, src, targ, "dtrac")
        + coefs[1] * stok2d.kernel(mu, src, targ, "strac"),
    )
    np.testing.assert_allclose(
        stok2d.kernel(mu, src, targ, "cgrad", coefs),
        coefs[0] * stok2d.kernel(mu, src, targ, "dgrad")
        + coefs[1] * stok2d.kernel(mu, src, targ, "sgrad"),
    )
