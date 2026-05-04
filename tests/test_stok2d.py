import numpy as np

from chunkie import PointInfo, kernel
from chunkie.chnk import stok2d


def test_stokes_kernel_shapes():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]), n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))

    assert stok2d.kern(1.1, src, targ, "s").shape == (6, 4)
    assert stok2d.kern(1.1, src, targ, "spres").shape == (3, 4)
    assert stok2d.kern(1.1, src, targ, "sgrad").shape == (12, 4)
    assert kernel("stok", "dtrac", 1.1).opdims == (2, 2)
    assert kernel("stok", "cpres", 1.1).opdims == (1, 2)
    assert kernel("stok", "cgrad", 1.1).opdims == (4, 2)


def test_stokes_dtrac_matches_pressure_gradient_stress_identity():
    rng = np.random.default_rng(8675309)
    src = PointInfo(r=rng.random((2, 1)), n=rng.random((2, 1)))
    targ = PointInfo(r=rng.random((2, 1)), n=rng.random((2, 1)))
    strengths = rng.random(2)
    mu = 1.1

    traction = stok2d.kern(mu, src, targ, "dtrac") @ strengths
    grad = stok2d.kern(mu, src, targ, "dgrad") @ strengths
    pressure = stok2d.kern(mu, src, targ, "dpres") @ strengths

    du = grad.reshape(2, 2)
    strain = du + du.T
    expected = np.array(
        [
            -pressure[0] * targ.n[0, 0] + mu * (strain[0, 0] * targ.n[0, 0] + strain[0, 1] * targ.n[1, 0]),
            -pressure[0] * targ.n[1, 0] + mu * (strain[1, 0] * targ.n[0, 0] + strain[1, 1] * targ.n[1, 0]),
        ]
    )

    np.testing.assert_allclose(traction, expected)


def test_stokes_combined_variants_match_linear_combinations():
    src = PointInfo(r=np.array([[0.0, 0.7], [0.0, -0.2]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[1.1, -0.4], [0.8, 0.5]]), n=np.array([[0.6, 0.8], [0.8, -0.6]]))
    coefs = np.array([1.5, -0.25])
    mu = 1.3

    np.testing.assert_allclose(
        stok2d.kern(mu, src, targ, "cvel", coefs),
        coefs[0] * stok2d.kern(mu, src, targ, "d") + coefs[1] * stok2d.kern(mu, src, targ, "s"),
    )
    np.testing.assert_allclose(
        stok2d.kern(mu, src, targ, "cpres", coefs),
        coefs[0] * stok2d.kern(mu, src, targ, "dpres") + coefs[1] * stok2d.kern(mu, src, targ, "spres"),
    )
    np.testing.assert_allclose(
        stok2d.kern(mu, src, targ, "ctrac", coefs),
        coefs[0] * stok2d.kern(mu, src, targ, "dtrac") + coefs[1] * stok2d.kern(mu, src, targ, "strac"),
    )
    np.testing.assert_allclose(
        stok2d.kern(mu, src, targ, "cgrad", coefs),
        coefs[0] * stok2d.kern(mu, src, targ, "dgrad") + coefs[1] * stok2d.kern(mu, src, targ, "sgrad"),
    )
