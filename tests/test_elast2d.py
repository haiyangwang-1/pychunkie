import numpy as np

from chunkie import PointInfo, kernel
from chunkie.kernels import elasticity as elast2d


def _interleave_2x2(kxx, kxy, kyx, kyy):
    nt, ns = kxx.shape
    out = np.zeros((2 * nt, 2 * ns), dtype=np.result_type(kxx, kxy, kyx, kyy))
    out[0::2, 0::2] = kxx
    out[0::2, 1::2] = kxy
    out[1::2, 0::2] = kyx
    out[1::2, 1::2] = kyy
    return out


def test_elasticity_kernel_shapes_and_factory():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(
        r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]),
        n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
    )
    lam = 1.5
    mu = 2.1

    beta = (lam + 3.0 * mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    gamma = -(lam + mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    eta = mu / (2.0 * np.pi * (lam + 2.0 * mu))
    zeta = (lam + mu) / (np.pi * (lam + 2.0 * mu))
    x = targ.r[0, :, None] - src.r[0, None, :]
    y = targ.r[1, :, None] - src.r[1, None, :]
    r2 = x**2 + y**2
    r4 = r2**2

    expected_s = _interleave_2x2(
        beta * np.log(r2) / 2.0 + gamma / 2.0 + gamma * x**2 / r2,
        gamma * x * y / r2,
        gamma * x * y / r2,
        beta * np.log(r2) / 2.0 + gamma / 2.0 + gamma * y**2 / r2,
    )
    rn_src = x * src.n[0, None, :] + y * src.n[1, None, :]
    expected_d = _interleave_2x2(
        -(eta * rn_src / r2 + zeta * rn_src * x**2 / r4),
        -(eta * (-x * src.n[1, None, :] + y * src.n[0, None, :]) / r2 + zeta * rn_src * x * y / r4),
        -(eta * (-y * src.n[0, None, :] + x * src.n[1, None, :]) / r2 + zeta * rn_src * x * y / r4),
        -(eta * rn_src / r2 + zeta * rn_src * y**2 / r4),
    )
    expected_dalt = _interleave_2x2(
        -2.0 * eta * rn_src / r2 - zeta * rn_src * x**2 / r4,
        -zeta * rn_src * x * y / r4,
        -zeta * rn_src * x * y / r4,
        -2.0 * eta * rn_src / r2 - zeta * rn_src * y**2 / r4,
    )

    single = elast2d.kernel(lam, mu, src, targ, "s")
    dalt = elast2d.kernel(lam, mu, src, targ, "dalt")
    np.testing.assert_allclose(single, expected_s)
    np.testing.assert_allclose(elast2d.kernel(lam, mu, src, targ, "d"), expected_d)
    np.testing.assert_allclose(dalt, expected_dalt)

    eps = 1.0e-6
    txp = PointInfo(r=targ.r + np.array([[eps], [0.0]]), n=targ.n)
    txm = PointInfo(r=targ.r - np.array([[eps], [0.0]]), n=targ.n)
    typ = PointInfo(r=targ.r + np.array([[0.0], [eps]]), n=targ.n)
    tym = PointInfo(r=targ.r - np.array([[0.0], [eps]]), n=targ.n)
    d_dx = (elast2d.kernel(lam, mu, src, txp, "s") - elast2d.kernel(lam, mu, src, txm, "s")) / (
        2.0 * eps
    )
    d_dy = (elast2d.kernel(lam, mu, src, typ, "s") - elast2d.kernel(lam, mu, src, tym, "s")) / (
        2.0 * eps
    )
    expected_sgrad = np.zeros((4 * targ.r.shape[1], 2 * src.r.shape[1]))
    expected_sgrad[0::4] = d_dx[0::2]
    expected_sgrad[1::4] = d_dy[0::2]
    expected_sgrad[2::4] = d_dx[1::2]
    expected_sgrad[3::4] = d_dy[1::2]
    sgrad = elast2d.kernel(lam, mu, src, targ, "sgrad")
    np.testing.assert_allclose(sgrad, expected_sgrad, rtol=1e-5, atol=1e-7)

    dens = np.array([0.4, -0.7, 0.2, 0.9])
    grad_applied = (sgrad @ dens).reshape(-1, 4)
    div = grad_applied[:, 0] + grad_applied[:, 3]
    shear = mu * (grad_applied[:, 1] + grad_applied[:, 2])
    expected_strac = np.vstack(
        (
            lam * targ.n[0] * div + shear * targ.n[1] + 2.0 * mu * grad_applied[:, 0] * targ.n[0],
            lam * targ.n[1] * div + shear * targ.n[0] + 2.0 * mu * grad_applied[:, 3] * targ.n[1],
        )
    ).T.reshape(-1)
    np.testing.assert_allclose(
        elast2d.kernel(lam, mu, src, targ, "strac") @ dens, expected_strac, rtol=1e-5, atol=1e-7
    )

    daltgrad = elast2d.kernel(lam, mu, src, targ, "daltgrad")
    dalt_grad_applied = (daltgrad @ dens).reshape(-1, 4)
    dalt_div = dalt_grad_applied[:, 0] + dalt_grad_applied[:, 3]
    dalt_shear = mu * (dalt_grad_applied[:, 1] + dalt_grad_applied[:, 2])
    expected_dalttrac = np.vstack(
        (
            lam * targ.n[0] * dalt_div
            + dalt_shear * targ.n[1]
            + 2.0 * mu * dalt_grad_applied[:, 0] * targ.n[0],
            lam * targ.n[1] * dalt_div
            + dalt_shear * targ.n[0]
            + 2.0 * mu * dalt_grad_applied[:, 3] * targ.n[1],
        )
    ).T.reshape(-1)
    np.testing.assert_allclose(
        elast2d.kernel(lam, mu, src, targ, "dalttrac") @ dens, expected_dalttrac
    )
    assert kernel("elast", "s", lam, mu).opdims == (2, 2)
    assert kernel("elast", "sgrad", lam, mu).opdims == (4, 2)


def test_elasticity_single_layer_is_symmetric_in_components():
    src = PointInfo(r=np.array([[0.0], [0.0]]))
    targ = PointInfo(r=np.array([[1.0], [2.0]]))
    mat = elast2d.kernel(1.5, 2.1, src, targ, "s")

    np.testing.assert_allclose(mat[0, 1], mat[1, 0])


def test_elasticity_single_gradient_matches_target_finite_difference():
    src = PointInfo(r=np.array([[0.2], [-0.4]]))
    targ = PointInfo(r=np.array([[1.1], [0.7]]))
    lam = 1.5
    mu = 2.1
    eps = 1.0e-6

    grad = elast2d.kernel(lam, mu, src, targ, "sgrad")
    tx = PointInfo(r=targ.r + np.array([[eps], [0.0]]))
    ty = PointInfo(r=targ.r + np.array([[0.0], [eps]]))
    base = elast2d.kernel(lam, mu, src, targ, "s")
    d_dx = (elast2d.kernel(lam, mu, src, tx, "s") - base) / eps
    d_dy = (elast2d.kernel(lam, mu, src, ty, "s") - base) / eps

    expected = np.vstack((d_dx[0], d_dy[0], d_dx[1], d_dy[1]))
    np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-7)


def test_elasticity_dalt_traction_is_gradient_traction():
    src = PointInfo(r=np.array([[0.2], [-0.4]]), n=np.array([[0.6], [0.8]]))
    targ = PointInfo(r=np.array([[1.1], [0.7]]), n=np.array([[-0.3], [0.95]]))
    targ.n = targ.n / np.linalg.norm(targ.n, axis=0)
    lam = 1.5
    mu = 2.1
    dens = np.array([0.4, -0.7])

    grad = (elast2d.kernel(lam, mu, src, targ, "daltgrad") @ dens).reshape(4, 1)
    traction = elast2d.kernel(lam, mu, src, targ, "dalttrac") @ dens
    div = grad[0] + grad[3]
    shear = mu * (grad[1] + grad[2])
    expected = np.array(
        [
            lam * targ.n[0, 0] * div[0]
            + shear[0] * targ.n[1, 0]
            + 2.0 * mu * grad[0, 0] * targ.n[0, 0],
            lam * targ.n[1, 0] * div[0]
            + shear[0] * targ.n[0, 0]
            + 2.0 * mu * grad[3, 0] * targ.n[1, 0],
        ]
    )

    np.testing.assert_allclose(traction, expected)
