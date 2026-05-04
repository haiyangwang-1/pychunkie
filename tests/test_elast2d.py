import numpy as np

from chunkie import PointInfo, kernel
from chunkie.chnk import elast2d


def test_elasticity_kernel_shapes_and_factory():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]), n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    lam = 1.5
    mu = 2.1

    assert elast2d.kern(lam, mu, src, targ, "s").shape == (6, 4)
    assert elast2d.kern(lam, mu, src, targ, "strac").shape == (6, 4)
    assert elast2d.kern(lam, mu, src, targ, "sgrad").shape == (12, 4)
    assert elast2d.kern(lam, mu, src, targ, "d").shape == (6, 4)
    assert elast2d.kern(lam, mu, src, targ, "dalt").shape == (6, 4)
    assert elast2d.kern(lam, mu, src, targ, "daltgrad").shape == (12, 4)
    assert elast2d.kern(lam, mu, src, targ, "dalttrac").shape == (6, 4)
    assert kernel("elast", "s", lam, mu).opdims == (2, 2)
    assert kernel("elast", "sgrad", lam, mu).opdims == (4, 2)


def test_elasticity_single_layer_is_symmetric_in_components():
    src = PointInfo(r=np.array([[0.0], [0.0]]))
    targ = PointInfo(r=np.array([[1.0], [2.0]]))
    mat = elast2d.kern(1.5, 2.1, src, targ, "s")

    np.testing.assert_allclose(mat[0, 1], mat[1, 0])


def test_elasticity_single_gradient_matches_target_finite_difference():
    src = PointInfo(r=np.array([[0.2], [-0.4]]))
    targ = PointInfo(r=np.array([[1.1], [0.7]]))
    lam = 1.5
    mu = 2.1
    eps = 1.0e-6

    grad = elast2d.kern(lam, mu, src, targ, "sgrad")
    tx = PointInfo(r=targ.r + np.array([[eps], [0.0]]))
    ty = PointInfo(r=targ.r + np.array([[0.0], [eps]]))
    base = elast2d.kern(lam, mu, src, targ, "s")
    d_dx = (elast2d.kern(lam, mu, src, tx, "s") - base) / eps
    d_dy = (elast2d.kern(lam, mu, src, ty, "s") - base) / eps

    expected = np.vstack((d_dx[0], d_dy[0], d_dx[1], d_dy[1]))
    np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-7)


def test_elasticity_dalt_traction_is_gradient_traction():
    src = PointInfo(r=np.array([[0.2], [-0.4]]), n=np.array([[0.6], [0.8]]))
    targ = PointInfo(r=np.array([[1.1], [0.7]]), n=np.array([[-0.3], [0.95]]))
    targ.n = targ.n / np.linalg.norm(targ.n, axis=0)
    lam = 1.5
    mu = 2.1
    dens = np.array([0.4, -0.7])

    grad = (elast2d.kern(lam, mu, src, targ, "daltgrad") @ dens).reshape(4, 1)
    traction = elast2d.kern(lam, mu, src, targ, "dalttrac") @ dens
    div = grad[0] + grad[3]
    shear = mu * (grad[1] + grad[2])
    expected = np.array(
        [
            lam * targ.n[0, 0] * div[0] + shear[0] * targ.n[1, 0] + 2.0 * mu * grad[0, 0] * targ.n[0, 0],
            lam * targ.n[1, 0] * div[0] + shear[0] * targ.n[0, 0] + 2.0 * mu * grad[3, 0] * targ.n[1, 0],
        ]
    )

    np.testing.assert_allclose(traction, expected)
