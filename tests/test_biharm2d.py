import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, kernel
from chunkie.kernels import biharmonic as biharm2d


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_biharmonic_green_gradient_matches_finite_difference():
    src = np.array([[0.3], [-0.2]])
    targ = np.array([[1.1], [0.7]])
    eps = 1.0e-6

    val, grad, hess, lap = biharm2d.green(src, targ)
    val_xp = biharm2d.green(src, targ + np.array([[eps], [0.0]]))[0]
    val_xm = biharm2d.green(src, targ - np.array([[eps], [0.0]]))[0]
    gx_fd = (val_xp - val_xm) / (2.0 * eps)

    np.testing.assert_allclose(grad[:, :, 0], gx_fd, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(hess[:, :, 0] + hess[:, :, 2], lap)


def test_biharmonic_green_coincident_limits_preserve_singular_second_derivatives():
    pts = np.array([[0.0, 0.5], [0.0, -0.25]])

    val, grad, hess, lap = biharm2d.green(pts, pts)
    diag = np.diag_indices(pts.shape[1])

    np.testing.assert_allclose(val[diag], 0.0)
    np.testing.assert_allclose(grad[diag[0], diag[1], :], 0.0)
    assert not np.isfinite(hess[diag[0], diag[1], :]).all()
    assert not np.isfinite(lap[diag]).all()


def test_biharmonic_kernel_selectors_and_factory_shapes():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]), n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))

    val, grad, hess, _ = biharm2d.green(src.r, targ.r)
    expected_d = -(grad[:, :, 0] * src.n[0, None, :] + grad[:, :, 1] * src.n[1, None, :])
    expected_sp = grad[:, :, 0] * targ.n[0, :, None] + grad[:, :, 1] * targ.n[1, :, None]
    expected_sgrad = grad.transpose(0, 2, 1).reshape(2 * targ.r.shape[1], src.r.shape[1])
    expected_shess = hess.transpose(0, 2, 1).reshape(3 * targ.r.shape[1], src.r.shape[1])

    np.testing.assert_allclose(biharm2d.kern(src, targ, "s"), val)
    np.testing.assert_allclose(biharm2d.kern(src, targ, "d"), expected_d)
    np.testing.assert_allclose(biharm2d.kern(src, targ, "sp"), expected_sp)
    np.testing.assert_allclose(biharm2d.kern(src, targ, "sgrad"), expected_sgrad)
    np.testing.assert_allclose(biharm2d.kern(src, targ, "shess"), expected_shess)
    assert kernel("biharm", "sgrad").opdims == (2, 1)


def test_biharmonic_layer_evaluation_uses_special_quadrature():
    chnkr, _ = chunkerfunc(circle, min_chunks=6, order=8)
    kern = kernel("biharmonic", "s")
    target = np.array([[0.25], [0.1]])
    vals = chunkerkerneval(chnkr, kern, np.ones(chnkr.npt), target)

    assert vals.shape == (1, 1)
    radius_sq = float(target[0, 0] ** 2 + target[1, 0] ** 2)
    np.testing.assert_allclose(vals.ravel(), [radius_sq / 4.0], atol=5e-12)
