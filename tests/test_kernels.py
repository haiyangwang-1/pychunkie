import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, chunkermat, kernel
from chunkie.kernels import helmholtz as helm2d
from chunkie.kernels import laplace as lap2d

pointinfo = PointInfo.from_any


def circle(t, radius=1.0):
    t = np.asarray(t)
    return (
        np.vstack((radius * np.cos(t), radius * np.sin(t))),
        np.vstack((-radius * np.sin(t), radius * np.cos(t))),
        np.vstack((-radius * np.cos(t), -radius * np.sin(t))),
    )


def test_laplace_green_matches_direct_formula():
    src = np.array([[0.0, 1.0], [0.0, 0.0]])
    targ = np.array([[0.0], [2.0]])

    val, grad, hess = lap2d.green(src, targ)

    rx = targ[0, :, None] - src[0, None, :]
    ry = targ[1, :, None] - src[1, None, :]
    r2 = rx**2 + ry**2
    r4 = r2**2
    expected_grad = np.stack((-rx / (2.0 * np.pi * r2), -ry / (2.0 * np.pi * r2)), axis=2)
    expected_hess = np.stack(
        (
            rx**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2),
            rx * ry / (np.pi * r4),
            ry**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2),
        ),
        axis=2,
    )
    np.testing.assert_allclose(val, -np.log(r2) / (4.0 * np.pi))
    np.testing.assert_allclose(grad, expected_grad)
    np.testing.assert_allclose(hess, expected_hess)


def test_laplace_direct_layer_evaluation_on_circle():
    radius = 2.0
    chnkr, _ = chunkerfunc(lambda t: circle(t, radius), {"nchmin": 8}, {"k": 16})
    target = np.array([[0.0], [0.0]])

    single = chunkerkerneval(chnkr, lambda s, t: lap2d.kern(s, t, "s"), np.ones(chnkr.npt), target)
    double = chunkerkerneval(chnkr, lambda s, t: lap2d.kern(s, t, "d"), np.ones(chnkr.npt), target)

    np.testing.assert_allclose(single.ravel(), [-radius * np.log(radius)], atol=1e-12)
    np.testing.assert_allclose(double.ravel(), [-1.0], atol=1e-12)


def test_laplace_kernel_selectors_have_expected_shapes():
    chnkr, _ = chunkerfunc(lambda t: circle(t, 1.0), {"nchmin": 4}, {"k": 8})
    info = pointinfo(chnkr)
    target = {"r": np.array([[0.25, 0.5], [0.1, -0.2]]), "n": np.array([[1.0, 0.0], [0.0, 1.0]])}

    targ = pointinfo(target)
    val, grad, hess = lap2d.green(info.r, targ.r)
    expected_d = -(grad[:, :, 0] * info.n[0, None, :] + grad[:, :, 1] * info.n[1, None, :])
    expected_sp = grad[:, :, 0] * targ.n[0, :, None] + grad[:, :, 1] * targ.n[1, :, None]
    expected_dp = -(
        hess[:, :, 0] * info.n[0, None, :] * targ.n[0, :, None]
        + hess[:, :, 1] * (info.n[1, None, :] * targ.n[0, :, None] + info.n[0, None, :] * targ.n[1, :, None])
        + hess[:, :, 2] * info.n[1, None, :] * targ.n[1, :, None]
    )
    expected_sgrad = grad.transpose(0, 2, 1).reshape(2 * targ.r.shape[1], info.r.shape[1])

    np.testing.assert_allclose(lap2d.kern(info, target, "s"), val)
    np.testing.assert_allclose(lap2d.kern(info, target, "d"), expected_d)
    np.testing.assert_allclose(lap2d.kern(info, target, "c"), expected_d + val)
    np.testing.assert_allclose(lap2d.kern(info, target, "cp"), expected_dp + expected_sp)
    np.testing.assert_allclose(lap2d.kern(info, target, "sgrad"), expected_sgrad)


def test_helmholtz_green_gradient_matches_finite_difference():
    src = np.array([[0.3], [-0.2]])
    targ = np.array([[1.1], [0.7]])
    zk = 1.2 + 0.4j
    eps = 1e-6

    val, grad, hess = helm2d.green(zk, src, targ)
    val_xp = helm2d.green(zk, src, targ + np.array([[eps], [0.0]]))[0]
    val_xm = helm2d.green(zk, src, targ - np.array([[eps], [0.0]]))[0]
    val_yp = helm2d.green(zk, src, targ + np.array([[0.0], [eps]]))[0]
    val_ym = helm2d.green(zk, src, targ - np.array([[0.0], [eps]]))[0]
    gx_fd = (val_xp - val_xm) / (2 * eps)
    gy_fd = (val_yp - val_ym) / (2 * eps)
    grad_xp = helm2d.green(zk, src, targ + np.array([[eps], [0.0]]))[1]
    grad_xm = helm2d.green(zk, src, targ - np.array([[eps], [0.0]]))[1]
    grad_yp = helm2d.green(zk, src, targ + np.array([[0.0], [eps]]))[1]
    grad_ym = helm2d.green(zk, src, targ - np.array([[0.0], [eps]]))[1]
    hess_fd = np.empty_like(hess)
    hess_fd[:, :, 0] = (grad_xp[:, :, 0] - grad_xm[:, :, 0]) / (2 * eps)
    hess_fd[:, :, 1] = (grad_yp[:, :, 0] - grad_ym[:, :, 0]) / (2 * eps)
    hess_fd[:, :, 2] = (grad_yp[:, :, 1] - grad_ym[:, :, 1]) / (2 * eps)

    np.testing.assert_allclose(grad[:, :, 0], gx_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad[:, :, 1], gy_fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess, hess_fd, rtol=5e-5, atol=5e-6)


def test_helmdiffgreen_fills_same_point_value_and_gradient_limits():
    pts = np.array([[0.0, 0.5], [0.0, -0.25]])
    zk = 1.3 + 0.2j

    val, grad, _ = helm2d.helmdiffgreen(zk, pts, pts)
    expected_limit = 0.25j - (np.log(zk / 2.0) + helm2d._EULER_GAMMA) / (2.0 * np.pi)
    diag = np.diag_indices(pts.shape[1])

    np.testing.assert_allclose(val[diag], expected_limit)
    np.testing.assert_allclose(grad[diag[0], diag[1], :], 0.0)
    assert np.isfinite(val).all()
    assert np.isfinite(grad).all()


def test_helmdiff_single_layer_smooth_diagonal_is_finite():
    chnkr, _ = chunkerfunc(lambda t: circle(t, 1.0), min_chunks=4, order=8)
    kern = kernel("helmdiff", "s", [1.3, 2.1])

    smooth = chunkermat(chnkr, kern, quadrature="smooth")

    assert np.isfinite(smooth).all()


def test_helmholtz_kernel_selectors_have_expected_shapes():
    chnkr, _ = chunkerfunc(lambda t: circle(t, 1.0), {"nchmin": 4}, {"k": 8})
    info = pointinfo(chnkr)
    target = {
        "r": np.array([[0.25, 0.5], [0.1, -0.2]]),
        "n": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "d": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }

    targ = pointinfo(target)
    val, grad, hess = helm2d.green(1.3, info.r, targ.r)
    expected_d = -(grad[:, :, 0] * info.n[0, None, :] + grad[:, :, 1] * info.n[1, None, :])
    expected_dp = -(
        hess[:, :, 0] * info.n[0, None, :] * targ.n[0, :, None]
        + hess[:, :, 1] * (info.n[1, None, :] * targ.n[0, :, None] + info.n[0, None, :] * targ.n[1, :, None])
        + hess[:, :, 2] * info.n[1, None, :] * targ.n[1, :, None]
    )

    np.testing.assert_allclose(helm2d.kern(1.3, info, target, "s"), val)
    np.testing.assert_allclose(helm2d.kern(1.3, info, target, "d"), expected_d)
    np.testing.assert_allclose(helm2d.kern(1.3, info, target, "dp"), expected_dp)
    np.testing.assert_allclose(helm2d.kern(1.3, info, target, "c"), expected_d + 1.0j * val)
