import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, pointinfo
from chunkie.chnk import helm2d, lap2d


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

    r2 = np.array([[4.0, 5.0]])
    np.testing.assert_allclose(val, -np.log(r2) / (4.0 * np.pi))
    assert grad.shape == (1, 2, 2)
    assert hess.shape == (1, 2, 3)


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

    assert lap2d.kern(info, target, "s").shape == (2, chnkr.npt)
    assert lap2d.kern(info, target, "d").shape == (2, chnkr.npt)
    assert lap2d.kern(info, target, "sgrad").shape == (4, chnkr.npt)


def test_helmholtz_green_gradient_matches_finite_difference():
    src = np.array([[0.3], [-0.2]])
    targ = np.array([[1.1], [0.7]])
    zk = 1.2 + 0.4j
    eps = 1e-6

    val, grad, hess = helm2d.green(zk, src, targ)
    val_xp = helm2d.green(zk, src, targ + np.array([[eps], [0.0]]))[0]
    val_xm = helm2d.green(zk, src, targ - np.array([[eps], [0.0]]))[0]
    gx_fd = (val_xp - val_xm) / (2 * eps)

    np.testing.assert_allclose(grad[:, :, 0], gx_fd, rtol=1e-6, atol=1e-7)
    assert val.shape == (1, 1)
    assert hess.shape == (1, 1, 3)
