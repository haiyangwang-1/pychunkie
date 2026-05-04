import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, kernel
from chunkie.chnk import biharm2d


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


def test_biharmonic_kernel_selectors_and_factory_shapes():
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]), n=np.array([[1.0, 0.0], [0.0, 1.0]]))
    targ = PointInfo(r=np.array([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]]), n=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))

    assert biharm2d.kern(src, targ, "s").shape == (3, 2)
    assert biharm2d.kern(src, targ, "d").shape == (3, 2)
    assert biharm2d.kern(src, targ, "sp").shape == (3, 2)
    assert biharm2d.kern(src, targ, "sgrad").shape == (6, 2)
    assert biharm2d.kern(src, targ, "shess").shape == (9, 2)
    assert kernel("biharm", "sgrad").opdims == (2, 1)


def test_biharmonic_layer_evaluation_uses_special_quadrature():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    kern = kernel("biharmonic", "s")
    vals = chunkerkerneval(chnkr, kern, np.ones(chnkr.npt), np.array([[0.25], [0.1]]))

    assert vals.shape == (1, 1)
    assert np.isfinite(vals).all()
