import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, kernel


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_laplace_kernel_wrapper_evaluates_directly():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    kern = kernel("laplace", "d")

    vals = chunkerkerneval(chnkr, kern, np.ones(chnkr.npt), np.array([[0.0], [0.0]]))

    assert kern.opdims == (1, 1)
    assert kern.sing == "smooth"
    np.testing.assert_allclose(vals.ravel(), [-1.0], atol=1e-12)


def test_kernel_add_scale_zero_and_nan_behaviors():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    target = np.array([[0.0], [0.0]])

    zero = kernel("zero")
    nan_k = kernel("nan")
    combined = 2.0 * kernel("lap", "d") + zero

    np.testing.assert_allclose(chunkerkerneval(chnkr, zero, np.ones(chnkr.npt), target), 0.0)
    np.testing.assert_allclose(chunkerkerneval(chnkr, combined, np.ones(chnkr.npt), target), -2.0, atol=1e-12)
    assert np.isnan(chunkerkerneval(chnkr, nan_k, np.ones(chnkr.npt), target)).all()


def test_kernel_subtract_negate_divide_and_conjugate():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    target = np.array([[0.0], [0.0]])
    dens = np.ones(chnkr.npt)

    lapd = kernel("lap", "d")
    np.testing.assert_allclose(chunkerkerneval(chnkr, lapd - lapd, dens, target), 0.0, atol=1e-12)
    np.testing.assert_allclose(chunkerkerneval(chnkr, -lapd, dens, target), 1.0, atol=1e-12)
    np.testing.assert_allclose(chunkerkerneval(chnkr, lapd / 2.0, dens, target), -0.5, atol=1e-12)

    custom = kernel(lambda src, targ: (1.0 + 2.0j) * np.ones((targ.r.shape[1], src.r.shape[1])))
    vals = chunkerkerneval(chnkr, custom.conj(), dens, target)
    np.testing.assert_allclose(vals, (1.0 - 2.0j) * np.sum(chnkr.wts))
