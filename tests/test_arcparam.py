import numpy as np

from chunkie import chunkerfunc, merge
from chunkie.chnk import arcparam


def circle(t, radius=1.0, center=(0.0, 0.0)):
    t = np.asarray(t)
    return (
        np.vstack((center[0] + radius * np.cos(t), center[1] + radius * np.sin(t))),
        np.vstack((-radius * np.sin(t), radius * np.cos(t))),
        np.vstack((-radius * np.cos(t), -radius * np.sin(t))),
    )


def test_arcparam_evaluates_original_chunk_nodes():
    a, _ = chunkerfunc(lambda t: circle(t, 1.0), {"nchmin": 4}, {"k": 16})
    b, _ = chunkerfunc(lambda t: circle(t, 0.5, (3.0, 0.0)), {"nchmin": 4}, {"k": 16})
    chnkr = merge([a, b])

    pdata = arcparam.init(chnkr)
    s = np.concatenate((a.arclengthfun().reshape(-1, order="F"), b.arclengthfun().reshape(-1, order="F") + np.sum(a.wts)))
    r, d, d2 = arcparam.eval(s, pdata)

    np.testing.assert_allclose(r, chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F"), atol=1e-11)
    np.testing.assert_allclose(np.sqrt(np.sum(d**2, axis=0)), 1.0, atol=1e-11)
    np.testing.assert_allclose(np.sum(d * d2, axis=0), 0.0, atol=1e-10)


def test_arcparam_derivatives_are_consistent_on_circle():
    chnkr, _ = chunkerfunc(lambda t: circle(t, 2.0), {"nchmin": 4}, {"k": 18})
    pdata = arcparam.init(chnkr)
    s = np.linspace(0.1, np.sum(chnkr.wts) - 0.1, 25)
    r, d, d2 = arcparam.eval(s, pdata)

    np.testing.assert_allclose(np.sum(r * d, axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.sum(d**2, axis=0)), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.sum(d * d2, axis=0), 0.0, atol=1e-10)


def test_arcresample_makes_panel_speed_constant():
    chnkr, _ = chunkerfunc(lambda t: circle(t, 1.5), {"nchmin": 5}, {"k": 16})

    resampled, eps = chnkr.arcresample()

    assert eps >= 0.0
    np.testing.assert_allclose(resampled.area(), chnkr.area(), atol=1e-10)
    np.testing.assert_allclose(np.sum(resampled.wts), np.sum(chnkr.wts), atol=1e-10)
    speed = resampled.arclengthdens()
    expected = np.ones((resampled.k, 1)) @ (resampled.chunklen()[None, :] / 2.0)
    np.testing.assert_allclose(speed, expected, atol=1e-10)
