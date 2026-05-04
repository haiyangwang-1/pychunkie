import numpy as np

from chunkie.chnk import spcl


def test_absconvgauss_derivatives_match_finite_differences():
    x = np.linspace(-0.3, 0.3, 9)
    a = -1.0
    b = -0.75
    h = 0.125
    eps = 1e-6

    val, der, der2 = spcl.absconvgauss(x, a, b, h)
    vp = spcl.absconvgauss(x + eps, a, b, h)[0]
    vm = spcl.absconvgauss(x - eps, a, b, h)[0]
    dp = spcl.absconvgauss(x + eps, a, b, h)[1]
    dm = spcl.absconvgauss(x - eps, a, b, h)[1]

    np.testing.assert_allclose((vp - vm) / (2 * eps), der, atol=1e-9)
    np.testing.assert_allclose((dp - dm) / (2 * eps), der2, atol=1e-8)
    assert val.shape == x.shape
