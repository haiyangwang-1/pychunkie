import numpy as np
from scipy.special import erf

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

    x2 = x / (np.sqrt(2.0) * h)
    expfac = np.exp(-(x * x) / (2.0 * h * h))
    expected_val = a * x * erf(x2) + np.sqrt(2.0 / np.pi) * a * h * expfac + b
    expected_der = a * erf(x2)
    expected_der2 = a * np.sqrt(2.0 / np.pi) / h * expfac

    np.testing.assert_allclose(val, expected_val, atol=0.0)
    np.testing.assert_allclose(der, expected_der, atol=0.0)
    np.testing.assert_allclose(der2, expected_der2, atol=0.0)
    np.testing.assert_allclose((vp - vm) / (2 * eps), der, atol=1e-9)
    np.testing.assert_allclose((dp - dm) / (2 * eps), der2, atol=1e-8)
