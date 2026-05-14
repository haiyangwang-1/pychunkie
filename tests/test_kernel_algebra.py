import numpy as np

from chunkie import PointInfo, kernel


def test_zero_scaled_kernel_evaluates_as_zero_even_at_singular_points():
    pt = PointInfo(r=np.array([[0.0], [0.0]]))

    singular_zero = 0.0 * kernel("lap", "s")
    nan_zero = 0.0 * kernel("nan")

    assert singular_zero.iszero
    assert not singular_zero.isnan
    assert nan_zero.iszero
    assert not nan_zero.isnan
    np.testing.assert_allclose(singular_zero(pt, pt), [[0.0]])
    np.testing.assert_allclose(nan_zero(pt, pt), [[0.0]])
