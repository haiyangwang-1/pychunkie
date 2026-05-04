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
    assert elast2d.kern(lam, mu, src, targ, "d").shape == (6, 4)
    assert elast2d.kern(lam, mu, src, targ, "dalt").shape == (6, 4)
    assert kernel("elast", "s", lam, mu).opdims == (2, 2)


def test_elasticity_single_layer_is_symmetric_in_components():
    src = PointInfo(r=np.array([[0.0], [0.0]]))
    targ = PointInfo(r=np.array([[1.0], [2.0]]))
    mat = elast2d.kern(1.5, 2.1, src, targ, "s")

    np.testing.assert_allclose(mat[0, 1], mat[1, 0])
