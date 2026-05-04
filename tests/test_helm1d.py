import numpy as np

from chunkie import kernel
from chunkie.chnk import helm1d


def test_helm1d_green_gradient_matches_finite_difference():
    src = np.array([[0.2], [-0.1]])
    targ = np.array([[1.1], [0.7]])
    zk = 1.4 + 0.2j
    eps = 1e-6

    val, grad, hess = helm1d.green(zk, src, targ)
    val_xp = helm1d.green(zk, src, targ + np.array([[eps], [0.0]]))[0]
    val_xm = helm1d.green(zk, src, targ - np.array([[eps], [0.0]]))[0]

    np.testing.assert_allclose(grad[:, :, 0], (val_xp - val_xm) / (2 * eps), rtol=1e-6, atol=1e-7)
    assert val.shape == (1, 1)
    assert hess.shape == (1, 1, 3)


def test_helm1d_kernel_selectors_and_kernel_wrapper():
    src = {
        "r": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "n": np.array([[0.0, 0.0], [-1.0, -1.0]]),
        "d": np.array([[1.0, 1.0], [0.0, 0.0]]),
    }
    targ = {
        "r": np.array([[0.25, 0.75], [0.5, 0.5]]),
        "n": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "d": np.array([[1.0, 1.0], [0.0, 0.0]]),
    }
    zk = 1.2

    assert helm1d.kern(zk, src, targ, "s").shape == (2, 2)
    assert helm1d.kern(zk, src, targ, "d").shape == (2, 2)
    assert helm1d.kern(zk, src, targ, "dp").shape == (2, 2)
    assert helm1d.kern(zk, src, targ, "c2trans").shape == (4, 2)
    assert helm1d.kern(zk, src, targ, "all", np.eye(2)).shape == (4, 4)
    assert kernel("helm1d", "s", zk).sing == "removable"


def test_helm1d_sweep_matches_direct_causal_sums():
    ts = np.linspace(-1.0, 1.0, 6)
    wts = np.linspace(0.5, 1.0, 6)
    inds = np.array([1, 2, 3])
    uin = np.array([2.0, -1.0, 0.5])
    zk = 0.7 + 0.1j

    got = helm1d.sweep(uin, inds, ts, wts, zk)
    u = np.zeros(ts.size)
    u[inds] = uin
    charges = u * wts
    expected = np.zeros(ts.size, dtype=complex)
    for i, ti in enumerate(ts):
        expected[i] = np.sum(np.exp(1j * zk * np.abs(ti - ts)) * charges)

    np.testing.assert_allclose(got, expected)
