import importlib

import numpy as np

from chunkie import Kernel, chunkerfunc, chunkerkerneval, kernel
from chunkie.geometry import PointInfo


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_laplace_kernel_wrapper_evaluates_directly():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    kern = kernel("laplace", "d")

    vals = chunkerkerneval(chnkr, kern, np.ones(chnkr.npt), np.array([[0.0], [0.0]]))

    assert kern.opdims == (1, 1)
    assert kern.sing == "smooth"
    np.testing.assert_allclose(vals.ravel(), [-1.0], atol=1e-12)


def test_kernel_add_scale_zero_and_nan_behaviors():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = np.array([[0.0], [0.0]])

    zero = kernel("zero")
    nan_k = kernel("nan")
    combined = 2.0 * kernel("lap", "d") + zero

    np.testing.assert_allclose(chunkerkerneval(chnkr, zero, np.ones(chnkr.npt), target), 0.0)
    np.testing.assert_allclose(chunkerkerneval(chnkr, combined, np.ones(chnkr.npt), target), -2.0, atol=1e-12)
    assert np.isnan(chunkerkerneval(chnkr, nan_k, np.ones(chnkr.npt), target)).all()
    assert Kernel.zeros().iszero
    assert Kernel.nans().isnan


def test_kernel_subtract_negate_divide_and_conjugate():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = np.array([[0.0], [0.0]])
    dens = np.ones(chnkr.npt)

    lapd = kernel("lap", "d")
    np.testing.assert_allclose(chunkerkerneval(chnkr, lapd - lapd, dens, target), 0.0, atol=1e-12)
    np.testing.assert_allclose(chunkerkerneval(chnkr, -lapd, dens, target), 1.0, atol=1e-12)
    np.testing.assert_allclose(chunkerkerneval(chnkr, lapd / 2.0, dens, target), -0.5, atol=1e-12)

    custom = kernel(lambda src, targ: (1.0 + 2.0j) * np.ones((targ.r.shape[1], src.r.shape[1])))
    vals = chunkerkerneval(chnkr, custom.conj(), dens, target)
    np.testing.assert_allclose(vals, (1.0 - 2.0j) * np.sum(chnkr.wts))


def test_kernel_fmm_fallback_matches_direct_layer_evaluation():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = np.array([[0.25, -0.4], [0.1, 0.3]])
    dens = np.cos(chnkr.r.reshape(2, -1, order="F")[0])
    kern = kernel("lap", "s")

    direct = chunkerkerneval(chnkr, kern, dens, target)
    via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm")

    assert kern.fmm is not None
    np.testing.assert_allclose(via_fmm, direct)


def test_fmm2dpy_laplace_gradient_and_helmholtz_layers_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]])
    dens = np.cos(chnkr.r.reshape(2, -1, order="F")[0])

    for kern in (
        kernel("lap", "sgrad"),
        kernel("lap", "d"),
        kernel("lap", "dgrad"),
        kernel("helm", "s", 1.3 + 0.2j),
        kernel("helm", "d", 1.3 + 0.2j),
        kernel("helm", "sgrad", 1.3 + 0.2j),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_fmm2dpy_laplace_derived_selectors_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = PointInfo(
        r=np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]]),
        n=np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]]),
    )
    dens = np.cos(chnkr.r.reshape(2, -1, order="F")[0])

    for kern in (
        kernel("lap", "sp"),
        kernel("lap", "stau"),
        kernel("lap", "hilb"),
        kernel("lap", "dp"),
        kernel("lap", "cp", [0.4, -0.7]),
        kernel("lap", "cg", [0.4, -0.7]),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_fmm2dpy_helmholtz_derived_selectors_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = PointInfo(
        r=np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]]),
        n=np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]]),
        d=np.array([[0.0, 1.0, -0.8], [1.0, 0.0, 0.6]]),
    )
    dens = np.cos(chnkr.r.reshape(2, -1, order="F")[0])

    for kern in (
        kernel("helm", "sp", 1.3 + 0.2j),
        kernel("helm", "stau", 1.3 + 0.2j),
        kernel("helm", "dp", 1.3 + 0.2j),
        kernel("helm", "cp", 1.3 + 0.2j, [0.4, -0.7]),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_helmholtz_factory_transmission_selectors_match_direct_kernel():
    src = PointInfo(
        r=np.array([[0.1, -0.8], [-0.3, 0.55]]),
        n=np.array([[0.6, -0.2], [0.8, 0.98]]),
    )
    targ = PointInfo(
        r=np.array([[-0.45, 0.7, 1.2], [1.1, -0.65, 0.25]]),
        n=np.array([[0.1, -0.7, 0.3], [0.99, 0.71, -0.95]]),
    )
    zk = 1.25 + 0.35j
    coefs = np.array([0.8 - 0.1j, -0.45 + 0.3j])
    all_coefs = np.array([[1.1, -0.4], [0.25, 0.7]])

    cases = {
        "cgrad": coefs,
        "c2trans": coefs,
        "all": all_coefs,
        "trans_rep": coefs,
        "trans_rep_prime": coefs,
        "trans_rep_grad": coefs,
    }
    for selector, selector_coefs in cases.items():
        kern = kernel("helm", selector, zk, selector_coefs)
        actual = kern(src, targ)
        expected = importlib.import_module("chunkie.kernels.helmholtz").kern(zk, src, targ, selector, selector_coefs)
        np.testing.assert_allclose(actual, expected)


def test_helmholtz_double_gradient_fmm_requests_dipole_gradients(monkeypatch):
    kernel_mod = importlib.import_module("chunkie.kernels.factory")

    class FakeFmm2d:
        def __init__(self):
            self.kwargs = None

        def hfmm2d(self, **kwargs):
            self.kwargs = kwargs

            class Output:
                pottarg = np.array([10.0 + 1.0j, 20.0 + 2.0j])
                gradtarg = np.array([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]])

            return Output()

    fake = FakeFmm2d()
    monkeypatch.setattr(kernel_mod, "_fmm2dpy", fake)

    kern = kernel_mod.helm2d_kernel("dgrad", 1.2 + 0.3j)
    src = PointInfo(
        r=np.array([[0.0, 1.0], [0.0, 0.0]]),
        n=np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    targ = PointInfo(r=np.array([[0.25, -0.4], [0.1, 0.3]]))
    sigma = np.array([0.5, -0.25])

    vals = kern.fmm(1e-11, src, targ, sigma)

    assert fake.kwargs["pgt"] == 2
    assert "dipstr" in fake.kwargs
    assert "dipvec" in fake.kwargs
    assert "charges" not in fake.kwargs
    np.testing.assert_allclose(fake.kwargs["dipvec"], src.n)
    np.testing.assert_allclose(
        vals,
        np.asarray([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]]).reshape(-1, order="F"),
    )


def test_biharmonic_laplacian_fmm_reuses_laplace_single_layer(monkeypatch):
    kernel_mod = importlib.import_module("chunkie.kernels.factory")

    class FakeFmm2d:
        def __init__(self):
            self.kwargs = None

        def lfmm2d(self, **kwargs):
            self.kwargs = kwargs

            class Output:
                pottarg = np.array([6.0, 8.0])
                gradtarg = np.zeros((2, 2))

            return Output()

    fake = FakeFmm2d()
    monkeypatch.setattr(kernel_mod, "_fmm2dpy", fake)

    kern = kernel_mod.biharm2d_kernel("lap")
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]))
    targ = PointInfo(r=np.array([[0.25, -0.4], [0.1, 0.3]]))
    sigma = np.array([0.5, -0.25])

    vals = kern.fmm(1e-11, src, targ, sigma)

    assert fake.kwargs["pgt"] == 2
    assert "charges" in fake.kwargs
    expected = (np.array([6.0, 8.0]) + np.sum(sigma)) / (2.0 * np.pi)
    np.testing.assert_allclose(vals, expected)


def test_fmm2dpy_biharmonic_selectors_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = PointInfo(
        r=np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]]),
        n=np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]]),
    )
    dens = np.cos(chnkr.r.reshape(2, -1, order="F")[0])

    for kern in (
        kernel("biharm", "s"),
        kernel("biharm", "sgrad"),
        kernel("biharm", "shess"),
        kernel("biharm", "lap"),
        kernel("biharm", "d"),
        kernel("biharm", "sp"),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_fmm2dpy_stokes_layers_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]])
    pts = chnkr.r.reshape(2, -1, order="F")
    dens = np.vstack((np.cos(pts[0]), np.sin(pts[1]))).reshape(-1, order="F")

    for kern in (
        kernel("stok", "s", 1.7),
        kernel("stok", "d", 1.7),
        kernel("stok", "spres", 1.7),
        kernel("stok", "dpres", 1.7),
        kernel("stok", "sgrad", 1.7),
        kernel("stok", "dgrad", 1.7),
        kernel("stok", "c", 1.7, [0.4, -0.2]),
        kernel("stok", "cpres", 1.7, [0.4, -0.2]),
        kernel("stok", "cgrad", 1.7, [0.4, -0.2]),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_stokes_traction_fmm_reconstructs_stress_from_pressure_and_gradient(monkeypatch):
    kernel_mod = importlib.import_module("chunkie.kernels.factory")

    class FakeFmm2d:
        def __init__(self):
            self.calls = []

        def stfmm2d(self, **kwargs):
            self.calls.append(kwargs)

            class Output:
                pottarg = np.zeros((1, 2, 2))
                pretarg = np.array([[2.0, 3.0]])
                gradtarg = np.array([[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]])

            return Output()

    fake = FakeFmm2d()
    monkeypatch.setattr(kernel_mod, "_fmm2dpy", fake)

    mu = 1.7
    kern = kernel_mod.stok2d_kernel("strac", mu)
    src = PointInfo(r=np.array([[0.0, 1.0], [0.0, 0.0]]))
    targ = PointInfo(
        r=np.array([[0.25, -0.4], [0.1, 0.3]]),
        n=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    sigma = np.array([0.5, -0.25, 0.75, 1.25])

    vals = kern.fmm(1e-11, src, targ, sigma)

    assert [call["ifppregtarg"] for call in fake.calls] == [2, 3]
    assert all("stoklet" in call for call in fake.calls)
    expected = np.array([0.0, 5.0 / (2.0 * np.pi), 13.0 / (2.0 * np.pi), 13.0 / (2.0 * np.pi)])
    np.testing.assert_allclose(vals, expected)


def test_fmm2dpy_elasticity_selectors_match_direct():
    chnkr, _ = chunkerfunc(circle, min_chunks=5, order=8)
    target = PointInfo(
        r=np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]]),
        n=np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]]),
    )
    pts = chnkr.r.reshape(2, -1, order="F")
    dens = np.vstack((np.cos(pts[0]), np.sin(pts[1]))).reshape(-1, order="F")

    for kern in (
        kernel("elast", "s", 1.5, 2.1),
        kernel("elast", "sgrad", 1.5, 2.1),
        kernel("elast", "strac", 1.5, 2.1),
        kernel("elast", "d", 1.5, 2.1),
        kernel("elast", "dalt", 1.5, 2.1),
        kernel("elast", "daltgrad", 1.5, 2.1),
        kernel("elast", "dalttrac", 1.5, 2.1),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, acceleration="fmm", tol=1e-12)

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_kernel_fmm_fallback_tracks_kernel_algebra():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = np.array([[0.25], [0.1]])
    dens = np.ones(chnkr.npt)
    combined = 2.0 * kernel("lap", "s") - kernel("lap", "d")

    direct = chunkerkerneval(chnkr, combined, dens, target)
    via_fmm = chunkerkerneval(chnkr, combined, dens, target, acceleration="fmm")

    assert combined.fmm is not None
    np.testing.assert_allclose(via_fmm, direct)


def test_kernel_interleave_builds_mixed_block_systems():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = np.array([[0.25, -0.4], [0.1, 0.3]])
    dens = np.vstack((np.ones(chnkr.npt), 2.0 * np.ones(chnkr.npt))).reshape(-1, order="F")

    s = kernel("lap", "s")
    d = kernel("lap", "d")
    z = kernel("zero")
    mixed = kernel([[d, -s], [s, z]])

    assert mixed.opdims == (2, 2)
    mat = mixed.eval(chnkr, target)
    vals = chunkerkerneval(chnkr, mixed, dens, target)

    np.testing.assert_allclose(vals.reshape(-1, order="F"), mat @ (dens * np.repeat(chnkr.wts.reshape(-1, order="F"), 2)))
    np.testing.assert_allclose(chunkerkerneval(chnkr, mixed, dens, target, acceleration="fmm"), vals, atol=1e-14)


def test_interleaved_helmholtz_fmm_preserves_complex_output_for_real_density():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    target = PointInfo(
        r=np.array([[0.25, -0.4], [0.1, 0.3]]),
        n=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    zk = 1.2 + 0.3j
    mixed = kernel(
        [
            [kernel("helm", "d", zk), kernel("helm", "s", zk)],
            [kernel("lap", "d"), kernel("lap", "s")],
        ]
    )
    dens = np.vstack((np.ones(chnkr.npt), 0.5 * np.ones(chnkr.npt))).reshape(-1, order="F")

    direct = chunkerkerneval(chnkr, mixed, dens, target)
    via_fmm = chunkerkerneval(chnkr, mixed, dens, target, acceleration="fmm", tol=1e-12)

    assert np.iscomplexobj(via_fmm)
    np.testing.assert_allclose(via_fmm, direct, rtol=5e-9, atol=5e-10)
