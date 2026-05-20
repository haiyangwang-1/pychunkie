import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, kernel
from chunkie.chnk import bhfmm2d, cfmm2d


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_cfmm2d_direct_selectors_match_fmm2dpy():
    import fmm2dpy

    src = PointInfo(r=np.array([[0.1, -0.4, 0.6], [0.2, 0.3, -0.5]]))
    targ = PointInfo(r=np.array([[0.7, -0.2], [-0.1, 0.9]]))
    charges = np.array([1.0 + 2.0j, -0.3 + 0.5j, 0.8 - 0.2j])
    dipoles = np.array([0.7 - 0.2j, -1.1 + 0.4j, 0.25 + 0.6j])
    strengths = np.vstack((charges, dipoles)).reshape(-1, order="F")

    out = fmm2dpy.cfmm2d(
        eps=1.0e-13,
        sources=src.r,
        charges=charges,
        dipstr=dipoles,
        targets=targ.r,
        pgt=3,
    )

    np.testing.assert_allclose(cfmm2d.kern(src, targ, "charge") @ charges, out.pottarg - cfmm2d.kern(src, targ, "dipole") @ dipoles)
    np.testing.assert_allclose(cfmm2d.kern(src, targ, "potential") @ strengths, out.pottarg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cfmm2d.kern(src, targ, "derivative") @ strengths, out.gradtarg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cfmm2d.kern(src, targ, "second_derivative") @ strengths, out.hesstarg, rtol=1e-12, atol=1e-12)
    all_vals = (cfmm2d.kern(src, targ, "all") @ strengths).reshape(3, targ.r.shape[1], order="F")
    np.testing.assert_allclose(all_vals, np.vstack((out.pottarg, out.gradtarg, out.hesstarg)), rtol=1e-12, atol=1e-12)


def test_bhfmm2d_direct_potential_uses_current_fmm2d_definition():
    src = PointInfo(r=np.array([[0.1, -0.4, 0.6], [0.2, 0.3, -0.5]]))
    targ = PointInfo(r=np.array([[0.7, -0.2], [-0.1, 0.9]]))
    strengths = np.array(
        [
            [1.0 + 2.0j, -0.3 + 0.5j, 0.8 - 0.2j],
            [0.7 - 0.4j, 0.2 + 0.9j, -0.1 + 0.3j],
            [0.1 - 0.2j, 0.3 + 0.4j, -0.5 + 0.2j],
            [0.5 + 0.1j, -0.6 + 0.2j, 0.7 - 0.1j],
            [-0.7 + 0.3j, 0.8 - 0.1j, 0.4 + 0.5j],
        ]
    )
    zsrc = src.r[0] + 1j * src.r[1]
    ztarg = targ.r[0] + 1j * targ.r[1]
    diff = ztarg[:, None] - zsrc[None, :]
    cdiff = np.conjugate(diff)
    expected = np.sum(
        2.0 * strengths[0][None, :] * np.log(np.abs(diff))
        + strengths[1][None, :] * diff / cdiff
        + strengths[2][None, :] / diff
        + strengths[4][None, :] / cdiff
        + strengths[3][None, :] * diff / cdiff**2,
        axis=1,
    )

    actual = bhfmm2d.kern(src, targ, "potential") @ strengths.reshape(-1, order="F")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_bhfmm2d_direct_selectors_match_fmm2dpy():
    import fmm2dpy

    src = PointInfo(r=np.array([[0.1, -0.4, 0.6], [0.2, 0.3, -0.5]]))
    targ = PointInfo(r=np.array([[0.7, -0.2], [-0.1, 0.9]]))
    strengths = np.array(
        [
            [1.0 + 2.0j, -0.3 + 0.5j, 0.8 - 0.2j],
            [0.7 - 0.4j, 0.2 + 0.9j, -0.1 + 0.3j],
            [0.1 - 0.2j, 0.3 + 0.4j, -0.5 + 0.2j],
            [0.5 + 0.1j, -0.6 + 0.2j, 0.7 - 0.1j],
            [-0.7 + 0.3j, 0.8 - 0.1j, 0.4 + 0.5j],
        ]
    )
    flat = strengths.reshape(-1, order="F")
    out = fmm2dpy.bhfmm2d(
        eps=1.0e-13,
        sources=src.r,
        charges=strengths[:2],
        dipoles=strengths[2:],
        targets=targ.r,
        pgt=2,
    )

    np.testing.assert_allclose(bhfmm2d.kern(src, targ, "potential") @ flat, out.pottarg, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        (bhfmm2d.kern(src, targ, "gradient") @ flat).reshape(3, targ.r.shape[1], order="F"),
        out.gradtarg,
        rtol=1e-12,
        atol=1e-12,
    )
    all_vals = (bhfmm2d.kern(src, targ, "all") @ flat).reshape(4, targ.r.shape[1], order="F")
    np.testing.assert_allclose(all_vals, np.vstack((out.pottarg, out.gradtarg)), rtol=1e-12, atol=1e-12)


def test_raw_fmm2d_kernel_factories_match_dense_target_evaluation():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 5}, {"k": 8})
    target = np.array([[0.25, -0.4, 1.5], [0.1, 0.3, -0.2]])
    pts = chnkr.r.reshape(2, -1, order="F")
    cfmm_dens = np.vstack((np.cos(pts[0]) + 0.2j, np.sin(pts[1]) - 0.1j)).reshape(-1, order="F")
    bhfmm_dens = np.vstack(
        (
            np.cos(pts[0]) + 0.2j,
            np.sin(pts[1]) - 0.1j,
            0.3 * np.cos(pts[1]) + 0.4j,
            -0.2 * np.sin(pts[0]) + 0.1j,
            0.5 * np.ones(chnkr.npt) - 0.3j,
        )
    ).reshape(-1, order="F")

    assert kernel("cfmm2d", "charge").opdims == (1, 1)
    assert kernel("cfmm2d", "all").opdims == (3, 2)
    assert kernel("bhfmm2d", "all").opdims == (4, 5)

    for kern, dens in (
        (kernel("cfmm2d", "potential"), cfmm_dens),
        (kernel("cfmm2d", "derivative"), cfmm_dens),
        (kernel("cfmm2d", "second_derivative"), cfmm_dens),
        (kernel("cfmm2d", "all"), cfmm_dens),
        (kernel("bhfmm2d", "potential"), bhfmm_dens),
        (kernel("bhfmm2d", "gradient"), bhfmm_dens),
        (kernel("bhfmm2d", "all"), bhfmm_dens),
    ):
        direct = chunkerkerneval(chnkr, kern, dens, target)
        via_fmm = chunkerkerneval(chnkr, kern, dens, target, {"acceleration": "fmm", "eps": 1.0e-12})

        assert kern.fmm is not None
        np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)
