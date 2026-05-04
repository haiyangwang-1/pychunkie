import numpy as np

from chunkie import (
    PointInfo,
    chunkerfunc,
    chunkerinterior,
    chunkerintegral,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    chunkerpoly,
    kernel,
    pointinfo,
)
from chunkie.chnk import quadnative


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def smooth_kernel(src: PointInfo, targ: PointInfo):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    return 1.0 + dx**2 + 0.5 * dy**2


def test_chunkermat_matches_chunkerkerneval_on_boundary_for_smooth_kernel():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))

    mat_vals = chunkermatapply(chnkr, smooth_kernel, dens)
    eval_vals = chunkerkerneval(chnkr, smooth_kernel, dens, chnkr).reshape(-1)

    np.testing.assert_allclose(mat_vals, eval_vals)


def test_pointinfo_uses_matlab_chunk_contiguous_ordering():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 3}, {"k": 5})
    info = pointinfo(chnkr)

    np.testing.assert_allclose(info.r[:, : chnkr.k], chnkr.r[:, :, 0])
    np.testing.assert_allclose(info.r[:, chnkr.k : 2 * chnkr.k], chnkr.r[:, :, 1])


def test_chunkerkernevalmat_matches_direct_target_evaluation():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    dens = np.sin(chnkr.r[1].reshape(-1, order="F"))
    targets = np.array([[0.0, 2.0], [0.0, -0.25]])

    mat = chunkerkernevalmat(chnkr, smooth_kernel, targets)
    vals = chunkerkerneval(chnkr, smooth_kernel, dens, targets).reshape(-1)

    np.testing.assert_allclose(mat @ dens, vals, atol=1e-14)


def test_chunkermat_accepts_kernel_objects():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    zero = kernel("zero")
    mat = chunkermat(chnkr, zero)

    assert mat.shape == (chnkr.npt, chnkr.npt)
    np.testing.assert_allclose(mat, 0.0)


def test_quadnative_buildmat_matches_dense_chunkermat():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 3}, {"k": 8})

    native = quadnative.buildmat(chnkr, smooth_kernel, (1, 1))
    dense = chunkermat(chnkr, smooth_kernel)

    np.testing.assert_allclose(native, dense)


def test_chunkerintegral_accepts_values_and_callables():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 16})

    np.testing.assert_allclose(chunkerintegral(chnkr, np.ones(chnkr.npt)), 2.0 * np.pi, atol=1e-13)
    np.testing.assert_allclose(
        chunkerintegral(chnkr, lambda r: r[0] ** 2),
        np.pi,
        atol=1e-13,
    )


def test_chunkerinterior_classifies_points_and_grids():
    square = chunkerpoly(
        np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        {"ifclosed": True},
        {"k": 8},
    )

    pts = np.array([[0.5, 1.5, 0.25], [0.5, 0.5, 1.25]])
    np.testing.assert_array_equal(chunkerinterior(square, pts), [True, False, False])

    grid = chunkerinterior(square, (np.array([-0.5, 0.5, 1.5]), np.array([0.5])))
    np.testing.assert_array_equal(grid, [[False, True, False]])
