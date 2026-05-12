import numpy as np
from scipy import sparse

from chunkie import (
    ChunkerFLAMMatrix,
    PointInfo,
    chunkerfunc,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    kernel,
)
from chunkie.chnk import flam


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


def test_flam_kernbyindex_matches_dense_and_sparse_overwrites():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    dense = chunkermat(chnkr, smooth_kernel)
    rows = np.array([0, 3, 7, 12], dtype=np.int64)
    cols = np.array([1, 2, 7], dtype=np.int64)

    actual = flam.kernbyindex(rows, cols, chnkr, smooth_kernel, (1, 1))
    np.testing.assert_allclose(actual, dense[np.ix_(rows, cols)], atol=1e-14)

    overwrite = sparse.csr_matrix((np.array([9.0, -4.0]), (np.array([3, 7]), np.array([2, 7]))), shape=dense.shape)
    overwritten = flam.kernbyindex(rows, cols, chnkr, smooth_kernel, (1, 1), overwrite)
    assert overwritten[1, 1] == 9.0
    assert overwritten[2, 2] == -4.0


def test_flam_proxy_square_geometry_and_proxyfun_shapes():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    pr, ptau, pw, pin = flam.proxy_square_pts(64)

    assert pr.shape == (2, 64)
    assert ptau.shape == (2, 64)
    np.testing.assert_allclose(np.sum(pw), 12.0, atol=1e-14)
    np.testing.assert_array_equal(pin(np.array([[0.0, 2.0], [0.0, 0.0]])), [True, False])

    slf = np.arange(5, dtype=np.int64)
    nbr = np.arange(chnkr.npt, dtype=np.int64)
    Kpxy, nbr_out = flam.proxyfun(slf, nbr, np.array([1.0, 1.0]), np.zeros(2), chnkr, smooth_kernel, (1, 1), pr, ptau, pw, pin)

    assert Kpxy.shape == (128, slf.size)
    np.testing.assert_array_equal(nbr_out, nbr)


def test_chunkermat_flam_applies_solves_and_logdet_against_dense():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    lap_s = kernel("lap", "s")
    dense = chunkermat(chnkr, lap_s) + np.eye(chnkr.npt)
    rhs = np.sin(np.arange(chnkr.npt))
    opts = {"acceleration": "flam", "dval": 1.0, "occ": 8, "rank_or_tol": 1e-10, "useproxy": False}

    flam_mat = chunkermat(chnkr, lap_s, opts)

    assert isinstance(flam_mat, ChunkerFLAMMatrix)
    np.testing.assert_allclose(flam_mat @ rhs, dense @ rhs, rtol=1e-10, atol=1e-11)
    sol = flam_mat.solve(rhs)
    np.testing.assert_allclose(dense @ sol, rhs, rtol=1e-10, atol=1e-11)
    sign, logabs = np.linalg.slogdet(dense)
    np.testing.assert_allclose(flam_mat.logdet(), np.log(np.asarray(sign, dtype=complex)) + logabs, rtol=1e-10, atol=1e-10)


def test_chunkerkerneval_flam_matches_eval_matrix_and_dense():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    targets = np.array([[0.0, 1.4, -0.25], [0.0, 0.2, 1.3]])
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))
    opts = {"acceleration": "flam", "occ": 8, "rank_or_tol": 1e-10, "useproxy": False}

    dense_mat = chunkerkernevalmat(chnkr, smooth_kernel, targets)
    flam_mat = chunkerkernevalmat(chnkr, smooth_kernel, targets, opts)
    flam_vals = chunkerkerneval(chnkr, smooth_kernel, dens, targets, opts).reshape(-1, order="F")

    np.testing.assert_allclose(flam_mat, dense_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(flam_vals, dense_mat @ dens, rtol=1e-10, atol=1e-11)


def test_chunkerinterior_flam_matches_direct_classification():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 6})
    pts = np.array([[0.0, 1.25, 0.999999, 1.000001], [0.0, 0.0, 0.0, 0.0]])

    direct = chunkerinterior(chnkr, pts)
    actual = chunkerinterior(chnkr, pts, {"acceleration": "flam", "occ": 8, "rank_or_tol": 1e-8, "useproxy": False})

    np.testing.assert_array_equal(actual, direct)
