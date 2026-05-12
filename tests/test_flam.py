import numpy as np
import pytest
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


def vector_smooth_kernel(src: PointInfo, targ: PointInfo):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    base = 1.0 + dx**2 + 0.5 * dy**2
    ntarget, nsource = base.shape
    out = np.zeros((2 * ntarget, 2 * nsource))
    out[0::2, 0::2] = base
    out[1::2, 1::2] = 2.0 + 0.25 * base
    out[0::2, 1::2] = 0.1 * dx
    out[1::2, 0::2] = -0.2 * dy
    return out


vector_smooth_kernel.opdims = (2, 2)


def data_kernel(src: PointInfo, targ: PointInfo):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    return 1.0 + dx + 0.1 * src.data[0][None, :] + 0.2 * targ.data[0][:, None]


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


def test_flam_kernbyindexr_matches_dense_and_sparse_overwrites():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    targets = np.array([[0.0, 1.4, -0.25], [0.0, 0.2, 1.3]])
    dense = chunkerkernevalmat(chnkr, smooth_kernel, targets)
    rows = np.array([0, 1, 2], dtype=np.int64)
    cols = np.array([1, 3, 5], dtype=np.int64)

    actual = flam.kernbyindexr(rows, cols, targets, chnkr, smooth_kernel, (1, 1))
    np.testing.assert_allclose(actual, dense[np.ix_(rows, cols)], atol=1e-14)

    overwrite = sparse.csr_matrix((np.array([7.0, -3.0]), (np.array([1, 2]), np.array([3, 5]))), shape=dense.shape)
    overwritten = flam.kernbyindexr(rows, cols, targets, chnkr, smooth_kernel, (1, 1), overwrite)
    assert overwritten[1, 1] == 7.0
    assert overwritten[2, 2] == -3.0


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


def test_flam_proxyfunr_column_and_row_shapes():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    targets = np.array([[0.0, 1.4, -0.25], [0.0, 0.2, 1.3]])
    pr, ptau, pw, pin = flam.proxy_square_pts(64)
    rows = np.array([0, 1, 2], dtype=np.int64)
    cols = np.array([1, 3, 5], dtype=np.int64)
    cx = chnkr.r.reshape(2, -1, order="F")
    targinfo = PointInfo(r=targets)

    Kc, nbr_c = flam.proxyfunr("c", targets, cx, cols, rows, np.array([1.0, 1.0]), np.zeros(2), chnkr, smooth_kernel, (1, 1), pr, ptau, pw, pin, targobj=targinfo)
    Kr, nbr_r = flam.proxyfunr("r", targets, cx, rows, cols, np.array([1.0, 1.0]), np.zeros(2), chnkr, smooth_kernel, (1, 1), pr, ptau, pw, pin, targobj=targinfo)

    assert Kc.shape == (64, cols.size)
    assert Kr.shape == (rows.size, 64)
    np.testing.assert_array_equal(nbr_c, rows)
    np.testing.assert_array_equal(nbr_r, cols)


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


def test_chunkermat_flam_proxy_paths_match_dense_application():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    lap_s = kernel("lap", "s")
    dense = chunkermat(chnkr, lap_s) + np.eye(chnkr.npt)
    rhs = np.sin(np.arange(chnkr.npt))
    opts = {"acceleration": "flam", "dval": 1.0, "occ": 8, "rank_or_tol": 1e-8}

    default_proxy = chunkermat(chnkr, lap_s, opts)
    proxy_by_level = chunkermat(chnkr, lap_s, {**opts, "proxybylevel": True})
    rskel_proxy = chunkermat(chnkr, lap_s, {**opts, "flamtype": "rskel"})

    np.testing.assert_allclose(default_proxy @ rhs, dense @ rhs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(proxy_by_level @ rhs, dense @ rhs, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rskel_proxy @ rhs, dense @ rhs, rtol=1e-8, atol=1e-10)
    with pytest.raises(NotImplementedError, match="solve is only available for rskelf"):
        rskel_proxy.solve(rhs)


def test_chunkermat_flam_adds_dval_without_replacing_smooth_diagonal():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    dense_scalar = chunkermat(chnkr, smooth_kernel) + 0.5 * np.eye(chnkr.npt)
    rhs_scalar = np.cos(np.arange(chnkr.npt))
    flam_scalar = chunkermat(chnkr, smooth_kernel, {"acceleration": "flam", "dval": 0.5, "occ": 16, "rank_or_tol": 1e-10, "useproxy": False})

    np.testing.assert_allclose(flam_scalar @ rhs_scalar, dense_scalar @ rhs_scalar, rtol=1e-10, atol=1e-11)

    dense_complex = chunkermat(chnkr, smooth_kernel) + (0.5 + 0.2j) * np.eye(chnkr.npt)
    flam_complex = chunkermat(chnkr, smooth_kernel, {"acceleration": "flam", "dval": 0.5 + 0.2j, "occ": 16, "rank_or_tol": 1e-10, "useproxy": False})
    np.testing.assert_allclose(flam_complex @ rhs_scalar, dense_complex @ rhs_scalar, rtol=1e-10, atol=1e-11)

    dense_vector = chunkermat(chnkr, vector_smooth_kernel) + 0.25 * np.eye(2 * chnkr.npt)
    rhs_vector = np.sin(np.arange(2 * chnkr.npt))
    flam_vector = chunkermat(chnkr, vector_smooth_kernel, {"acceleration": "flam", "dval": 0.25, "occ": 16, "rank_or_tol": 1e-10, "useproxy": False})

    np.testing.assert_allclose(flam_vector @ rhs_vector, dense_vector @ rhs_vector, rtol=1e-10, atol=1e-11)


def test_chunkermat_flam_l2scale_matches_scaled_dense_matrix():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    weighted_dense = chunkermat(chnkr, smooth_kernel)
    weights = chnkr.wts.reshape(-1, order="F")
    scaled_dense = np.sqrt(weights)[:, None] * weighted_dense * (1.0 / np.sqrt(weights))[None, :] + 0.75 * np.eye(chnkr.npt)
    rhs = np.sin(np.arange(chnkr.npt))
    flam_mat = chunkermat(chnkr, smooth_kernel, {"acceleration": "flam", "dval": 0.75, "l2scale": True, "occ": 16, "rank_or_tol": 1e-10, "useproxy": False})

    np.testing.assert_allclose(flam_mat @ rhs, scaled_dense @ rhs, rtol=1e-10, atol=1e-11)

    lap_s = kernel("lap", "s")
    special_dense = chunkermat(chnkr, lap_s)
    scaled_special = np.sqrt(weights)[:, None] * special_dense * (1.0 / np.sqrt(weights))[None, :] + np.eye(chnkr.npt)
    flam_special = chunkermat(chnkr, lap_s, {"acceleration": "flam", "dval": 1.0, "l2scale": True, "occ": 8, "rank_or_tol": 1e-10, "useproxy": False})

    np.testing.assert_allclose(flam_special @ rhs, scaled_special @ rhs, rtol=1e-10, atol=1e-11)


def test_chunkermat_flam_preserves_point_data_without_proxy():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    chnkr.makedatarows(1)
    pts = chnkr.r.reshape(2, chnkr.npt, order="F")
    chnkr.data[0, :, :] = pts[0].reshape(chnkr.k, chnkr.nch, order="F")
    dense = chunkermat(chnkr, data_kernel) + 0.5 * np.eye(chnkr.npt)
    rhs = np.cos(np.arange(chnkr.npt))
    flam_mat = chunkermat(chnkr, data_kernel, {"acceleration": "flam", "dval": 0.5, "occ": 16, "rank_or_tol": 1e-10})

    np.testing.assert_allclose(flam_mat @ rhs, dense @ rhs, rtol=1e-10, atol=1e-11)


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

    pts = chnkr.r.reshape(2, chnkr.npt, order="F")
    dens_vec = np.vstack((np.cos(pts[0]), np.sin(pts[1]))).reshape(-1, order="F")
    dense_vec_mat = chunkerkernevalmat(chnkr, vector_smooth_kernel, targets)
    flam_vec_mat = chunkerkernevalmat(chnkr, vector_smooth_kernel, targets, opts)
    flam_vec_vals = chunkerkerneval(chnkr, vector_smooth_kernel, dens_vec, targets, opts).reshape(-1, order="F")

    np.testing.assert_allclose(flam_vec_mat, dense_vec_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(flam_vec_vals, dense_vec_mat @ dens_vec, rtol=1e-10, atol=1e-11)


def test_chunkerkerneval_flam_default_proxy_matches_dense():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    targets = np.array([[0.0, 1.4, -0.25, 0.7], [0.0, 0.2, 1.3, -1.2]])
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))
    opts = {"acceleration": "flam", "occ": 8, "rank_or_tol": 1e-10}

    dense_mat = chunkerkernevalmat(chnkr, smooth_kernel, targets)
    proxy_mat = chunkerkernevalmat(chnkr, smooth_kernel, targets, opts)
    proxy_vals = chunkerkerneval(chnkr, smooth_kernel, dens, targets, opts).reshape(-1, order="F")

    np.testing.assert_allclose(proxy_mat, dense_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(proxy_vals, dense_mat @ dens, rtol=1e-10, atol=1e-11)


def test_chunkerkerneval_flam_forceadap_matches_dense_adaptive_corrections():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 6})
    targets = np.array([[0.98, 1.4, -0.25], [0.0, 0.2, 1.3]])
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))
    lap_s = kernel("lap", "s")
    opts = {"forceadap": True}
    flam_opts = {"acceleration": "flam", "forceadap": True, "occ": 8, "rank_or_tol": 1e-10}

    dense_mat = chunkerkernevalmat(chnkr, lap_s, targets, opts)
    flam_mat = chunkerkernevalmat(chnkr, lap_s, targets, flam_opts)
    flam_vals = chunkerkerneval(chnkr, lap_s, dens, targets, flam_opts).reshape(-1, order="F")

    np.testing.assert_allclose(flam_mat, dense_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(flam_vals, dense_mat @ dens, rtol=1e-10, atol=1e-11)


def test_chunkerinterior_flam_matches_direct_classification():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 6})
    pts = np.array([[0.0, 1.25, 0.999999, 1.000001], [0.0, 0.0, 0.0, 0.0]])

    direct = chunkerinterior(chnkr, pts)
    actual = chunkerinterior(chnkr, pts, {"acceleration": "flam", "occ": 8, "rank_or_tol": 1e-8, "useproxy": False})

    np.testing.assert_array_equal(actual, direct)
