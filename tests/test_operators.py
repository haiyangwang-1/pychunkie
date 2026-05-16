import numpy as np
import pytest

import chunkie.operators as operators_mod
from _performance import record_backend_metrics, timed_call
from chunkie import (
    ChunkerFMMMatrix,
    PointInfo,
    chunkerfunc,
    chunkerintegral,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    chunkerpoly,
    kernel,
)
from chunkie.operators import _evaluation as operators_evaluation
from chunkie.quadrature import native as quadnative

pointinfo = PointInfo.from_any


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


def probe_fragile_kernel(src: PointInfo, targ: PointInfo):
    if src.r.shape[1] == 1 and targ.r.shape[1] == 1:
        raise ValueError("single-point dtype probe is not supported")
    return smooth_kernel(src, targ)


probe_fragile_kernel.opdims = (1, 1)


def test_chunkermat_matches_chunkerkerneval_on_boundary_for_smooth_kernel():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))
    info = pointinfo(chnkr)
    weights = chnkr.wts.reshape(-1, order="F")
    expected = smooth_kernel(info, info) @ (dens * weights)

    mat_vals = chunkermatapply(chnkr, smooth_kernel, dens)
    eval_vals = chunkerkerneval(chnkr, smooth_kernel, dens, chnkr).reshape(-1)

    np.testing.assert_allclose(mat_vals, expected)
    np.testing.assert_allclose(mat_vals, eval_vals)


def test_chunkermat_allows_dtype_probe_fallback_for_custom_kernel():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)

    actual = chunkermat(chnkr, probe_fragile_kernel)
    expected = (
        smooth_kernel(pointinfo(chnkr), pointinfo(chnkr))
        * chnkr.wts.reshape(-1, order="F")[None, :]
    )

    np.testing.assert_allclose(actual, expected)


def test_chunkermat_does_not_swallow_real_custom_kernel_errors():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)

    def broken_kernel(src: PointInfo, targ: PointInfo):
        raise RuntimeError("custom kernel failed")

    broken_kernel.opdims = (1, 1)
    with pytest.raises(RuntimeError, match="custom kernel failed"):
        chunkermat(chnkr, broken_kernel)


def test_chunkermatapply_fmm_matches_special_matrix_application(test_metrics):
    chnkr, _ = chunkerfunc(circle, min_chunks=6, order=8)
    lap_s = kernel("lap", "s")
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))

    direct, direct_elapsed = timed_call(lambda: chunkermat(chnkr, lap_s) @ dens)
    via_fmm, fmm_elapsed = timed_call(
        lambda: chunkermatapply(chnkr, lap_s, dens, acceleration="fmm", tol=1e-12)
    )

    np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)
    record_backend_metrics(
        test_metrics,
        "chunkermatapply_fmm",
        backend="fmm",
        problem_size={"source_nodes": chnkr.npt, "target_nodes": chnkr.npt, "rhs_columns": 1},
        elapsed_s=fmm_elapsed,
        reference_elapsed_s=direct_elapsed,
        actual=via_fmm,
        expected=direct,
        abs_tol=1e-10,
        rel_tol=1e-9,
    )


def test_chunkerkerneval_same_source_fmm_uses_smooth_fmm_plus_correction(monkeypatch):
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    lap_s = kernel("lap", "s")
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))
    calls = 0
    original_fmm = lap_s.fmm

    def counting_fmm(eps, srcinfo, targinfo, sigma):
        nonlocal calls
        calls += 1
        return original_fmm(eps, srcinfo, targinfo, sigma)

    monkeypatch.setattr(lap_s, "fmm", counting_fmm)

    direct = chunkermat(chnkr, lap_s) @ dens
    via_fmm = chunkerkerneval(chnkr, lap_s, dens, chnkr, acceleration="fmm", tol=1e-12).reshape(
        -1, order="F"
    )

    assert calls == 1
    np.testing.assert_allclose(via_fmm, direct, rtol=1e-9, atol=1e-10)


def test_chunkermat_fmm_returns_matrix_free_operator_matching_dense_application(test_metrics):
    chnkr, _ = chunkerfunc(circle, min_chunks=6, order=8)
    lap_s = kernel("lap", "s")
    x = chnkr.r[0].reshape(-1, order="F")
    y = chnkr.r[1].reshape(-1, order="F")
    dens = np.cos(x)
    rhs = np.column_stack((dens, np.sin(y)))

    dense, dense_elapsed = timed_call(lambda: chunkermat(chnkr, lap_s))
    via_fmm, fmm_setup_elapsed = timed_call(
        lambda: chunkermat(chnkr, lap_s, acceleration="fmm", tol=1e-12)
    )
    fmm_applied, fmm_apply_elapsed = timed_call(lambda: via_fmm @ rhs)
    dense_applied, dense_apply_elapsed = timed_call(lambda: dense @ rhs)

    assert isinstance(via_fmm, ChunkerFMMMatrix)
    assert via_fmm.shape == dense.shape
    np.testing.assert_allclose(via_fmm @ dens, dense @ dens, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fmm_applied, dense_applied, rtol=1e-9, atol=1e-10)
    test_metrics.record("chunkermat_fmm_setup_elapsed_s", fmm_setup_elapsed)
    test_metrics.record("chunkermat_fmm_dense_build_elapsed_s", dense_elapsed)
    record_backend_metrics(
        test_metrics,
        "chunkermat_fmm_matmat",
        backend="fmm",
        problem_size={
            "source_nodes": chnkr.npt,
            "target_nodes": chnkr.npt,
            "rhs_columns": rhs.shape[1],
        },
        elapsed_s=fmm_apply_elapsed,
        reference_elapsed_s=dense_apply_elapsed,
        actual=fmm_applied,
        expected=dense_applied,
        abs_tol=1e-10,
        rel_tol=1e-9,
    )


def test_fmm_request_warns_when_kernel_uses_direct_fallback():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    custom = kernel(smooth_kernel)
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))

    with pytest.warns(RuntimeWarning, match="direct dense matrix-vector fallback"):
        via_fmm = chunkermatapply(chnkr, custom, dens, acceleration="fmm")
    dense = chunkermat(chnkr, custom) @ dens

    np.testing.assert_allclose(via_fmm, dense, rtol=1e-13, atol=1e-13)


def test_block_chunkermat_fmm_matches_dense_application_and_l2scale():
    first, _ = chunkerfunc(circle, min_chunks=4, order=8)
    second = first.translate(np.array([2.8, 0.15]))
    chunkers = [first, second]
    lap_s = kernel("lap", "s")
    zero = kernel("zero")
    blocks = [[zero, lap_s], [-lap_s, zero]]
    rhs = np.sin(0.17 * np.arange(first.npt + second.npt))
    rhs2 = np.column_stack((rhs, np.cos(0.11 * np.arange(rhs.size))))

    dense = chunkermat(chunkers, blocks)
    via_fmm = chunkermat(chunkers, blocks, acceleration="fmm", tol=1e-12)
    applied = chunkermatapply(chunkers, blocks, rhs2, acceleration="fmm", tol=1e-12)
    dense_l2 = chunkermat(chunkers, blocks, l2scale=True)
    via_fmm_l2 = chunkermat(chunkers, blocks, acceleration="fmm", tol=1e-12, l2scale=True)

    assert isinstance(via_fmm, ChunkerFMMMatrix)
    assert via_fmm.shape == dense.shape
    np.testing.assert_allclose(via_fmm @ rhs, dense @ rhs, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(via_fmm @ rhs2, dense @ rhs2, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(applied, dense @ rhs2, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(via_fmm_l2 @ rhs, dense_l2 @ rhs, rtol=1e-9, atol=1e-10)


def test_block_chunkermat_fmm_applies_self_special_corrections_once():
    first, _ = chunkerfunc(circle, min_chunks=3, order=6)
    second = first.translate(np.array([2.8, 0.15]))
    chunkers = [first, second]
    lap_s = kernel("lap", "s")
    blocks = [[lap_s, lap_s], [lap_s, lap_s]]
    rhs = np.sin(0.17 * np.arange(first.npt + second.npt))
    rhs2 = np.column_stack((rhs, np.cos(0.11 * np.arange(rhs.size))))

    dense = chunkermat(chunkers, blocks)
    via_fmm = chunkermat(chunkers, blocks, acceleration="fmm", tol=1e-12)

    np.testing.assert_allclose(via_fmm @ rhs, dense @ rhs, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(via_fmm @ rhs2, dense @ rhs2, rtol=1e-9, atol=1e-10)


def test_block_chunkermat_reuses_special_quadrature_for_self_blocks():
    first, _ = chunkerfunc(circle, min_chunks=3, order=6)
    second = first.translate(np.array([2.8, 0.15]))
    lap_s = kernel("lap", "s")
    dense = chunkermat([first, second], [[lap_s, lap_s], [lap_s, lap_s]])

    assert np.isfinite(dense).all()
    nfirst = first.npt
    np.testing.assert_allclose(dense[:nfirst, :nfirst], chunkermat(first, lap_s), atol=1e-13)
    np.testing.assert_allclose(dense[nfirst:, nfirst:], chunkermat(second, lap_s), atol=1e-13)


def test_pointinfo_uses_matlab_chunk_contiguous_ordering():
    chnkr, _ = chunkerfunc(circle, min_chunks=3, order=5)
    info = pointinfo(chnkr)

    np.testing.assert_allclose(info.r, chnkr.r.reshape(chnkr.dim, -1, order="F"))
    np.testing.assert_allclose(info.d, chnkr.d.reshape(chnkr.dim, -1, order="F"))
    np.testing.assert_allclose(info.d2, chnkr.d2.reshape(chnkr.dim, -1, order="F"))
    np.testing.assert_allclose(info.n, chnkr.n.reshape(chnkr.dim, -1, order="F"))


def test_chunkerkernevalmat_matches_direct_target_evaluation():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    dens = np.sin(chnkr.r[1].reshape(-1, order="F"))
    targets = np.array([[0.0, 2.0], [0.0, -0.25]])
    weights = chnkr.wts.reshape(-1, order="F")
    expected_mat = smooth_kernel(pointinfo(chnkr), PointInfo(r=targets)) * weights[None, :]

    mat = chunkerkernevalmat(chnkr, smooth_kernel, targets)
    vals = chunkerkerneval(chnkr, smooth_kernel, dens, targets).reshape(-1)

    np.testing.assert_allclose(mat, expected_mat)
    np.testing.assert_allclose(mat @ dens, vals, atol=1e-14)


def test_chunkerkernevalmat_fmm_materializes_target_eval_matrix():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    lap_s = kernel("lap", "s")
    targets = np.array([[0.0, 1.6, -1.35], [0.0, -0.4, 0.8]])
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))

    dense_mat = chunkerkernevalmat(chnkr, lap_s, targets)
    fmm_mat = chunkerkernevalmat(chnkr, lap_s, targets, acceleration="fmm", tol=1e-12)
    fmm_vals = chunkerkerneval(chnkr, lap_s, dens, targets, acceleration="fmm", tol=1e-12).reshape(
        -1, order="F"
    )

    np.testing.assert_allclose(fmm_mat, dense_mat, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fmm_mat @ dens, fmm_vals, rtol=1e-10, atol=1e-11)


def test_chunkerkernevalmat_fmm_materializes_same_source_special_matrix():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    lap_s = kernel("lap", "s")
    dens = np.cos(chnkr.r[0].reshape(-1, order="F"))

    dense_mat = chunkermat(chnkr, lap_s)
    fmm_mat = chunkerkernevalmat(chnkr, lap_s, chnkr, acceleration="fmm", tol=1e-12)

    np.testing.assert_allclose(fmm_mat @ dens, dense_mat @ dens, rtol=1e-9, atol=1e-10)


def test_chunkermat_accepts_kernel_objects():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    zero = kernel("zero")
    mat = chunkermat(chnkr, zero)

    assert mat.shape == (chnkr.npt, chnkr.npt)
    np.testing.assert_allclose(mat, 0.0)


def test_quadnative_buildmat_matches_dense_chunkermat():
    chnkr, _ = chunkerfunc(circle, min_chunks=3, order=8)
    weights = chnkr.wts.reshape(-1, order="F")
    expected = smooth_kernel(pointinfo(chnkr), pointinfo(chnkr)) * weights[None, :]

    native = quadnative.buildmat(chnkr, smooth_kernel, (1, 1))
    dense = chunkermat(chnkr, smooth_kernel)

    np.testing.assert_allclose(native, expected)
    np.testing.assert_allclose(native, dense)


def test_chunkerintegral_accepts_values_and_callables():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=16)

    np.testing.assert_allclose(chunkerintegral(chnkr, np.ones(chnkr.npt)), 2.0 * np.pi, atol=1e-13)
    np.testing.assert_allclose(
        chunkerintegral(chnkr, lambda r: r[0] ** 2),
        np.pi,
        atol=1e-13,
    )


def test_chunkerinterior_classifies_points_and_grids():
    square = chunkerpoly(
        np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        closed=True,
        order=8,
    )

    pts = np.array([[0.5, 1.5, 0.25], [0.5, 0.5, 1.25]])
    np.testing.assert_array_equal(chunkerinterior(square, pts), [True, False, False])

    grid = chunkerinterior(square, (np.array([-0.5, 0.5, 1.5]), np.array([0.5])))
    np.testing.assert_array_equal(grid, [[False, True, False]])


def test_chunkerinterior_fmm_matches_direct_with_close_correction(monkeypatch):
    chnkr, _ = chunkerfunc(circle, min_chunks=8, order=8)
    pts = np.array([[0.0, 1.25, 0.999999, 1.000001], [0.0, 0.0, 0.0, 0.0]])
    expected = np.array([True, False, True, False])
    calls = []
    original = operators_evaluation.chunkerkerneval

    def wrapped(*args, **kwargs):
        calls.append(args[4] if len(args) > 4 else kwargs.get("options", kwargs.get("opts")))
        return original(*args, **kwargs)

    monkeypatch.setattr(operators_evaluation, "chunkerkerneval", wrapped)

    direct = operators_mod.chunkerinterior(chnkr, pts)
    via_fmm = operators_mod.chunkerinterior(chnkr, pts, acceleration="fmm", near_factor=0.25)

    assert calls and calls[0]["acceleration"] == "fmm"
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(via_fmm, direct)
