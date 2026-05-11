import numpy as np
from scipy import sparse

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, kernel, merge
from chunkie.chnk import quadadap, quadggq


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_matlab_log_quadrature_tables_load_for_each_legendre_node():
    aux = quadggq.setup(8)
    xleg = np.polynomial.legendre.leggauss(8)[0]

    xs1, wts1, xs0, wts0 = quadggq.getlogquad(8, 2)
    assert xs1.shape == wts1.shape
    assert len(xs0) == len(wts0) == 8
    assert 16 in quadggq.logavail()
    assert 17 not in quadggq.logavail()
    np.testing.assert_allclose(xs1[0], -0.9999983834562877, atol=1e-15)
    np.testing.assert_allclose(wts1[0], 4.264322824107065e-06, atol=1e-18)
    np.testing.assert_allclose(xs0[0][0], 0.9764569073834163, atol=1e-15)
    np.testing.assert_allclose(wts0[0][0], 0.06013321084659288, atol=1e-15)

    assert len(aux.xs0) == 8
    assert aux.ainterp1.shape[1] == 8
    for node, xs, wts, interp in zip(xleg, aux.xs0, aux.wts0, aux.ainterps0):
        assert xs.shape == wts.shape
        assert interp.shape == (xs.size, 8)
        assert np.all(xs >= -1.0)
        assert np.all(xs <= 1.0)
        assert not np.any(np.isclose(xs, node))
        np.testing.assert_allclose(np.sum(wts), 2.0, atol=1e-14)


def test_quadggq_buildmat_removes_laplace_single_layer_diagonal_infinities():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    lap_s = kernel("lap", "s")

    smooth = chunkermat(chnkr, lap_s, {"forcesmooth": True})
    special = quadggq.buildmat(chnkr, lap_s, lap_s.opdims)

    assert np.isinf(np.diag(smooth)).all()
    assert np.isfinite(special).all()
    np.testing.assert_allclose(special @ np.ones(chnkr.npt), 0.0, atol=5e-5)


def test_chunkermat_uses_special_quadrature_for_log_kernels_by_default():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    lap_s = kernel("lap", "s")
    dens = np.ones(chnkr.npt)

    mat_vals = chunkermat(chnkr, lap_s) @ dens
    eval_vals = chunkerkerneval(chnkr, lap_s, dens, chnkr).reshape(-1, order="F")

    assert np.isfinite(mat_vals).all()
    np.testing.assert_allclose(mat_vals, eval_vals)
    np.testing.assert_allclose(mat_vals, 0.0, atol=5e-5)


def test_quadggq_handles_complex_helmholtz_single_layer_blocks():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    helm_s = kernel("helm", "s", 1.3 + 0.2j)

    mat = quadggq.buildmat(chnkr, helm_s, helm_s.opdims)

    assert np.iscomplexobj(mat)
    assert np.isfinite(mat).all()


def test_nearbuildmat_matches_buildmat_neighbor_block_and_correction():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    lap_s = kernel("lap", "s")
    aux = quadggq.setup(chnkr.k, "log")
    src_chunk = 0
    targ_chunk = int(chnkr.adj[1, src_chunk]) - 1

    full = quadggq.buildmat(chnkr, lap_s, lap_s.opdims, type="log", auxquads=aux)
    near = quadggq.nearbuildmat(chnkr, targ_chunk, src_chunk, lap_s, lap_s.opdims, aux)
    rows = slice(targ_chunk * chnkr.k, (targ_chunk + 1) * chnkr.k)
    cols = slice(src_chunk * chnkr.k, (src_chunk + 1) * chnkr.k)

    np.testing.assert_allclose(near, full[rows, cols])

    corrected = quadggq.nearbuildmat(
        chnkr,
        targ_chunk,
        src_chunk,
        lap_s,
        lap_s.opdims,
        aux,
        corrections=True,
    )
    native = lap_s(
        {"r": chnkr.r[:, :, src_chunk], "d": chnkr.d[:, :, src_chunk], "d2": chnkr.d2[:, :, src_chunk], "n": chnkr.n[:, :, src_chunk]},
        {"r": chnkr.r[:, :, targ_chunk], "d": chnkr.d[:, :, targ_chunk], "d2": chnkr.d2[:, :, targ_chunk], "n": chnkr.n[:, :, targ_chunk]},
    ) * chnkr.wts[:, src_chunk][None, :]
    np.testing.assert_allclose(corrected, near - native)


def test_buildmat_ilist_skips_bad_neighbor_and_self_special_blocks():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    lap_s = kernel("lap", "s")

    special = quadggq.buildmat(chnkr, lap_s, lap_s.opdims)
    skipped = quadggq.buildmat(chnkr, lap_s, lap_s.opdims, ilist=np.array([0, 1]))
    smooth = chunkermat(chnkr, lap_s, {"forcesmooth": True})

    block = lambda mat, i, j: mat[i * chnkr.k : (i + 1) * chnkr.k, j * chnkr.k : (j + 1) * chnkr.k]
    np.testing.assert_allclose(block(skipped, 1, 0), block(smooth, 1, 0))
    np.testing.assert_allclose(block(skipped, 0, 0), block(smooth, 0, 0))
    np.testing.assert_allclose(block(skipped, 2, 1), block(special, 2, 1))


def test_buildmattd_returns_sparse_special_blocks_only():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    lap_s = kernel("lap", "s")

    td = quadggq.buildmattd(chnkr, lap_s, lap_s.opdims)
    full = quadggq.buildmat(chnkr, lap_s, lap_s.opdims)

    assert sparse.issparse(td)
    dense_td = td.toarray()

    block = lambda mat, i, j: mat[i * chnkr.k : (i + 1) * chnkr.k, j * chnkr.k : (j + 1) * chnkr.k]
    np.testing.assert_allclose(block(dense_td, 0, 0), block(full, 0, 0))
    np.testing.assert_allclose(block(dense_td, 1, 0), block(full, 1, 0))
    np.testing.assert_allclose(block(dense_td, 3, 0), 0.0)

    skipped = quadggq.buildmattd(chnkr, lap_s, lap_s.opdims, ilist=np.array([0, 1])).toarray()
    np.testing.assert_allclose(block(skipped, 0, 0), 0.0)
    np.testing.assert_allclose(block(skipped, 1, 0), 0.0)
    np.testing.assert_allclose(block(skipped, 2, 1), block(full, 2, 1))


def test_pv_and_hs_ggq_tables_are_available_for_matlab_orders():
    assert 8 in quadggq.hqsuppavail()
    pv_xs, pv_ws = quadggq.getpvquad(8)
    hs_xs, hs_ws = quadggq.gethsquad(8)

    assert len(pv_xs) == 8
    assert len(hs_xs) == 8
    assert pv_xs[0].shape == pv_ws[0].shape
    assert hs_xs[0].shape == hs_ws[0].shape
    np.testing.assert_allclose(pv_xs[0][0], -0.9965414829599555, atol=1e-15)
    np.testing.assert_allclose(hs_xs[0][0], -0.9965754957842378, atol=1e-15)


def test_setup_accepts_pv_and_hs_singularities():
    pv = quadggq.setup(8, "pv")
    hs = quadggq.setup(8, "hs")

    assert pv.type == "pv"
    assert hs.type == "hs"
    assert len(pv.xs0) == 8
    assert len(hs.xs0) == 8


def test_chunkermat_uses_special_quadrature_for_pv_and_hs_kernels():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    lap_sgrad = kernel("lap", "sgrad")
    lap_dgrad = kernel("lap", "dgrad")

    pv_mat = chunkermat(chnkr, lap_sgrad)
    hs_mat = chunkermat(chnkr, lap_dgrad)

    assert pv_mat.shape == (2 * chnkr.npt, chnkr.npt)
    assert hs_mat.shape == (2 * chnkr.npt, chnkr.npt)
    assert np.isfinite(pv_mat).all()
    assert np.isfinite(hs_mat).all()


def test_quadadap_buildmat_uses_adaptive_neighbor_blocks(monkeypatch):
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    lap_s = kernel("lap", "s")
    calls = []
    original = quadadap.adapgausswts

    def wrapped(*args, **kwargs):
        calls.append((args[1], args[2].r.shape[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(quadadap, "adapgausswts", wrapped)

    adap = quadadap.buildmat(chnkr, lap_s, opts={"sing": "log"})
    ggq = quadggq.buildmat(chnkr, lap_s, lap_s.opdims, type="log")

    assert len(calls) == 2 * chnkr.nch
    assert {ntarg for _, ntarg in calls} == {chnkr.k}
    np.testing.assert_allclose(adap, ggq)


def test_quadadap_robust_mode_repairs_non_neighbor_close_blocks(monkeypatch):
    left, _ = chunkerfunc(lambda t: circle(t), {"nchmin": 8}, {"k": 8})

    def shifted(t):
        r, d, d2 = circle(t)
        r = r + np.array([[2.05], [0.0]])
        return r, d, d2

    right, _ = chunkerfunc(shifted, {"nchmin": 8}, {"k": 8})
    chnkr = merge([left, right])
    lap_s = kernel("lap", "s")
    calls = []
    original = quadadap.adapgausswts

    def wrapped(*args, **kwargs):
        calls.append((args[1], args[2].r.shape[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(quadadap, "adapgausswts", wrapped)

    robust = quadadap.buildmat(chnkr, lap_s, opts={"sing": "log", "robust": True})

    assert np.isfinite(robust).all()
    assert any(ntarg != chnkr.k for _, ntarg in calls)
