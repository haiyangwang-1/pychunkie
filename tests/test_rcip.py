import numpy as np
import pytest

from chunkie import chunkerkerneval, chunkermat, chunkgraph, kernel, lege
from chunkie.quadrature import panel as pquad
from chunkie.quadrature import rcip


def test_ipinit_interpolates_to_half_panels_and_preserves_weights():
    t, w, _, _ = lege.exps(8)
    ip, ipw = rcip.IPinit(t, w)
    t2 = np.concatenate((t - 1.0, t + 1.0)) / 2.0

    vals = t**5 - 0.2 * t**3 + 0.7
    expected = t2**5 - 0.2 * t2**3 + 0.7

    np.testing.assert_allclose(ip @ vals, expected, atol=1e-14)
    np.testing.assert_allclose(w @ vals, np.sum(ipw @ (w * vals)), atol=1e-14)


def test_setup_returns_zero_based_rcip_indices_and_block_shapes():
    out = rcip.setup(4, 2, 3, np.array([True, False, True]))
    pbc, pwbc, starL, circL, starS, circS, ilist, starL1, circL1 = out
    t, w, _, _ = lege.exps(4)
    ip, ipw = rcip.IPinit(t, w)
    expected_pbc = np.kron(np.eye(3), np.kron(ip, np.eye(2)))
    expected_pwbc = np.kron(np.eye(3), np.kron(ipw, np.eye(2)))

    np.testing.assert_allclose(pbc, expected_pbc)
    np.testing.assert_allclose(pwbc, expected_pwbc)
    assert starL.size == 48
    assert circL.size == 24
    assert starS.size == 24
    assert circS.size == 24
    assert starL.min() == 0
    assert circL.max() < 3 * 3 * 4 * 2
    np.testing.assert_array_equal(ilist[:, 0], [0, 1])
    np.testing.assert_array_equal(ilist[:, 1], [1, 2])
    assert starL1.size == starL.size // 2
    assert circL1.size == circL.size // 2


def test_schurbana_matches_direct_block_formula_shapes():
    rng = np.random.default_rng(1234)
    nbad = 4
    ngood = 2
    size = nbad + ngood
    p = rng.random((nbad, nbad // 2))
    pw = rng.random((nbad, nbad // 2))
    k = rng.random((size, size))
    k[np.ix_(range(nbad, size), range(nbad, size))] += 3.0 * np.eye(ngood)
    a = np.eye(nbad)

    out = rcip.SchurBana(
        p,
        pw,
        k,
        a,
        np.arange(nbad),
        np.arange(nbad, size),
        np.arange(ngood),
        np.arange(ngood, nbad),
    )

    assert out.shape == (nbad, nbad)
    expected = a.copy()
    va = k[np.ix_(np.arange(nbad, size), np.arange(nbad))] @ expected
    pta = pw.T @ expected
    ptau = pta @ k[np.ix_(np.arange(nbad), np.arange(nbad, size))]
    dvaui = np.linalg.inv(k[np.ix_(np.arange(nbad, size), np.arange(nbad, size))] - va @ k[np.ix_(np.arange(nbad), np.arange(nbad, size))])
    dvauivap = dvaui @ (va @ p)
    expected[np.ix_(np.arange(ngood), np.arange(ngood))] = pta @ p + ptau @ dvauivap
    expected[np.ix_(np.arange(ngood, nbad), np.arange(ngood, nbad))] = dvaui
    expected[np.ix_(np.arange(ngood, nbad), np.arange(ngood))] = -dvauivap
    expected[np.ix_(np.arange(ngood), np.arange(ngood, nbad))] = -ptau @ dvaui
    np.testing.assert_allclose(out, expected)


def test_rcompchunk_identity_baseline_and_corner_refine():
    verts = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    edges = np.array([[0, 1], [1, 2]])
    cg = chunkgraph(verts, edges, pref={"k": 6}, cparams={"nchmin": 1})

    refined = rcip.corner_refine(cg, vertices=[1], depth=2)
    assert refined.echnks[0].nch == cg.echnks[0].nch + 2
    assert refined.echnks[1].nch == cg.echnks[1].nch + 2

    rmat, saved = rcip.Rcompchunk(cg.echnks, np.array([0, 1]), lambda s, t: np.eye(1), 1, cg.verts[:, 1])

    assert rmat.shape == (2 * 2 * cg.k, 2 * 2 * cg.k)
    np.testing.assert_allclose(rmat, np.eye(rmat.shape[0]))
    assert saved.nedge == 2
    assert saved.R[0].shape == rmat.shape

    rho, srcinfo, wts = rcip.rhohatInterp(np.arange(rmat.shape[0]), saved)
    np.testing.assert_array_equal(rho[0], np.arange(rmat.shape[0]))
    assert srcinfo == [None]
    assert wts == [None]


def test_rcompchunk_runs_recursive_compression_for_corner_edges():
    verts = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    edges = np.array([[0, 1], [1, 2]])
    cg = chunkgraph(verts, edges, pref={"k": 4}, cparams={"nchmin": 2})

    rmat, saved = rcip.Rcompchunk(
        cg.echnks,
        np.array([0, 1]),
        kernel("lap", "d"),
        1,
        cg.verts[:, 1],
        opts={"nsub": 2, "rcip_savedepth": 2},
    )

    assert rmat.shape == (2 * 2 * cg.k, 2 * 2 * cg.k)
    assert np.isfinite(rmat).all()
    assert saved.nsub == 2
    assert len(saved.R) == 3
    assert len(saved.MAT) == 2
    assert len(saved.chnkrlocals) == 2
    assert np.linalg.norm(rmat - np.eye(rmat.shape[0])) > 1e-3
    np.testing.assert_allclose(np.trace(rmat), 16.300387693756427, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rmat, "fro"), 4.120788302796726, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        rmat[0, :6],
        [
            1.0033853368397785,
            0.0088738397763664,
            0.0078763442808689,
            0.0096558477593892,
            0.0019520338244678,
            0.0044367509083990,
        ],
        rtol=1e-12,
        atol=1e-12,
    )

    rho, srcinfo, wts = rcip.rhohatInterp(np.arange(rmat.shape[0], dtype=float), saved, 2)
    assert len(rho) == 2
    assert rho[0].size == 4 * cg.k
    assert srcinfo[0].r.shape == (2, 4 * cg.k)
    assert wts[0].shape == (4 * cg.k,)


def test_rcompchunk_rejects_nonfinite_local_kernel_blocks():
    verts = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    edges = np.array([[0, 1], [1, 2]])
    cg = chunkgraph(verts, edges, pref={"k": 4}, cparams={"nchmin": 2})

    def bad_kernel(src, targ):
        return np.full((targ.r.shape[1], src.r.shape[1]), np.nan)

    bad_kernel.opdims = (1, 1)

    with pytest.raises(ValueError, match=r"RCIP local matrix block .* non-finite"):
        rcip.Rcompchunk(
            cg.echnks,
            np.array([0, 1]),
            bad_kernel,
            1,
            cg.verts[:, 1],
            opts={"nsub": 1, "rcip_savedepth": 1},
        )


def test_chunkgraph_rcip_runs_selected_vertices_and_ignores_marked_vertices():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    cg = chunkgraph(verts, edges, pref={"k": 4}, cparams={"nchmin": 2})

    result = rcip.chunkgraph_rcip(
        cg,
        kernel("lap", "d"),
        1,
        opts={"nsub": 1, "rcip_savedepth": 1},
        ignore_vertices=[0],
    )

    np.testing.assert_array_equal(result.vertices, [1, 2, 3])
    assert len(result.R) == 3
    assert len(result.saved) == 3
    for rmat, saved, edge_indices in zip(result.R, result.saved, result.edge_indices):
        assert edge_indices.size == 2
        assert rmat.shape == (4 * cg.k, 4 * cg.k)
        assert saved.nedge == 2
        assert saved.nsub == 1
        assert np.isfinite(rmat).all()
    for vertex, rmat, edge_indices in zip(result.vertices, result.R, result.edge_indices):
        direct, direct_saved = rcip.Rcompchunk(
            cg.echnks,
            edge_indices,
            kernel("lap", "d"),
            1,
            cg.verts[:, vertex],
            opts={"nsub": 1, "rcip_savedepth": 1},
        )
        np.testing.assert_allclose(rmat, direct)
        np.testing.assert_allclose(result.saved[np.where(result.vertices == vertex)[0][0]].R[-1], direct_saved.R[-1])


def test_chunkgraph_rcip_subselects_global_block_kernels():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    cg = chunkgraph(verts, edges, pref={"k": 4}, cparams={"nchmin": 2})
    calls: list[tuple[int, int]] = []

    def zero_kernel(label):
        def kern(src, targ):
            calls.append(label)
            return np.zeros((targ.r.shape[1], src.r.shape[1]))

        return kern

    blocks = np.empty((4, 4), dtype=object)
    for iedge in range(4):
        for jedge in range(4):
            blocks[iedge, jedge] = zero_kernel((iedge, jedge))

    result = rcip.chunkgraph_rcip(cg, blocks, 1, vertices=[1], opts={"nsub": 1, "rcip_savedepth": 1})

    np.testing.assert_array_equal(result.vertices, [1])
    np.testing.assert_array_equal(result.edge_indices[0], [0, 1])
    assert result.kernels[0].shape == (2, 2)
    assert result.kernels[0][0, 1] is blocks[0, 1]
    assert set(calls) <= {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert calls
    assert result.R[0].shape == (4 * cg.k, 4 * cg.k)
    np.testing.assert_allclose(result.R[0], np.eye(4 * cg.k), atol=1e-14)


def test_chunkermat_defaults_to_rcip_on_nonsmooth_chunkgraph_and_evaluates_corners(monkeypatch):
    verts = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    cg = chunkgraph(verts, edges, pref={"k": 6}, cparams={"nchmin": 4})
    chnkr = cg.merged()
    boundary = chnkr.r.reshape(2, chnkr.npt, order="F")
    system_kernel = -2.0 * kernel("lap", "d")

    mat = chunkermat(cg, system_kernel, {"nsub": 4, "rcip_savedepth": 4})

    assert mat.rcip is not None
    assert len(mat.rcip.saved) == 4
    assert getattr(cg, "_last_rcip_context") is mat.rcip
    sigma = np.linalg.solve(np.eye(chnkr.npt) + mat, boundary[0])
    targets = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
    values = chunkerkerneval(cg, system_kernel, sigma, targets).reshape(-1)
    np.testing.assert_allclose(values, targets[0], atol=2e-9)
    near_targets = np.array([[0.99, 0.999, 0.999], [0.2, 0.0, 0.8]])
    pquad_calls: list[tuple[int, int, str]] = []
    original_panel_matrix = pquad.panel_matrix

    def wrapped_panel_matrix(*args, **kwargs):
        pquad_calls.append((args[0].nch, args[1], args[4]))
        return original_panel_matrix(*args, **kwargs)

    monkeypatch.setattr(pquad, "panel_matrix", wrapped_panel_matrix)

    near_values = chunkerkerneval(
        cg,
        system_kernel,
        sigma,
        near_targets,
        {"forceadap": True, "usepquad": True},
    ).reshape(-1)
    np.testing.assert_allclose(near_values, near_targets[0], atol=1e-6)
    assert pquad_calls
    assert any(nch != chnkr.nch for nch, _, _ in pquad_calls)

    direct = chunkermat(cg, system_kernel, {"rcip": False})
    assert not hasattr(direct, "rcip")
    assert getattr(cg, "_last_rcip_context") is None
