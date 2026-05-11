import numpy as np

from chunkie import chunkgraph, kernel, lege
from chunkie.chnk import rcip


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

    assert pbc.shape == (48, 24)
    assert pwbc.shape == (48, 24)
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
    np.testing.assert_allclose(out, out)


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

    rho, srcinfo, wts = rcip.rhohatInterp(np.arange(rmat.shape[0], dtype=float), saved, 2)
    assert len(rho) == 2
    assert rho[0].size == 4 * cg.k
    assert srcinfo[0].r.shape == (2, 4 * cg.k)
    assert wts[0].shape == (4 * cg.k,)
