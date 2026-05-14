import numpy as np

from chunkie import (
    Kernel,
    PointInfo,
    chunkerfunc,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkgraph,
    kernel,
    merge,
)
from chunkie.kernels import helmholtz as helm2d
from chunkie.kernels import laplace as lap2d
from chunkie.misc import smoother
from chunkie.quadrature import adaptive as quadadap
from chunkie.quadrature import ggq as quadggq
from chunkie.quadrature import rcip

pointinfo = PointInfo.from_any


def _unit_columns(arr):
    arr = np.asarray(arr, dtype=float)
    return arr / np.linalg.norm(arr, axis=0, keepdims=True)


def _stress_pointinfo():
    src_r = np.array(
        [
            [-1.2, -0.35, 0.15, 0.85, 1.25],
            [0.45, -0.75, 0.35, -0.2, 0.9],
        ]
    )
    targ_r = np.array(
        [
            [-0.95, 0.1, 0.72, 1.45],
            [0.15, -0.42, 0.88, -0.05],
        ]
    )
    src_n = _unit_columns(
        np.array([[0.8, -0.25, 0.5, -0.7, 0.1], [0.6, 0.97, -0.86, 0.71, -0.99]])
    )
    targ_n = _unit_columns(np.array([[0.3, -0.82, 0.68, -0.55], [0.95, 0.57, -0.73, 0.84]]))
    src_d = np.vstack((-src_n[1], src_n[0]))
    targ_d = np.vstack((-targ_n[1], targ_n[0]))
    return PointInfo(r=src_r, d=src_d, n=src_n), PointInfo(r=targ_r, d=targ_d, n=targ_n)


def _wobbly_curve(t):
    t = np.asarray(t)
    rad = 1.0 + 0.18 * np.cos(3.0 * t) + 0.07 * np.sin(5.0 * t)
    drad = -0.54 * np.sin(3.0 * t) + 0.35 * np.cos(5.0 * t)
    d2rad = -1.62 * np.cos(3.0 * t) - 1.75 * np.sin(5.0 * t)
    theta = t + 0.15 * np.sin(2.0 * t)
    dtheta = 1.0 + 0.3 * np.cos(2.0 * t)
    d2theta = -0.6 * np.sin(2.0 * t)

    c = np.cos(theta)
    s = np.sin(theta)
    e = np.vstack((c, s))
    ep = np.vstack((-s, c))
    r = rad * e
    d = drad * e + rad * dtheta * ep
    d2 = (d2rad - rad * dtheta**2) * e + (2.0 * drad * dtheta + rad * d2theta) * ep
    return r, d, d2


def _wobbly_chunker(k=8, nchmin=7):
    chnkr, _ = chunkerfunc(_wobbly_curve, min_chunks=nchmin, refine=False, order=k)
    return chnkr


def _smooth_vector_kernel(src, targ):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    normal_term = 0.0 if src.n is None else 0.15 * src.n[0, None, :]
    out = np.empty((2 * targ.r.shape[1], src.r.shape[1]))
    out[0::2, :] = 1.0 + 0.2 * dx + dx**2 + 0.3 * dy**2
    out[1::2, :] = np.sin(1.3 * dx) + np.cos(0.7 * dy) + normal_term
    return out


def _block(mat, chnkr, target_chunk, source_chunk, opdims=(1, 1)):
    row_start = target_chunk * chnkr.k * opdims[0]
    row_stop = row_start + chnkr.k * opdims[0]
    col_start = source_chunk * chnkr.k * opdims[1]
    col_stop = col_start + chnkr.k * opdims[1]
    return mat[row_start:row_stop, col_start:col_stop]


def _polygon_area(verts):
    x = verts[0]
    y = verts[1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def test_point_kernels_stress_combined_selectors_and_green_gradients():
    src, targ = _stress_pointinfo()
    lap_coefs = (1.7, -0.35)
    helm_coefs = (-0.4 + 0.2j, 1.1 - 0.15j)
    zk = 1.4 + 0.35j

    lap_s = lap2d.kern(src, targ, "s")
    lap_d = lap2d.kern(src, targ, "d")
    lap_sp = lap2d.kern(src, targ, "sp")
    lap_dp = lap2d.kern(src, targ, "dp")
    lap_sgrad = lap2d.kern(src, targ, "sgrad")
    lap_dgrad = lap2d.kern(src, targ, "dgrad")
    np.testing.assert_allclose(lap2d.kern(src, targ, "c", lap_coefs), lap_coefs[0] * lap_d + lap_coefs[1] * lap_s)
    np.testing.assert_allclose(lap2d.kern(src, targ, "cp", lap_coefs), lap_coefs[0] * lap_dp + lap_coefs[1] * lap_sp)
    np.testing.assert_allclose(
        lap2d.kern(src, targ, "cgrad", lap_coefs),
        lap_coefs[0] * lap_dgrad + lap_coefs[1] * lap_sgrad,
    )
    np.testing.assert_allclose(
        helm2d.kern(zk, src, targ, "c", helm_coefs),
        helm_coefs[0] * helm2d.kern(zk, src, targ, "d")
        + helm_coefs[1] * helm2d.kern(zk, src, targ, "s"),
    )
    np.testing.assert_allclose(
        helm2d.kern(zk, src, targ, "cp", helm_coefs),
        helm_coefs[0] * helm2d.kern(zk, src, targ, "dp")
        + helm_coefs[1] * helm2d.kern(zk, src, targ, "sp"),
    )

    eps = 2.0e-6
    _, lap_grad, _ = lap2d.green(src.r, targ.r)
    _, helm_grad, _ = helm2d.green(zk, src.r, targ.r)
    for axis in range(2):
        shift = np.zeros_like(targ.r)
        shift[axis, :] = eps
        lap_fd = (lap2d.green(src.r, targ.r + shift)[0] - lap2d.green(src.r, targ.r - shift)[0]) / (
            2.0 * eps
        )
        helm_fd = (
            helm2d.green(zk, src.r, targ.r + shift)[0] - helm2d.green(zk, src.r, targ.r - shift)[0]
        ) / (2.0 * eps)
        np.testing.assert_allclose(lap_grad[:, :, axis], lap_fd, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(helm_grad[:, :, axis], helm_fd, rtol=2e-8, atol=2e-9)


def test_dense_native_operator_stress_on_wobbly_curve_matches_manual_weighting():
    chnkr = _wobbly_chunker(k=8, nchmin=7)
    vec_kernel = Kernel(eval=_smooth_vector_kernel, opdims=(2, 1), sing="smooth")
    src = pointinfo(chnkr)
    weights = chnkr.wts.reshape(-1, order="F")

    dense = chunkermat(chnkr, vec_kernel)
    raw = _smooth_vector_kernel(src, src)
    np.testing.assert_allclose(dense, raw * weights[None, :], atol=1e-13)

    targets = PointInfo(
        r=np.array([[1.45, -1.1, 0.25, 0.8], [0.05, 0.45, -1.35, 1.2]]),
        n=_unit_columns(np.array([[0.8, -0.4, 0.1, -0.6], [0.6, 0.92, -0.99, 0.8]])),
    )
    dens = 0.4 + np.sin(src.r[0]) - 0.2 * np.cos(2.0 * src.r[1])
    eval_mat = chunkerkernevalmat(chnkr, vec_kernel, targets)
    eval_vals = chunkerkerneval(chnkr, vec_kernel, dens, targets).reshape(-1, order="F")
    manual = _smooth_vector_kernel(src, targets) @ (dens * weights)

    np.testing.assert_allclose(eval_mat @ dens, manual, atol=1e-13)
    np.testing.assert_allclose(eval_vals, manual, atol=1e-13)


def test_quadggq_stress_noncircle_complex_special_blocks_and_robust_close_eval():
    chnkr = _wobbly_chunker(k=8, nchmin=7)
    helm_s = kernel("helm", "s", 1.15 + 0.25j)

    full = quadggq.buildmat(chnkr, helm_s, helm_s.opdims, "log")
    topological = quadggq.buildmattd(chnkr, helm_s, helm_s.opdims, "log").toarray()
    source_chunk = 2
    neighbor_chunk = int(chnkr.adj[1, source_chunk] - 1)
    excluded = {source_chunk, int(chnkr.adj[0, source_chunk] - 1), neighbor_chunk}
    far_chunk = next(idx for idx in range(chnkr.nch) if idx not in excluded)

    np.testing.assert_allclose(
        _block(topological, chnkr, source_chunk, source_chunk),
        _block(full, chnkr, source_chunk, source_chunk),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _block(topological, chnkr, neighbor_chunk, source_chunk),
        _block(full, chnkr, neighbor_chunk, source_chunk),
        atol=1e-12,
    )
    np.testing.assert_allclose(_block(topological, chnkr, far_chunk, source_chunk), 0.0, atol=1e-14)

    skipped = quadggq.buildmattd(chnkr, helm_s, helm_s.opdims, "log", ilist=[source_chunk]).toarray()
    np.testing.assert_allclose(_block(skipped, chnkr, source_chunk, source_chunk), 0.0, atol=1e-14)
    assert np.linalg.norm(_block(skipped, chnkr, neighbor_chunk, source_chunk)) > 1e-8

    close_left = _wobbly_chunker(k=6, nchmin=4)
    close_right = close_left + np.array([2.05, 0.04])
    close_pair = merge([close_left, close_right])
    lap_s = kernel("lap", "s")
    standard = quadadap.buildmat(
        close_pair,
        lap_s,
        opts={"sing": "log", "robust": False, "eps": 1e-9},
    )
    robust = quadadap.buildmat(
        close_pair,
        lap_s,
        opts={"sing": "log", "robust": True, "eps": 1e-9},
    )

    assert np.isfinite(robust).all()
    assert np.linalg.norm(robust - standard) > 1e-10


def test_schurbana_stress_matches_independent_block_update():
    rng = np.random.default_rng(20260511)
    nstar = 6
    ngood = 3
    nred = 3
    size = nstar + ngood
    pbc = rng.normal(size=(nstar, nred))
    pwbc = rng.normal(size=(nstar, nred))
    kmat = rng.normal(size=(size, size))
    kmat[np.ix_(range(nstar, size), range(nstar, size))] += 4.0 * np.eye(ngood)
    amat = rng.normal(size=(nstar, nstar)) + 3.0 * np.eye(nstar)
    amat_in = amat.copy()

    star_l = np.arange(nstar)
    circ_l = np.arange(nstar, size)
    star_s = np.arange(nred)
    circ_s = np.arange(nred, nred + ngood)
    out = rcip.SchurBana(pbc, pwbc, kmat, amat, star_l, circ_l, star_s, circ_s)

    expected = amat.copy()
    va = kmat[np.ix_(circ_l, star_l)] @ expected
    pta = pwbc.T @ expected
    ptau = pta @ kmat[np.ix_(star_l, circ_l)]
    dvaui = np.linalg.inv(kmat[np.ix_(circ_l, circ_l)] - va @ kmat[np.ix_(star_l, circ_l)])
    dvauivap = dvaui @ (va @ pbc)
    expected[np.ix_(star_s, star_s)] = pta @ pbc + ptau @ dvauivap
    expected[np.ix_(circ_s, circ_s)] = dvaui
    expected[np.ix_(circ_s, star_s)] = -dvauivap
    expected[np.ix_(star_s, circ_s)] = -ptau @ dvaui

    np.testing.assert_allclose(out, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(amat, amat_in)


def test_chunkgraph_rcip_stress_nonorthogonal_vertex_and_global_blocks():
    verts = np.array([[0.0, 1.25, 1.75, 0.55, -0.35], [0.0, -0.15, 0.9, 1.55, 0.75]])
    edges = np.vstack((np.arange(verts.shape[1]), np.roll(np.arange(verts.shape[1]), -1)))
    cg = chunkgraph(verts, edges, pref={"k": 4}, cparams={"_chunkie_normalized_geometry_options": True, "nchmin": 2})
    nedge = len(cg.echnks)
    blocks = np.empty((nedge, nedge), dtype=object)
    lap_d = kernel("lap", "d")
    for iedge in range(nedge):
        for jedge in range(nedge):
            blocks[iedge, jedge] = (1.0 + 0.07 * iedge - 0.03 * jedge) * lap_d

    result = rcip.chunkgraph_rcip(
        cg,
        blocks,
        1,
        vertices=[1, 3],
        ignore_vertices=[3],
        opts={"_chunkie_normalized_operator_options": True, "nsub": 2, "rcip_savedepth": 2},
    )
    expected_edges = np.asarray(cg.vstruc[1][0], dtype=int)

    np.testing.assert_array_equal(result.vertices, [1])
    np.testing.assert_array_equal(result.edge_indices[0], expected_edges)
    assert result.kernels[0].shape == (2, 2)
    for local_i, global_i in enumerate(expected_edges):
        for local_j, global_j in enumerate(expected_edges):
            assert result.kernels[0][local_i, local_j] is blocks[global_i, global_j]
    assert result.R[0].shape == (4 * cg.k, 4 * cg.k)
    assert np.isfinite(result.R[0]).all()
    assert np.linalg.norm(result.R[0] - np.eye(result.R[0].shape[0])) > 1e-6

    rhohat = np.linspace(-0.4, 0.9, result.R[0].shape[0])
    rho, srcinfo, wts = rcip.rhohatInterp(rhohat, result.saved[0], 2)
    assert len(rho) == 2
    assert all(item.size == 4 * cg.k for item in rho)
    assert all(info.r.shape == (2, 4 * cg.k) for info in srcinfo)
    assert all(weight.shape == (4 * cg.k,) for weight in wts)


def test_interleaved_fmm_stress_matches_direct_on_wobbly_curve():
    chnkr = _wobbly_chunker(k=8, nchmin=7)
    src = pointinfo(chnkr)
    targets = PointInfo(
        r=np.array([[1.65, -1.45, 0.35, -0.15, 0.95], [0.1, 0.55, -1.4, 1.35, -0.95]]),
        n=_unit_columns(
            np.array([[0.95, -0.25, 0.3, -0.75, 0.55], [0.31, 0.97, -0.95, 0.66, -0.84]])
        ),
    )
    zk = 1.2 + 0.1j
    mixed = kernel(
        [
            [kernel("helm", "d", zk), -0.35 * kernel("helm", "s", zk)],
            [kernel("lap", "d"), 0.2 * kernel("lap", "s")],
        ]
    )
    density = np.vstack(
        (
            np.cos(1.4 * src.r[0]) + 0.2j * np.sin(src.r[1]),
            np.sin(1.1 * src.r[1]) - 0.15j * np.cos(0.7 * src.r[0]),
        )
    ).reshape(-1, order="F")

    direct = chunkerkerneval(chnkr, mixed, density, targets)
    via_fmm = chunkerkerneval(chnkr, mixed, density, targets, acceleration="fmm", tol=1e-12)

    np.testing.assert_allclose(via_fmm, direct, rtol=5e-9, atol=5e-10)


def test_smoother_stress_returns_valid_rounded_asymmetric_polygon():
    verts = np.array([[0.0, 1.35, 1.8, 0.8, -0.3, -0.55], [0.0, 0.12, 1.0, 1.65, 1.25, 0.45]])
    widths = np.array([0.05, 0.07, 0.06, 0.08, 0.05, 0.04])
    chnkr, err, err_by_pt = smoother.smooth(verts, {"k": 10, "widths": widths, "return_error": True})

    assert chnkr.k == 10
    assert chnkr.nch == 2 * verts.shape[1]
    assert err == 0
    np.testing.assert_array_equal(err_by_pt, np.zeros(chnkr.npt))
    assert np.isfinite(chnkr.r).all()
    assert np.isfinite(chnkr.d).all()
    assert np.all(chnkr.chunklen() > 0)
    np.testing.assert_allclose(np.linalg.norm(chnkr.n, axis=0), 1.0, atol=1e-14)
    assert chnkr.checkadjinfo() == 0
    assert 0.0 < chnkr.area() < _polygon_area(verts)
