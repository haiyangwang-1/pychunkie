import numpy as np
import pytest

from chunkie import (
    Chunker,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerintegral,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkerpoly,
    chunkgraph,
    kernel,
    lege,
    tochunkgraph,
)
from chunkie.chnk import arcparam, curves, flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself, helm2d, quadadap, smoother, spcl
from chunkie.operators import PointInfo, pointinfo
from _fixture_generation import load_generated_mat_fixture


def load_devtools_easy():
    return load_generated_mat_fixture("devtools_easy.mat", squeeze_me=True, struct_as_record=False)["devtools_easy"]


def chunker_from_fields(fields) -> Chunker:
    k = int(fields.k)
    nch = int(fields.nch)
    dim = int(fields.dim)
    chnkr = Chunker(
        {"k": k, "dim": dim, "nchstor": nch, "nchmax": nch},
        np.asarray(fields.tstor).reshape(-1),
        np.asarray(fields.wstor).reshape(-1),
    )
    chnkr.addchunk(nch)
    chnkr.r = np.asarray(fields.r)
    chnkr.d = np.asarray(fields.d)
    chnkr.d2 = np.asarray(fields.d2)
    chnkr.n = np.asarray(fields.n)
    chnkr.wts = np.asarray(fields.wts)
    chnkr.adj = np.asarray(fields.adj, dtype=int)
    return chnkr


def pointinfo_from_mat(obj) -> PointInfo:
    return PointInfo(
        r=point_array(obj.r),
        d=point_array(obj.d) if hasattr(obj, "d") else None,
        d2=point_array(obj.d2) if hasattr(obj, "d2") else None,
        n=point_array(obj.n) if hasattr(obj, "n") else None,
    )


def point_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def laplace_green_identity_quantities(fixture):
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    targ = PointInfo(r=point_array(fixture.targets))
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    lap_s = kernel("lap", "s")
    lap_sp = kernel("lap", "sprime")
    lap_d = kernel("lap", "d")
    boundary = pointinfo(chnkr)
    densu = lap_s(src, boundary) @ strengths
    densun = lap_sp(src, boundary) @ strengths
    utarg = lap_s(src, targ) @ strengths
    return chnkr, lap_s, lap_d, densu, densun, utarg


def helmholtz_green_identity_quantities(fixture):
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    targ = PointInfo(r=point_array(fixture.targets))
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    helm_s = kernel("helm", "s", fixture.zk)
    helm_sp = kernel("helm", "sprime", fixture.zk)
    helm_d = kernel("helm", "d", fixture.zk)
    boundary = pointinfo(chnkr)
    densu = helm_s(src, boundary) @ strengths
    densun = helm_sp(src, boundary) @ strengths
    utarg = helm_s(src, targ) @ strengths
    return chnkr, helm_s, helm_d, densu, densun, utarg


def sorted_pairs(pairs: np.ndarray) -> np.ndarray:
    arr = np.asarray(pairs, dtype=int)
    if arr.size == 0:
        return arr.reshape(2, 0)
    order = np.lexsort((arr[1], arr[0]))
    return arr[:, order]


def assert_chunker_fields_match(chnkr: Chunker, fields, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(chnkr.r, fields.r, atol=atol)
    np.testing.assert_allclose(chnkr.d, fields.d, atol=atol)
    np.testing.assert_allclose(chnkr.d2, fields.d2, atol=atol)
    np.testing.assert_allclose(chnkr.n, fields.n, atol=atol)
    np.testing.assert_allclose(chnkr.wts, fields.wts, atol=atol)
    np.testing.assert_array_equal(chnkr.adj, np.asarray(fields.adj, dtype=int))
    np.testing.assert_allclose(chnkr.chunklen(), fields.chunklen, atol=atol)
    np.testing.assert_allclose(chnkr.area(), fields.area, atol=atol)


def test_absconvgauss_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().absconvgauss

    val, der, der2 = spcl.absconvgauss(fixture.x, float(fixture.m), float(fixture.offset), float(fixture.h))

    np.testing.assert_allclose(val, fixture.val, atol=1e-15)
    np.testing.assert_allclose(der, fixture.der, atol=1e-15)
    np.testing.assert_allclose(der2, fixture.der2, atol=1e-15)
    np.testing.assert_array_less(np.min(np.abs(fixture.errsf), axis=0), 1e-8)
    np.testing.assert_array_less(np.min(np.abs(fixture.errsg), axis=0), 1e-6)


def test_legeexpsunit_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().legeexpsunit
    k = int(fixture.k)
    x, w, u, v = lege.exps(k)
    dmat = lege.dermat(k, u, v)
    imat, *_ = lege.intmat(k, u, v)

    np.testing.assert_allclose(x, fixture.x, atol=1e-14)
    np.testing.assert_allclose(w, fixture.w, atol=1e-14)
    np.testing.assert_allclose(u, fixture.u, atol=1e-13)
    np.testing.assert_allclose(v, fixture.v, atol=1e-13)
    np.testing.assert_allclose(dmat, fixture.dmat, atol=1e-12)
    np.testing.assert_allclose(imat, fixture.imat, atol=1e-13)
    np.testing.assert_allclose(dmat @ np.sin(x), fixture.dpv, atol=1e-12)
    np.testing.assert_allclose(imat @ np.sin(x), fixture.ipv, atol=1e-13)
    np.testing.assert_allclose(lege.intpol(fixture.cfs), fixture.cfsint, atol=1e-13)
    np.testing.assert_allclose(lege.intpol(fixture.cfs, "original"), fixture.cfsint_original, atol=1e-13)
    np.testing.assert_allclose(lege.exev(x, lege.intpol(fixture.cfs)), fixture.integral_vals, atol=5e-14)


def test_arclengthfun_single_component_devtools_output_matches_matlab():
    fixture = load_devtools_easy().arclengthfun
    chnkr = chunker_from_fields(fixture.chunker_single)

    np.testing.assert_allclose(chnkr.arclengthfun(), fixture.s_single, atol=1e-13)
    np.testing.assert_allclose(fixture.s_single, fixture.ts_single, atol=1e-12)
    np.testing.assert_allclose(chnkr.chunklen(), fixture.chunker_single.chunklen, atol=1e-13)


def test_arclengthfun_merged_components_devtools_output_matches_matlab():
    fixture = load_devtools_easy().arclengthfun
    chnkr = chunker_from_fields(fixture.chunker_merged)

    np.testing.assert_allclose(chnkr.arclengthfun(), fixture.s_merged, atol=1e-13)


def test_chunkerarcparam_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerarcparam
    chnkr = chunker_from_fields(fixture.chunker)
    pdata = arcparam.init(chnkr)

    r_nodes, d_nodes, d2_nodes = arcparam.eval(np.asarray(fixture.s_nodes).reshape(-1, order="F"), pdata)
    np.testing.assert_allclose(r_nodes, fixture.r_nodes, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(d_nodes, fixture.d_nodes, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(d2_nodes, fixture.d2_nodes, rtol=1e-8, atol=1e-9)
    assert float(fixture.node_residual) < 1e-10

    sample_r, sample_d, sample_d2 = arcparam.eval(fixture.sample_s, pdata)
    np.testing.assert_allclose(sample_r, fixture.sample_r, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(sample_d, fixture.sample_d, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(sample_d2, fixture.sample_d2, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(fixture.der_r_residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(fixture.der_d_residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(fixture.orthogonality, 0.0, atol=1e-8)

    resampled, eps = chnkr.arcresample({"mv_bdries": 0})
    assert_chunker_fields_match(resampled, fixture.resampled, atol=1e-10)
    np.testing.assert_allclose(eps, fixture.resampled_eps, rtol=1e-8, atol=1e-12)
    assert float(fixture.resampled_area_err) < 1e-8
    assert float(fixture.resampled_len_err) < 1e-8
    np.testing.assert_allclose(fixture.resampled_speed_ratio, 1.0, atol=1e-6)


def test_chunker_diffintmat_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunker_diffintmat
    ellipse = chunker_from_fields(fixture.ellipse)
    dmat = ellipse.diffmat()
    imat = ellipse.intmat()
    x = ellipse.r[0].reshape(-1, order="F")
    y = ellipse.r[1].reshape(-1, order="F")
    dx = dmat @ x
    dy = dmat @ y
    x_int = imat @ dx
    y_int = imat @ dy

    np.testing.assert_allclose(dmat, fixture.ellipse_D, atol=1e-13)
    np.testing.assert_allclose(imat, fixture.ellipse_C, atol=1e-13)
    np.testing.assert_allclose(dx, fixture.ellipse_dx, atol=1e-12)
    np.testing.assert_allclose(dy, fixture.ellipse_dy, atol=1e-12)
    np.testing.assert_allclose(x_int, fixture.ellipse_x_int, atol=1e-12)
    np.testing.assert_allclose(y_int, fixture.ellipse_y_int, atol=1e-12)
    np.testing.assert_allclose(dx**2 + dy**2 - 1.0, fixture.ellipse_tangent_residual, atol=1e-12)
    np.testing.assert_allclose(x_int - x_int[0] - x + x[0], fixture.ellipse_x_residual, atol=1e-12)
    np.testing.assert_allclose(y_int - y_int[0] - y + y[0], fixture.ellipse_y_residual, atol=1e-12)
    assert np.linalg.norm(dx**2 + dy**2 - 1.0) < 1e-10
    assert np.linalg.norm(x_int - x_int[0] - x + x[0]) < 1e-10
    assert np.linalg.norm(y_int - y_int[0] - y + y[0]) < 1e-10

    circle = chunker_from_fields(fixture.circle)
    circle_dmat = circle.diffmat()
    circle_test_quant = circle_dmat @ circle.r[0].reshape(-1, order="F") + circle.r[1].reshape(-1, order="F")
    np.testing.assert_allclose(circle_dmat, fixture.circle_D, atol=1e-13)
    np.testing.assert_allclose(circle_test_quant, fixture.circle_test_quant, atol=1e-12)
    assert np.linalg.norm(circle_test_quant) < 1e-10


def test_chunker_nearest_devtools_output_matches_matlab():
    fixture = load_devtools_easy().chunker_nearest
    circle = chunker_from_fields(fixture.circle)

    rn, dn, d2n, dist, tn, ichn = circle.nearest(fixture.targs)
    theta_targ = np.arctan2(fixture.targs[1], fixture.targs[0])
    theta_near = np.arctan2(rn[1], rn[0])
    angle_err = np.abs(np.angle(np.exp(1j * (theta_targ - theta_near))))
    expected_ichn = np.asarray(fixture.ichn, dtype=int).reshape(-1) - 1

    np.testing.assert_allclose(rn, fixture.rn, rtol=0, atol=5e-12)
    np.testing.assert_allclose(dn, fixture.dn, rtol=0, atol=5e-12)
    np.testing.assert_allclose(d2n, fixture.d2n, rtol=0, atol=3e-11)
    np.testing.assert_allclose(dist, fixture.dist, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tn, fixture.tn, rtol=0, atol=5e-12)
    np.testing.assert_array_equal(ichn, expected_ichn)
    np.testing.assert_array_less(fixture.err, 1e-12)
    np.testing.assert_allclose(angle_err, fixture.err, rtol=0, atol=4e-12)
    np.testing.assert_array_less(angle_err, 5e-12)


def test_chunkerintegral_devtools_output_matches_matlab():
    fixture = load_devtools_easy().chunkerintegral
    chnkr = chunker_from_fields(fixture.chunker)

    def fscal(xx):
        return np.cos(xx[0] - 1.0) + np.sin(xx[1] - 0.5)

    fvals = fscal(chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F"))
    actual_from_values = chunkerintegral(chnkr, fixture.fvals, {"usesmooth": False})
    actual_from_callable = chunkerintegral(chnkr, fscal, {"usesmooth": False})

    np.testing.assert_allclose(fvals, fixture.fvals, atol=1e-13)
    np.testing.assert_allclose(actual_from_values, fixture.fscal_int1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(actual_from_callable, fixture.fscal_int2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(actual_from_values, fixture.fscal_int3, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(actual_from_callable, fixture.fscal_int4, rtol=1e-9, atol=1e-12)
    assert float(fixture.relerr12) < 1e-9
    assert float(fixture.relerr32) < 1e-9
    assert float(fixture.relerr42) < 1e-9


def test_chunkerfuncuni_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerfuncuni

    starfish = chunkerfuncuni(
        lambda t: curves.starfish(t, int(fixture.narms), float(fixture.amp)),
        int(fixture.nch),
    )
    bymode_reversed = chunkerfuncuni(
        lambda t: curves.bymode(t, fixture.modes, fixture.mode_ctr),
        int(fixture.nch),
    ).reverse()

    def circle(t):
        radius = float(fixture.circle_radius)
        ctr = np.asarray(fixture.circle_ctr).reshape(2)
        return (
            np.vstack((ctr[0] + radius * np.cos(t), ctr[1] + radius * np.sin(t))),
            np.vstack((-radius * np.sin(t), radius * np.cos(t))),
            np.vstack((-radius * np.cos(t), -radius * np.sin(t))),
        )

    circle_chunker = chunkerfuncuni(circle, int(fixture.nch))

    assert int(fixture.starfish_sort_ier) == 0
    assert int(fixture.circle_sort_ier) == 0
    assert float(fixture.circle_area_error) < 1e-12
    assert_chunker_fields_match(starfish, fixture.starfish, atol=1e-12)
    assert_chunker_fields_match(bymode_reversed, fixture.bymode_reversed, atol=1e-12)
    assert_chunker_fields_match(circle_chunker, fixture.circle, atol=1e-12)
    np.testing.assert_allclose(circle_chunker.area(), np.pi * float(fixture.circle_radius) ** 2, atol=1e-12)


def test_chunkerfunc_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerfunc
    cparams = {"eps": 1.0e-4}
    pref = {"k": 16}

    starfish, _ = chunkerfunc(
        lambda t: curves.starfish(t, int(fixture.narms), float(fixture.amp)),
        cparams,
        pref,
    )
    starfish_nout, _ = chunkerfunc(
        lambda t: curves.starfish(t, int(fixture.narms), float(fixture.amp)),
        {**cparams, "nout": 3},
        pref,
    )
    bymode, _ = chunkerfunc(
        lambda t: curves.bymode(t, fixture.modes, fixture.mode_ctr),
        {**cparams, "nout": 3},
    )
    bymode_reversed = bymode.reverse()

    def circle(t):
        radius = float(fixture.circle_radius)
        ctr = np.asarray(fixture.circle_ctr).reshape(2)
        return np.vstack((ctr[0] + radius * np.cos(t), ctr[1] + radius * np.sin(t)))

    circle_chunker, _ = chunkerfunc(circle, {**cparams, "nout": 3})
    circle_refined = circle_chunker.refine({"nover": 1})

    assert int(fixture.starfish_ier) == 0
    assert int(fixture.starfish_nout_ier) == 0
    assert int(fixture.bymode_ier) == 0
    assert int(fixture.bymode_reversed_ier) == 0
    assert int(fixture.circle_ier) == 0
    assert bool(fixture.closed_warning_seen)
    assert bool(fixture.near_closed_warning_seen)
    assert not bool(fixture.open_warning_seen)
    assert_chunker_fields_match(starfish, fixture.starfish, atol=1e-11)
    assert_chunker_fields_match(starfish_nout, fixture.starfish_nout, atol=1e-11)
    assert_chunker_fields_match(bymode, fixture.bymode, atol=1e-11)
    assert_chunker_fields_match(bymode_reversed, fixture.bymode_reversed, atol=1e-11)
    assert_chunker_fields_match(circle_chunker, fixture.circle, atol=1e-12)
    assert_chunker_fields_match(circle_refined, fixture.circle_refined, atol=1e-12)
    assert float(fixture.circle_area_error) < 1e-12
    assert float(fixture.circle_refined_area_error) < 1e-12
    np.testing.assert_allclose(circle_chunker.area(), np.pi * float(fixture.circle_radius) ** 2, atol=1e-12)
    np.testing.assert_allclose(circle_refined.area(), np.pi * float(fixture.circle_radius) ** 2, atol=1e-12)


def test_chunkerclassunit_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerclassunit

    with pytest.raises(ValueError):
        Chunker({"k": -1})
    with pytest.raises(ValueError):
        t, w, *_ = lege.exps(8)
        Chunker({"k": 9}, t, w)
    with pytest.raises(ValueError):
        chunkerfunc(lambda t: curves.starfish(t), {"nchmin": 101}, {"k": 4, "nchmax": 100})

    chnkr = chunker_from_fields(fixture.chunker)
    for j in range(chnkr.nch):
        i1 = chnkr.adj[0, j]
        i2 = chnkr.adj[1, j]
        if i1 > 0:
            assert chnkr.adj[1, i1 - 1] == j + 1
        if i2 > 0:
            assert chnkr.adj[0, i2 - 1] == j + 1

    plus_left = fixture.v + chnkr
    plus_right = chnkr + fixture.v
    mat_left = fixture.A @ chnkr
    scale_left = float(fixture.s) * chnkr
    scale_right = chnkr * float(fixture.s)

    assert bool(fixture.fail_negative_k)
    assert bool(fixture.fail_wrong_nodes)
    assert bool(fixture.fail_nchmax)
    assert bool(fixture.fail_right_matrix)
    assert bool(fixture.adj_ok)
    assert_chunker_fields_match(plus_left, fixture.plus_left, atol=1e-12)
    assert_chunker_fields_match(plus_right, fixture.plus_right, atol=1e-12)
    assert_chunker_fields_match(mat_left, fixture.mat_left, atol=1e-12)
    assert_chunker_fields_match(scale_left, fixture.scale_left, atol=1e-12)
    assert_chunker_fields_match(scale_right, fixture.scale_right, atol=1e-12)
    np.testing.assert_allclose(fixture.com_plus_left - fixture.com1, fixture.v, atol=1e-14)
    np.testing.assert_allclose(fixture.com_plus_right - fixture.com1, fixture.v, atol=1e-14)
    np.testing.assert_allclose(mat_left.area(), np.linalg.det(fixture.A) * chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(scale_left.area(), float(fixture.s) ** 2 * chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(scale_right.area(), float(fixture.s) ** 2 * chnkr.area(), atol=1e-13)
    with pytest.raises(TypeError):
        _ = chnkr * fixture.A


def test_chunkerfit_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerfit
    r, _, _ = curves.bymode(fixture.tt, fixture.modes)

    closed = chunkerfit(r, {"ifclosed": True, "cparams": {"eps": 1.0e-6}, "pref": {"k": 16}})
    open_chnkr = chunkerfit(r[:, :10], {"ifclosed": False, "cparams": {"eps": 1.0e-6}, "pref": {"k": 16}})

    np.testing.assert_allclose(r, fixture.r, atol=1e-13)
    assert int(fixture.closed_ier) == 0
    assert int(fixture.open_ier) == 0
    assert closed.checkadjinfo() == int(fixture.closed_ier)
    assert open_chnkr.checkadjinfo() == int(fixture.open_ier)


def test_tochunkgraph_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().tochunkgraph
    total = chunker_from_fields(fixture.total)
    arc = chunker_from_fields(fixture.arc)
    circle = chunker_from_fields(fixture.circle)

    graph = tochunkgraph(total)

    assert graph.verts.shape[1] == int(fixture.graph_nverts)
    assert len(graph.echnks) == int(fixture.graph_nedges)
    assert graph.npt == int(fixture.graph_npt)
    assert graph.npt == total.npt
    np.testing.assert_allclose(graph.verts, fixture.graph_verts, atol=1e-12)
    np.testing.assert_array_equal(graph.edgesendverts, np.asarray(fixture.graph_edgesendverts, dtype=int) - 1)
    assert_chunker_fields_match(graph.echnks[0], fixture.graph_first_edge, atol=1e-12)
    np.testing.assert_allclose(graph.echnks[0].r, arc.r, atol=1e-14)

    manual = chunkgraph(fixture.manual_verts, fixture.manual_edge2verts, [arc, circle])
    np.testing.assert_allclose(manual.verts, fixture.manual_graph_verts, atol=1e-14)
    np.testing.assert_array_equal(manual.edgesendverts, np.asarray(fixture.manual_graph_edgesendverts, dtype=int) - 1)
    np.testing.assert_allclose(manual.echnks[0].r[:, 0, 0], fixture.manual_verts[:, 0], atol=1e-2)
    np.testing.assert_allclose(manual.echnks[0].r[:, -1, -1], fixture.manual_verts[:, 1], atol=1e-2)
    np.testing.assert_allclose(manual.echnks[0].r[:, 0, 0], fixture.manual_first_start, atol=1e-12)
    np.testing.assert_allclose(manual.echnks[0].r[:, -1, -1], fixture.manual_first_end, atol=1e-12)


def test_slicegraph_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().slicegraph
    graph = chunkgraph(fixture.verts, np.asarray(fixture.edge_2_verts, dtype=int) - 1)
    mixed_edges = np.asarray(fixture.ichs_mixed, dtype=int).reshape(-1) - 1
    inner_edges = np.asarray(fixture.ichs_inner, dtype=int).reshape(-1) - 1

    mixed = graph.slicegraph(mixed_edges)
    inner = graph.slicegraph(inner_edges)
    lap_d = -2 * kernel("lap", "d")
    full = chunkermat(graph, lap_d)
    inner_mat = chunkermat(inner, lap_d)
    idslce = np.asarray(fixture.idslce, dtype=int).reshape(-1) - 1

    assert graph.npt == int(fixture.npt)
    assert len(graph.echnks) == int(fixture.nedges)
    np.testing.assert_allclose(mixed.r, fixture.mixed_r, atol=1e-13)
    np.testing.assert_allclose(mixed.r, fixture.mixed_merge_r, atol=1e-13)
    np.testing.assert_allclose(fixture.inner_sysmat, fixture.inner_sysmat_from_full, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(full[np.ix_(idslce, idslce)], inner_mat, rtol=1e-12, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(graph.edgeids([2, 3, 1, 0]), np.asarray(fixture.edgeids_outer_permuted, dtype=int).reshape(-1) - 1)
    np.testing.assert_array_equal(graph.edgeids(inner_edges), np.asarray(fixture.edgeids_inner, dtype=int).reshape(-1) - 1)


def test_chunkerinterior_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerinterior
    chnkr = chunker_from_fields(fixture.chunker)

    expected = np.asarray(fixture.expected_scal, dtype=bool).reshape(-1)
    direct = chunkerinterior(chnkr, fixture.targs, {"acceleration": "dense"})
    fmm = chunkerinterior(chnkr, fixture.targs, {"acceleration": "fmm"})

    np.testing.assert_array_equal(np.asarray(getattr(fixture, "in"), dtype=bool).reshape(-1), expected)
    np.testing.assert_array_equal(np.asarray(fixture.in_flam, dtype=bool).reshape(-1), expected)
    np.testing.assert_array_equal(np.asarray(fixture.in_fmm, dtype=bool).reshape(-1), expected)
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(fmm, expected)
    flam = chunkerinterior(chnkr, fixture.targs, {"acceleration": "flam", "occ": 32, "rank_or_tol": 1.0e-8, "useproxy": False})
    np.testing.assert_array_equal(flam, expected)

    inner = chunker_from_fields(fixture.inner_chunker)
    in_chunker = chunkerinterior(chnkr, inner, {"acceleration": "fmm"})
    np.testing.assert_array_equal(np.asarray(fixture.in_chunker, dtype=bool).reshape(-1), True)
    np.testing.assert_array_equal(in_chunker, True)

    axis = chunker_from_fields(fixture.axis_chunker)
    axis_expected = np.asarray(fixture.axis_expected, dtype=bool).reshape(-1)
    axis_actual = chunkerinterior(axis, fixture.axis_targs, {"axissym": True})
    np.testing.assert_array_equal(np.asarray(fixture.axis_in, dtype=bool).reshape(-1), axis_expected)
    np.testing.assert_array_equal(axis_actual, axis_expected)

    stress = chunker_from_fields(fixture.stress_chunker)
    stress_actual = chunkerinterior(stress, [fixture.stress_x, fixture.stress_x])
    stress_expected = np.asarray(fixture.stress_expected, dtype=bool)
    np.testing.assert_array_equal(np.asarray(fixture.stress_in, dtype=bool).reshape(-1), stress_expected.reshape(-1, order="F"))
    np.testing.assert_array_equal(stress_actual, stress_expected)


def test_chunkerpoly_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerpoly
    nverts = np.asarray(fixture.verts).shape[1]
    rounded = chunkerpoly(
        fixture.verts,
        {"widths": 0.1 * np.ones(nverts), "eps": 1.0e-8},
        {"k": 16, "dim": 2},
        fixture.edgevals,
    ).sort()[0]
    truepoly = chunkerpoly(
        fixture.verts,
        {"rounded": False, "depth": 8},
        {"k": 16, "dim": 2},
        fixture.edgevals,
    ).sort()[0]
    open_chnkr = chunkerpoly(
        fixture.open_verts,
        {
            "widths": 0.1 * np.ones(np.asarray(fixture.open_verts).shape[1]),
            "autowidths": True,
            "autowidthsfac": 0.1,
            "ifclosed": False,
            "eps": 1.0e-3,
        },
        {"k": 16, "dim": 2},
    )

    assert int(fixture.rounded_ier) == 0
    assert int(fixture.truepoly_ier) == 0
    assert int(fixture.open_ier) == 0
    assert rounded.checkadjinfo() == int(fixture.rounded_ier)
    assert truepoly.checkadjinfo() == int(fixture.truepoly_ier)
    assert open_chnkr.checkadjinfo() == int(fixture.open_ier)
    assert float(fixture.truepoly_area_err) < 1e-12
    assert float(fixture.truepoly_length_err) < 1e-12


def test_smoother_devtools_output_matches_matlab_thresholds():
    fixture = load_devtools_easy().smoother
    chnkr, err, err_by_pt = smoother.smooth(
        fixture.verts,
        {"lam": float(fixture.opts.lam), "return_error": True},
    )

    np.testing.assert_allclose(fixture.verts, np.asarray([[-0.5, -0.5, 1.0], [np.sqrt(3) / 2, -np.sqrt(3) / 2, 0.0]]), atol=1e-15)
    assert int(fixture.nv) == 3
    fixture_npt = int(getattr(fixture.chunker, "npt", int(fixture.chunker.k) * int(fixture.chunker.nch)))
    assert fixture_npt == int(fixture.chunker.k) * int(fixture.chunker.nch)
    assert float(fixture.err) < 1e-6
    assert float(err) < 1e-6
    assert np.asarray(fixture.err_by_pt).shape == (int(fixture.chunker.npt),)
    assert np.asarray(err_by_pt).shape == (chnkr.npt,)


def test_flagself_devtools_output_matches_matlab():
    fixture = load_devtools_easy().flagself
    actual = flagself(fixture.srcs, fixture.targs)
    expected = np.asarray(fixture.flagslf, dtype=int) - 1

    assert actual.shape[1] == int(fixture.nsrc)
    assert actual.shape[1] == int(fixture.flagged_count)
    assert int(fixture.err_count) == 0
    np.testing.assert_array_equal(sorted_pairs(actual), sorted_pairs(expected))
    np.testing.assert_allclose(fixture.srcs[:, actual[0]], fixture.targs[:, actual[1]], atol=1e-10)


def test_flagnear_devtools_output_matches_matlab():
    fixture = load_devtools_easy().flagnear
    chnkr = chunker_from_fields(fixture.chunker)

    actual = flagnear(chnkr, fixture.targs, {"fac": float(fixture.fac)})
    expected = np.asarray(fixture.flag, dtype=bool)
    expected_bruteforce = np.asarray(fixture.flag_bruteforce, dtype=bool)

    assert int(fixture.sortinfo.ier) == 0
    assert int(fixture.mismatch_count) == 0
    np.testing.assert_array_equal(expected, expected_bruteforce)
    np.testing.assert_array_equal(actual, expected)


def test_flagrect_devtools_output_matches_matlab():
    fixture = load_devtools_easy().flagrect
    chnkr = chunker_from_fields(fixture.chunker)

    direct = flagnear_rectangle(chnkr, fixture.targets)
    grid = flagnear_rectangle_grid(chnkr, fixture.x, fixture.y)
    expected = np.asarray(fixture.flag, dtype=bool)
    expected_grid = np.asarray(fixture.flag_grid, dtype=bool)

    assert int(fixture.mismatch_count) == 0
    np.testing.assert_array_equal(expected, expected_grid)
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(grid, expected_grid)


def test_helm2d_green_devtools_output_matches_matlab():
    fixture = load_devtools_easy().helm2d_green
    val, grad, hess = helm2d.green(fixture.zk, fixture.src, fixture.trg)

    np.testing.assert_allclose(np.squeeze(val), np.squeeze(fixture.val), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.squeeze(grad), np.squeeze(fixture.grad), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.squeeze(hess), np.squeeze(fixture.hess), rtol=1e-13, atol=1e-13)
    assert np.min(np.abs(fixture.errsf)) < 1e-10
    assert np.min(np.abs(fixture.errsfx)) < 1e-8
    assert np.min(np.abs(fixture.errsfy)) < 1e-8


def test_kernelop_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().kernelop
    src = pointinfo_from_mat(fixture.src)
    targ = pointinfo_from_mat(fixture.targ)
    skern = kernel("lap", "s")
    dkern = kernel("helm", "d", 1)
    fkern1 = kernel([[skern], [dkern]])
    scalar = fixture.a

    checks = {
        "skern": skern(src, targ),
        "dkern": dkern(src, targ),
        "fkern1": fkern1(src, targ),
        "fkern2": (scalar * fkern1)(src, targ),
        "fkern3": (fkern1 / scalar)(src, targ),
        "nkern": (-skern)(src, targ),
        "ckern1": (skern + dkern)(src, targ),
        "ckern2": (skern - dkern)(src, targ),
        "conj_dkern": dkern.conj()(src, targ),
    }

    for name, actual in checks.items():
        np.testing.assert_allclose(actual, getattr(fixture, name), atol=1e-13, err_msg=name)


def test_stokes_dtrac_devtools_output_matches_matlab():
    fixture = load_devtools_easy().stokes_dtrac
    src = pointinfo_from_mat(fixture.srcinfo)
    targ = pointinfo_from_mat(fixture.targinfo)
    strengths = np.asarray(fixture.strengths).reshape(-1)
    mu = float(fixture.mu)

    kt = kernel("stok", "dtrac", mu)(src, targ) @ strengths
    kg = kernel("stok", "dgrad", mu)(src, targ) @ strengths
    kp = kernel("stok", "dpres", mu)(src, targ) @ strengths

    du = kg.reshape(2, 2, 1, order="F")
    eu = du + np.transpose(du, (1, 0, 2))
    reconstructed = np.zeros(2)
    reconstructed[0::2] = -kp * targ.n[0] + (eu[0, 0] * targ.n[0] + eu[0, 1] * targ.n[1]) * mu
    reconstructed[1::2] = -kp * targ.n[1] + (eu[0, 1] * targ.n[0] + eu[1, 1] * targ.n[1]) * mu

    np.testing.assert_allclose(kt, fixture.Kt, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(kg, fixture.Kg, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(kp, fixture.Kp, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(reconstructed, fixture.reconstructed, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(reconstructed, kt, rtol=1e-13, atol=1e-13)
    assert float(fixture.residual_norm) < 1e-13


def test_kernelclass_devtools_green_identity_matches_matlab():
    fixture = load_devtools_easy().kernelclass
    chnkr, lap_s, lap_d, densu, densun, utarg = laplace_green_identity_quantities(fixture)

    opts = {"forceadap": True}
    Du = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, opts).reshape(-1, order="F")
    Sun = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, opts).reshape(-1, order="F")
    identity = Sun - Du

    np.testing.assert_allclose(densu, fixture.ubdry, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(densun, fixture.unbdry, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, fixture.utarg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(Du, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Du_fmm, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Sun_fmm, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.utarg_identity_fmm, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(utarg - identity) / np.linalg.norm(utarg) < 1e-11
    assert float(fixture.relerr) < 1e-11
    assert float(fixture.relerr_fmm) < 1e-11
    assert bool(fixture.nankern_isnan)
    assert bool(fixture.sum_isnan)
    assert bool(fixture.scaled_isnan)
    nan_k = kernel("nan")
    assert nan_k.isnan
    assert (lap_d + nan_k).isnan
    assert (np.nan * lap_s).isnan


def test_chunkerkerneval_greenlap_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerkerneval_greenlap
    chnkr, lap_s, lap_d, densu, densun, utarg = laplace_green_identity_quantities(fixture)

    opts = {"forceadap": True}
    Du_direct = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, opts).reshape(-1, order="F")
    Sun_direct = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, opts).reshape(-1, order="F")
    identity_direct = Sun_direct - Du_direct
    flam_opts = {"acceleration": "flam", "forceadap": True, "occ": 32, "rank_or_tol": 1.0e-8, "useproxy": False}
    Du_flam = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, flam_opts).reshape(-1, order="F")
    Sun_flam = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, flam_opts).reshape(-1, order="F")
    identity_flam = Sun_flam - Du_flam

    np.testing.assert_allclose(densu, fixture.densu, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(densun, fixture.densun, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, fixture.utarg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(Du_direct, fixture.Du_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun_direct, fixture.Sun_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity_direct, fixture.utarg_identity_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Du_fmm, fixture.Du_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Sun_fmm, fixture.Sun_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.utarg_identity_fmm, fixture.utarg_identity_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Du_flam, fixture.Du_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Sun_flam, fixture.Sun_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.utarg_identity_flam, fixture.utarg_identity_direct, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Du_flam, fixture.Du_direct, rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(Sun_flam, fixture.Sun_direct, rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(identity_flam, fixture.utarg_identity_direct, rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(Du_flam, fixture.Du_flam, rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(Sun_flam, fixture.Sun_flam, rtol=1e-8, atol=5e-9)
    assert np.linalg.norm(utarg - identity_direct) / np.linalg.norm(utarg) < 1e-11
    assert np.linalg.norm(utarg - identity_flam) / np.linalg.norm(utarg) < 1e-8
    assert float(fixture.relerr_direct) < 1e-11
    assert float(fixture.relerr_fmm) < 1e-11
    assert float(fixture.relerr_flam) < 1e-11
    assert not bool(fixture.flam_deferred)


def test_chunkerkernevalmat_greenlap_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerkernevalmat_greenlap
    chnkr, lap_s, lap_d, densu, densun, utarg = laplace_green_identity_quantities(fixture)

    opts = {"forceadap": True}
    Dmat = chunkerkernevalmat(chnkr, lap_d, fixture.targets, opts)
    Smat = chunkerkernevalmat(chnkr, lap_s, fixture.targets, opts)
    Du = Dmat @ densu
    Sun = Smat @ densun
    identity = Sun - Du

    np.testing.assert_allclose(densu, fixture.densu, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(densun, fixture.densun, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, fixture.utarg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(Dmat, fixture.Dmat_forceadap, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Smat, fixture.Smat, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Du, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(utarg - identity) / np.linalg.norm(utarg) < 1e-11
    assert float(fixture.relerr) < 1e-11
    assert float(fixture.relerr_forceadap) < 1e-11


def test_chunkerkerneval_greenhelm_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerkerneval_greenhelm
    chnkr, helm_s, helm_d, densu, densun, utarg = helmholtz_green_identity_quantities(fixture)

    opts = {"forceadap": True}
    Du = chunkerkerneval(chnkr, helm_d, densu, fixture.targets, opts).reshape(-1, order="F")
    Sun = chunkerkerneval(chnkr, helm_s, densun, fixture.targets, opts).reshape(-1, order="F")
    identity = Sun - Du

    np.testing.assert_allclose(densu, fixture.densu, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(densun, fixture.densun, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, fixture.utarg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(Du, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(utarg - identity) / np.linalg.norm(utarg) < 1e-11
    assert float(fixture.relerr) < 1e-11


def test_chunkermat_quadadap_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermat_quadadap
    chnkr = chunker_from_fields(fixture.chunker)
    kern = kernel("helm", "d", fixture.zk)

    ggq = chunkermat(chnkr, kern)
    adap = quadadap.buildmat(chnkr, kern, kern.opdims, {"sing": "log", "robust": False})

    np.testing.assert_allclose(ggq, fixture.mat_ggq, rtol=1e-9, atol=2e-9)
    np.testing.assert_allclose(adap, fixture.mat_adap, rtol=1e-9, atol=2e-9)
    assert float(fixture.relerr) < 1e-9
    assert np.linalg.norm(ggq - adap, "fro") / np.linalg.norm(ggq, "fro") < 1e-9
