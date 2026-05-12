import warnings

import numpy as np
import pytest
from scipy import sparse
from scipy.special import hankel1

from chunkie import (
    Chunker,
    Kernel,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerintegral,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    chunkerpoly,
    chunkgraph,
    chunkgraphinregion,
    find_edge_regions,
    kernel,
    lege,
    tochunkgraph,
)
from chunkie.chnk import elast2d, flam, helm1d, helm2d, lap2d
from chunkie.geometry import curves, flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself
from chunkie.numerics import arcparam, smoother, special
from chunkie.quadrature import adaptive as quadadap
from chunkie.operators import PointInfo, pointinfo
from _fixture_generation import chunker_from_fields, load_generated_mat_fixture, point_array, pointinfo_from_mat


def load_devtools_easy():
    return load_generated_mat_fixture("devtools_easy.mat", squeeze_me=True, struct_as_record=False)["devtools_easy"]


def _assert_kernder_algebra(case, coefs: np.ndarray, coefa: np.ndarray) -> None:
    s = _kernder_array(case.s)
    d = _kernder_array(case.d)
    c = _kernder_array(case.c)
    sp = _kernder_array(case.sp)
    dp = _kernder_array(case.dp)
    cp = _kernder_array(case.cp)
    sgrad = _kernder_array(case.sgrad)
    dgrad = _kernder_array(case.dgrad)
    cgrad = _kernder_array(case.cgrad)
    c2trans = _kernder_array(case.c2trans)

    np.testing.assert_allclose(coefs[0] * d + coefs[1] * s, c, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(coefs[0] * dp + coefs[1] * sp, cp, rtol=1e-12, atol=1e-13)

    all_mat = np.asarray(case.all)
    all_assembled = np.zeros_like(all_mat)
    all_assembled[0::2, 0::2] = coefa[0, 0] * d
    all_assembled[0::2, 1::2] = coefa[0, 1] * s
    all_assembled[1::2, 0::2] = coefa[1, 0] * dp
    all_assembled[1::2, 1::2] = coefa[1, 1] * sp
    np.testing.assert_allclose(all_assembled, all_mat, rtol=1e-12, atol=1e-13)

    np.testing.assert_allclose(np.hstack((coefs[0] * d, coefs[1] * s)), case.trans_rep, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.hstack((coefs[0] * dp, coefs[1] * sp)), case.trans_rep_prime, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(coefs[0] * dgrad + coefs[1] * sgrad, cgrad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.hstack((coefs[0] * dgrad, coefs[1] * sgrad)), case.trans_rep_grad, rtol=1e-12, atol=1e-13)

    c2_assembled = np.zeros_like(c2trans)
    c2_assembled[0::2] = coefs[0] * d + coefs[1] * s
    c2_assembled[1::2] = coefs[0] * dp + coefs[1] * sp
    np.testing.assert_allclose(c2_assembled, c2trans, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dp.reshape(-1, order="F"), np.asarray(case.dp_grad_dot).reshape(-1), rtol=1e-12, atol=1e-13)


def _kernder_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _assert_kernder_field(actual: np.ndarray, expected, label: str) -> None:
    expected_arr = np.asarray(expected)
    if expected_arr.shape != actual.shape:
        expected_arr = expected_arr.reshape(actual.shape, order="F")
    np.testing.assert_allclose(actual, expected_arr, rtol=1e-12, atol=1e-13, err_msg=label)


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


def transmission_all_kernel_from_fixture(fixture) -> Kernel:
    ks = np.asarray(fixture.ks).reshape(-1, order="F")
    cs = np.asarray(fixture.cs, dtype=int).reshape(2, -1, order="F")
    coefs = np.asarray(fixture.coefs).reshape(-1, order="F")
    d1 = int(cs[0, 0]) - 1
    d2 = int(cs[1, 0]) - 1
    c1 = coefs[d1]
    c2 = coefs[d2]
    alpha1 = 2.0 / (c1 + c2)
    alpha2 = 2.0 / (1.0 / c1 + 1.0 / c2)
    cc1 = np.array([[-alpha1 * c1, -alpha1], [alpha2, alpha2 / c1]], dtype=complex)
    cc2 = np.array([[-alpha1 * c2, -alpha1], [alpha2, alpha2 / c2]], dtype=complex)
    cc_use = np.stack((-cc1, -cc2), axis=2)
    return kernel("helmdiff", "all", [ks[d1], ks[d2]], cc_use)


def transmission_point_source_boundary_data(chnkr: Chunker, fixture) -> np.ndarray:
    ks = np.asarray(fixture.ks).reshape(-1, order="F")
    cs = np.asarray(fixture.cs, dtype=int).reshape(2, -1, order="F")
    coefs = np.asarray(fixture.coefs).reshape(-1, order="F")
    sources = np.asarray(fixture.sources)
    charges = np.asarray(fixture.charges).reshape(-1, order="F")
    d1 = int(cs[0, 0]) - 1
    d2 = int(cs[1, 0]) - 1
    c1 = coefs[d1]
    c2 = coefs[d2]
    alpha1 = 2.0 / (c1 + c2)
    alpha2 = 2.0 / (1.0 / c1 + 1.0 / c2)
    targ = pointinfo(chnkr)

    val1, grad1, _ = helm2d.green(ks[d1], sources[:, d1 : d1 + 1], targ.r)
    val2, grad2, _ = helm2d.green(ks[d2], sources[:, d2 : d2 + 1], targ.r)
    u1 = val1 @ charges[d1 : d1 + 1]
    u2 = val2 @ charges[d2 : d2 + 1]
    dudn1 = (grad1[:, :, 0] @ charges[d1 : d1 + 1]) * targ.n[0] + (grad1[:, :, 1] @ charges[d1 : d1 + 1]) * targ.n[1]
    dudn2 = (grad2[:, :, 0] @ charges[d2 : d2 + 1]) * targ.n[0] + (grad2[:, :, 1] @ charges[d2 : d2 + 1]) * targ.n[1]

    out = np.zeros(2 * chnkr.npt, dtype=complex)
    out[0::2] = alpha1 * (u1 - u2)
    out[1::2] = -alpha2 * (dudn1 / c1 - dudn2 / c2)
    return out


def kernel_interleave_helmholtz_kernels(fixture):
    zk = complex(fixture.zk)
    alpha = complex(fixture.alpha)
    c1 = complex(fixture.c1)
    c2 = complex(fixture.c2)
    c3 = complex(fixture.c3)
    sik = kernel("helm", "s", 1j * zk)
    sikp = kernel("helm", "sprime", 1j * zk)
    skp = kernel("helm", "sprime", zk)
    sk = kernel("helm", "s", zk)
    dk = kernel("helm", "d", zk)
    dkdiff = kernel("helmdiff", "dprime", [zk, 1j * zk])
    zero = kernel("zero")
    system = kernel(
        [
            [c1 * skp, c2 * dkdiff, c2 * sikp],
            [c3 * sik, zero, zero],
            [c3 * sikp, zero, zero],
        ]
    )
    eval_kernel = c1 * kernel([[sk, 1j * alpha * dk, zero]])
    return system, eval_kernel, skp, sk


def sorted_pairs(pairs: np.ndarray) -> np.ndarray:
    arr = np.asarray(pairs, dtype=int)
    if arr.size == 0:
        return arr.reshape(2, 0)
    order = np.lexsort((arr[1], arr[0]))
    return arr[:, order]


def sinearc(t, amp: float, frq: float):
    flat = np.asarray(t, dtype=float).reshape(-1)
    r = np.vstack((flat, amp * np.sin(frq * flat)))
    d = np.vstack((np.ones_like(flat), amp * frq * np.cos(frq * flat)))
    d2 = np.vstack((np.zeros_like(flat), -(frq**2) * amp * np.sin(flat)))
    return r, d, d2


def chunkermatapply_graph_from_fixture(fixture):
    edges = np.asarray(fixture.edgesendverts, dtype=int) - 1
    if hasattr(fixture, "echnks"):
        edge_chunks = [chunker_from_fields(edge) for edge in np.asarray(fixture.echnks).reshape(-1, order="F")]
        return chunkgraph(fixture.verts, edges, edge_chunks)
    edge_specs = [lambda t, amp=float(fixture.amp), frq=float(fixture.frq): sinearc(t, amp, frq) for _ in range(edges.shape[1])]
    return chunkgraph(fixture.verts, edges, edge_specs, {"nover": max(int(fixture.nover) - 1, 0)})


def loop_curve(t):
    flat = np.asarray(t, dtype=float).reshape(-1)
    r = np.vstack((np.cos(flat), np.sin(flat) * np.sin(0.5 * flat)))
    d = np.vstack(
        (
            -np.sin(flat),
            np.cos(flat) * np.sin(0.5 * flat) + 0.5 * np.sin(flat) * np.cos(0.5 * flat),
        )
    )
    d2 = np.vstack(
        (
            -np.cos(flat),
            -np.sin(flat) * np.sin(0.5 * flat)
            + np.cos(flat) * np.cos(0.5 * flat)
            - 0.25 * np.sin(flat) * np.sin(0.5 * flat),
        )
    )
    return r, d, d2


def chunkgraph_lastlength_graph(fixture):
    ncircedge = int(fixture.ncircedge)
    edge_specs = []
    for _ in range(ncircedge // 2):
        edge_specs.append(None)
        edge_specs.append(lambda t, amp=float(fixture.amp), frq=float(fixture.frq): sinearc(t, amp, frq))
    edge_specs.append(None)
    edge_specs.append(
        lambda t,
        narm=int(fixture.closed_narm),
        amp=float(fixture.closed_amp),
        ctr=np.asarray(fixture.closed_ctr, dtype=float).reshape(2),
        scale=float(fixture.closed_scale): curves.starfish(t, narm, amp, ctr, 0.0, scale)
    )
    cparams = [{"eps": 1e-8} for _ in edge_specs]
    cparams[-1].update({"ta": 0.0, "tb": 2 * np.pi})
    return chunkgraph(fixture.verts, np.asarray(fixture.edge2verts_with_closed, dtype=float), edge_specs, cparams)


def chunkgraph_region_graph(fixture):
    edge_specs = [None] * int(np.asarray(fixture.edgesendverts).shape[1])
    edge_specs[-1] = (
        lambda t,
        narm=int(fixture.narms),
        amp=float(fixture.amp),
        ctr=np.asarray(fixture.center, dtype=float).reshape(2),
        scale=float(fixture.scale): curves.starfish(t, narm, amp, ctr, 0.0, scale)
    )
    cparams = [{} for _ in edge_specs]
    cparams[-1].update({"ta": float(fixture.closed_ta), "tb": float(fixture.closed_tb)})
    edges = np.asarray(fixture.edgesendverts, dtype=float).copy()
    finite = np.isfinite(edges)
    edges[finite] -= 1
    return chunkgraph(fixture.verts, edges, edge_specs, cparams)


def packed_matlab_regions_to_python(fixture) -> list[list[list[int]]]:
    counts = np.asarray(fixture.region_loop_counts, dtype=int).reshape(-1)
    lens = np.asarray(fixture.region_loop_lens, dtype=int)
    loops = np.asarray(fixture.region_loops, dtype=int)
    regions: list[list[list[int]]] = []
    for ireg, loop_count in enumerate(counts):
        region: list[list[int]] = []
        for iloop in range(int(loop_count)):
            length = int(lens[iloop, ireg])
            matlab_loop = loops[:length, iloop, ireg].reshape(-1)
            region.append([int(edge) - 1 if int(edge) > 0 else int(edge) for edge in matlab_loop])
        regions.append(region)
    return regions


def chunkgraph_opdim_graph(fixture):
    return chunkgraph(fixture.verts, dense_int_array(fixture.edge2verts), None, {"maxchunklen": float(fixture.maxchunklen)})


def chunkgraph_opdim_kernels(fixture):
    zk0 = complex(fixture.zk0)
    zk1 = complex(fixture.zk1)
    coef = np.asarray(fixture.coef).reshape(-1, order="F")
    cc = np.asarray(fixture.cc)

    def fkern11(src, targ):
        return helm2d.kern(zk0, src, targ, "all", cc) - helm2d.kern(zk1, src, targ, "all", cc)

    def fkern12(src, targ):
        return helm2d.kern(zk0, src, targ, "c2trans", coef)

    def fkern21(src, targ):
        return helm2d.kern(zk0, src, targ, "trans_rep", [1, 1])

    def fkern22(src, targ):
        return helm2d.kern(zk0, src, targ, "c", [1, 1j])

    flags = np.asarray(fixture.trans_flag, dtype=int).reshape(-1)
    kernels = []
    for itarg in range(flags.size):
        row = []
        for isrc in range(flags.size):
            kind = (int(flags[itarg]), int(flags[isrc]))
            row.append(fkern11 if kind == (1, 1) else fkern12 if kind == (1, 0) else fkern21 if kind == (0, 1) else fkern22)
        kernels.append(row)
    return kernels


def graph_vertex_endpoint_arcs(cg, nverts: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    arcs = np.full((nverts, width), np.nan)
    degrees = np.zeros(nverts, dtype=int)
    for ivert in range(nverts):
        edges, signs = cg.vstruc[ivert]
        degrees[ivert] = edges.size
        for idx, (edge, sign) in enumerate(zip(edges, signs)):
            ichunk = 0 if sign < 0 else cg.echnks[int(edge)].nch - 1
            arcs[ivert, idx] = float(cg.echnks[int(edge)].chunklen([ichunk])[0])
    return arcs, degrees


def dense_int_array(value) -> np.ndarray:
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=int)


def assert_chunker_fields_match(chnkr: Chunker, fields, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(chnkr.r, fields.r, atol=atol)
    np.testing.assert_allclose(chnkr.d, fields.d, atol=atol)
    np.testing.assert_allclose(chnkr.d2, fields.d2, atol=atol)
    np.testing.assert_allclose(chnkr.n, fields.n, atol=atol)
    np.testing.assert_allclose(chnkr.wts, fields.wts, atol=atol)
    np.testing.assert_array_equal(chnkr.adj, np.asarray(fields.adj, dtype=int))
    np.testing.assert_allclose(chnkr.chunklen(), fields.chunklen, atol=atol)
    np.testing.assert_allclose(chnkr.area(), fields.area, atol=atol)


def assert_chunker_geometry_multiset_match(chnkr: Chunker, fields, atol: float = 1e-12) -> None:
    expected_r = np.asarray(fields.r)
    expected_d = np.asarray(fields.d)
    expected_d2 = np.asarray(fields.d2)
    expected_n = np.asarray(fields.n)
    expected_wts = np.asarray(fields.wts)
    assert chnkr.r.shape == expected_r.shape

    actual_centers = np.mean(chnkr.r, axis=1).T
    expected_centers = np.mean(expected_r, axis=1).T
    distances = np.linalg.norm(expected_centers[:, None, :] - actual_centers[None, :, :], axis=2)
    order = np.argmin(distances, axis=1)
    assert len(np.unique(order)) == chnkr.nch
    np.testing.assert_allclose(distances[np.arange(chnkr.nch), order], 0.0, atol=atol)

    for iexpected, iactual in enumerate(order):
        r_actual = chnkr.r[:, :, iactual]
        r_expected = expected_r[:, :, iexpected]
        if np.max(np.abs(r_actual - r_expected)) <= atol:
            np.testing.assert_allclose(chnkr.d[:, :, iactual], expected_d[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(chnkr.d2[:, :, iactual], expected_d2[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(chnkr.n[:, :, iactual], expected_n[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(chnkr.wts[:, iactual], expected_wts[:, iexpected], atol=atol)
        else:
            np.testing.assert_allclose(r_actual[:, ::-1], r_expected, atol=atol)
            np.testing.assert_allclose(-chnkr.d[:, ::-1, iactual], expected_d[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(chnkr.d2[:, ::-1, iactual], expected_d2[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(-chnkr.n[:, ::-1, iactual], expected_n[:, :, iexpected], atol=atol)
            np.testing.assert_allclose(chnkr.wts[::-1, iactual], expected_wts[:, iexpected], atol=atol)
    np.testing.assert_allclose(np.sort(chnkr.chunklen()), np.sort(np.asarray(fields.chunklen).reshape(-1)), atol=atol)
    np.testing.assert_allclose(chnkr.area(), fields.area, atol=atol)


def _elasticlet(lam: float, mu: float, src: PointInfo, targ: PointInfo, f: np.ndarray):
    return (
        elast2d.kern(lam, mu, src, targ, "s") @ f,
        elast2d.kern(lam, mu, src, targ, "strac") @ f,
        elast2d.kern(lam, mu, src, targ, "d") @ f,
        elast2d.kern(lam, mu, src, targ, "dalt") @ f,
        elast2d.kern(lam, mu, src, targ, "daltgrad") @ f,
        elast2d.kern(lam, mu, src, targ, "dalttrac") @ f,
        elast2d.kern(lam, mu, src, targ, "sgrad") @ f,
    )


def _shift_pointinfo(info: PointInfo, dx: float, dy: float) -> PointInfo:
    return PointInfo(
        r=info.r + np.array([[dx], [dy]]),
        d=None if info.d is None else info.d.copy(),
        n=None if info.n is None else info.n.copy(),
        d2=None if info.d2 is None else info.d2.copy(),
    )


def _elastic_finite_difference_errors(lam: float, mu: float, src: PointInfo, targ: PointInfo, f: np.ndarray, niter: int):
    u00, trac00, _, dalt00, dalt00grad, _, _ = _elasticlet(lam, mu, src, targ, f)
    pde_errs = np.zeros(niter)
    pdedalt_errs = np.zeros(niter)
    div_errs = np.zeros(niter)
    trac_errs = np.zeros(niter)
    daltgrad_errs = np.zeros(niter)

    for idx in range(niter):
        hh = 0.1 ** (idx + 1)
        u01, trac01, _, dalt01, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, 0.0, hh), f)
        u10, trac10, _, dalt10, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, hh, 0.0), f)
        u0m1, trac0m1, _, dalt0m1, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, 0.0, -hh), f)
        um10, tracm10, _, daltm10, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, -hh, 0.0), f)
        u11, _, _, dalt11, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, hh, hh), f)
        u1m1, _, _, dalt1m1, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, hh, -hh), f)
        um1m1, _, _, daltm1m1, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, -hh, -hh), f)
        um11, _, _, daltm11, _, _, _ = _elasticlet(lam, mu, src, _shift_pointinfo(targ, -hh, hh), f)

        lapu = (u01 + u10 + u0m1 + um10 - 4.0 * u00) / hh**2
        ux = (u10 - um10) / (2.0 * hh)
        uy = (u01 - u0m1) / (2.0 * hh)
        uxx = (u10 + um10 - 2.0 * u00) / hh**2
        uyy = (u01 + u0m1 - 2.0 * u00) / hh**2
        uxy = (u11 - um11 - u1m1 + um1m1) / (4.0 * hh**2)

        lapdalt = (dalt01 + dalt10 + dalt0m1 + daltm10 - 4.0 * dalt00) / hh**2
        daltx = (dalt10 - daltm10) / (2.0 * hh)
        dalty = (dalt01 - dalt0m1) / (2.0 * hh)
        daltxx = (dalt10 + daltm10 - 2.0 * dalt00) / hh**2
        daltyy = (dalt01 + dalt0m1 - 2.0 * dalt00) / hh**2
        daltxy = (dalt11 - daltm11 - dalt1m1 + daltm1m1) / (4.0 * hh**2)

        pdeuh = mu * lapu + (lam + mu) * np.array([uxx[0] + uxy[1], uxy[0] + uyy[1]])
        pdedalth = mu * lapdalt + (lam + mu) * np.array([daltxx[0] + daltxy[1], daltxy[0] + daltyy[1]])
        pde_errs[idx] = np.linalg.norm(pdeuh) / np.linalg.norm(u00)
        pdedalt_errs[idx] = np.linalg.norm(pdedalth) / np.linalg.norm(u00)
        div_errs[idx] = abs((trac10[0] - tracm10[0]) / (2.0 * hh) + (trac01[1] - trac0m1[1]) / (2.0 * hh)) / np.linalg.norm(trac00)

        jact = np.column_stack((ux, uy))
        epsmat = 0.5 * (jact + jact.T)
        normal = targ.n.reshape(2, 1)
        tracuh = (lam * (ux[0] + uy[1]) * np.eye(2) + 2.0 * mu * epsmat) @ normal
        trac_errs[idx] = np.linalg.norm(tracuh.reshape(-1) - trac00) / np.linalg.norm(trac00)
        daltgrad_errs[idx] = (
            np.linalg.norm(dalt00grad[0::2] - daltx) / np.linalg.norm(dalt00grad[0::2])
            + np.linalg.norm(dalt00grad[1::2] - dalty) / np.linalg.norm(dalt00grad[1::2])
        )
    return pde_errs, pdedalt_errs, div_errs, trac_errs, daltgrad_errs


def test_absconvgauss_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().absconvgauss

    val, der, der2 = special.absconvgauss(fixture.x, float(fixture.m), float(fixture.offset), float(fixture.h))

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

    resampled_mv, eps_mv = chnkr.arcresample({"mv_bdries": 1})
    np.testing.assert_allclose(eps_mv, fixture.resampled_mv_eps, rtol=1e-8, atol=1e-12)
    assert float(fixture.resampled_mv_area_err) < 1e-8
    assert float(fixture.resampled_mv_len_err) < 1e-8
    np.testing.assert_allclose(abs(resampled_mv.area() - chnkr.area()), 0.0, atol=1e-8)
    np.testing.assert_allclose(abs(np.sum(resampled_mv.wts) - np.sum(chnkr.wts)), 0.0, atol=1e-8)
    np.testing.assert_allclose(
        resampled_mv.arclengthdens() / (resampled_mv.chunklen() / 2)[None, :],
        1.0,
        atol=1e-6,
    )


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
    with pytest.warns(UserWarning, match="unit tangent vectors"):
        chunkerfunc(lambda t: np.vstack((np.cos(np.asarray(t).reshape(-1)), np.sin(np.asarray(t).reshape(-1) / 2))))
    with pytest.warns(UserWarning, match="start and end points"):
        chunkerfunc(
            lambda t: np.vstack((np.cos(np.asarray(t).reshape(-1)), np.sin(np.asarray(t).reshape(-1)))),
            {"ta": 0.0, "tb": 2 * np.pi - 1.0e-3},
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunkerfunc(
            lambda t: np.vstack((np.cos(np.asarray(t).reshape(-1)), np.sin(np.asarray(t).reshape(-1)))),
            {"ta": 0.0, "tb": 2 * np.pi - 1.0e-3, "ifclosed": False},
        )
    assert caught == []
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
    for actual, expected in ((closed, fixture.closed), (open_chnkr, fixture.open)):
        np.testing.assert_allclose(actual.r, expected.r, atol=1e-12)
        np.testing.assert_allclose(actual.d, expected.d, atol=1e-12)
        np.testing.assert_allclose(actual.d2, expected.d2, atol=1e-10)
        np.testing.assert_allclose(actual.n, expected.n, atol=1e-12)
        np.testing.assert_allclose(actual.wts, expected.wts, atol=1e-12)
        np.testing.assert_array_equal(actual.adj, np.asarray(expected.adj, dtype=int))
        np.testing.assert_allclose(actual.chunklen(), expected.chunklen, atol=1e-12)
        np.testing.assert_allclose(actual.area(), expected.area, atol=1e-12)


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


def test_chunkgrphconstruct_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkgrphconstruct
    edge_specs = [lambda t, amp=float(fixture.amp), frq=float(fixture.frq): sinearc(t, amp, frq) for _ in range(5)]

    legacy = chunkgraph(fixture.verts, dense_int_array(fixture.edge2verts), edge_specs)
    modern = chunkgraph(fixture.verts, dense_int_array(fixture.edgesendverts), edge_specs)

    np.testing.assert_allclose(legacy.verts, fixture.legacy_verts, atol=1e-14)
    np.testing.assert_allclose(modern.verts, fixture.new_verts, atol=1e-14)
    np.testing.assert_array_equal(legacy.v2emat, dense_int_array(fixture.legacy_v2emat))
    np.testing.assert_array_equal(modern.v2emat, dense_int_array(fixture.new_v2emat))
    np.testing.assert_array_equal(legacy.v2emat, modern.v2emat)
    np.testing.assert_array_equal(legacy.edgesendverts, dense_int_array(fixture.legacy_edgesendverts) - 1)
    np.testing.assert_array_equal(modern.edgesendverts, dense_int_array(fixture.new_edgesendverts) - 1)
    assert_chunker_fields_match(legacy.echnks[0], fixture.legacy_first_edge, atol=1e-10)
    assert_chunker_fields_match(modern.echnks[0], fixture.new_first_edge, atol=1e-10)
    np.testing.assert_allclose(legacy.echnks[0].r, modern.echnks[0].r, atol=1e-13)


def test_chunkgraph_basic_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkgraph_basic
    edge_specs = [lambda t: sinearc(t, 0.5, 6.0) for _ in range(5)]
    legacy = chunkgraph(fixture.pentagon_verts, dense_int_array(fixture.pentagon_edge2verts), edge_specs)
    modern = chunkgraph(fixture.pentagon_verts, dense_int_array(fixture.pentagon_endverts), edge_specs)
    np.testing.assert_array_equal(legacy.v2emat, dense_int_array(fixture.pentagon_legacy_v2emat))
    np.testing.assert_array_equal(modern.v2emat, dense_int_array(fixture.pentagon_new_v2emat))
    np.testing.assert_array_equal(legacy.v2emat, modern.v2emat)

    twoedge_legacy = chunkgraph(fixture.twoedge_verts, dense_int_array(fixture.twoedge_edge2verts))
    twoedge_modern = chunkgraph(fixture.twoedge_verts, dense_int_array(fixture.twoedge_endverts))
    np.testing.assert_array_equal(twoedge_legacy.v2emat, dense_int_array(fixture.twoedge_legacy_v2emat))
    np.testing.assert_array_equal(twoedge_modern.v2emat, dense_int_array(fixture.twoedge_new_v2emat))
    np.testing.assert_array_equal(twoedge_legacy.v2emat, twoedge_modern.v2emat)

    multi = chunkgraph(np.array([[1, 0, -1, 2, 0, -2], [0, 1, 0, -0.5, 2, -0.5]], dtype=float), np.array([[1, 2, 3, 4, 5, 6], [2, 3, 1, 5, 6, 4]]))
    bridge = chunkgraph(np.array([[1, 0, -1, 4, 3, 2], [0, 1, 0, 0, 1, 0]], dtype=float), np.array([[1, 2, 3, 4, 5, 6, 1], [3, 1, 2, 6, 4, 5, 6]]))
    loop = chunkgraph(np.array([[2.0], [1.0]]), np.array([[1], [1]]), [loop_curve], {"ta": 0.0, "tb": 2 * np.pi}, {"k": 12})
    nested = chunkgraph(np.array([[1, 0, -1, 2, 0, -2], [0, 1, 0, -1, 2, -1]], dtype=float), np.array([[1, 2, 3, 4, 5, 6], [2, 3, 1, 5, 6, 4]]))
    assert len(multi.regions) == int(fixture.multiconnected_region_count)
    assert len(bridge.regions) == int(fixture.bridge_region_count)
    assert len(loop.regions) == int(fixture.loop_region_count)
    assert len(nested.regions) == int(fixture.nested_region_count)

    adj = chunkgraph(fixture.adjtri_verts, dense_int_array(fixture.adjtri_edges))
    targets = np.asarray(fixture.adjtri_targets)
    ids = chunkgraphinregion(adj, targets)
    np.testing.assert_array_equal(ids, np.asarray(fixture.adjtri_ids, dtype=int).reshape(-1))
    np.testing.assert_array_equal(ids, np.asarray(fixture.adjtri_idstrue, dtype=int).reshape(-1))
    np.testing.assert_array_equal(
        chunkgraphinregion(adj, [fixture.x1, fixture.x1]).reshape(-1, order="F"),
        np.asarray(fixture.adjtri_ids_grid, dtype=int).reshape(-1, order="F"),
    )

    A = np.array([[3.0, 2.0], [1.0, 1.0]])
    v = np.array([[-1.0], [2.0]])
    affine_targets = A @ targets + v
    affine = A @ adj + v.reshape(2)
    np.testing.assert_array_equal(chunkgraphinregion(affine, affine_targets), np.asarray(fixture.adjtri_affine_ids, dtype=int).reshape(-1))
    scaled_targets = 2 * affine_targets
    scaled = affine * 2
    np.testing.assert_array_equal(chunkgraphinregion(scaled, scaled_targets), np.asarray(fixture.adjtri_scaled_ids, dtype=int).reshape(-1))
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_targets = rot @ scaled_targets
    rotated = scaled.rotate(theta)
    np.testing.assert_array_equal(chunkgraphinregion(rotated, rotated_targets), np.asarray(fixture.adjtri_rotated_ids, dtype=int).reshape(-1))
    reflected_targets = rotated_targets.copy()
    reflected_targets[0] *= -1
    reflected = rotated.reflect(np.pi / 2)
    np.testing.assert_array_equal(chunkgraphinregion(reflected, reflected_targets), np.asarray(fixture.adjtri_reflected_ids, dtype=int).reshape(-1))

    nested = chunkgraph(fixture.nested_verts, dense_int_array(fixture.nested_edges))
    nested_ids = chunkgraphinregion(nested, targets)
    np.testing.assert_array_equal(nested_ids, np.asarray(fixture.nested_ids, dtype=int).reshape(-1))
    np.testing.assert_array_equal(nested_ids, np.asarray(fixture.nested_idstrue, dtype=int).reshape(-1))

    refined = adj.refine({"nover": 1})
    np.testing.assert_array_equal([edge.nch for edge in adj.echnks], np.asarray(fixture.refine_nchs_before, dtype=int).reshape(-1))
    np.testing.assert_array_equal([edge.nch for edge in refined.echnks], np.asarray(fixture.refine_nchs_after, dtype=int).reshape(-1))
    np.testing.assert_array_equal(np.asarray(fixture.refine_nchs_after, dtype=int).reshape(-1), 2 * np.asarray(fixture.refine_nchs_before, dtype=int).reshape(-1))


def test_chunkgrphregion_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkgrphregion
    graph = chunkgraph_region_graph(fixture)
    expected_regions = packed_matlab_regions_to_python(fixture)

    assert len(graph.regions) == int(fixture.region_count)
    assert graph.regions == expected_regions
    np.testing.assert_array_equal(find_edge_regions(graph), np.asarray(fixture.edge_regions, dtype=int))
    assert int(np.min(fixture.edge_regions)) == 1
    assert int(np.max(fixture.edge_regions)) == len(expected_regions)


def test_chunkrgrph_opdim_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkrgrph_opdim
    graph = chunkgraph_opdim_graph(fixture)
    kernels = chunkgraph_opdim_kernels(fixture)
    mat = chunkermat(graph, kernels, {"nonsmoothonly": False, "rcip": True})

    np.testing.assert_array_equal([edge.npt for edge in graph.echnks], np.asarray(fixture.edge_npts, dtype=int).reshape(-1))
    np.testing.assert_array_equal(mat.shape, np.asarray(fixture.sysmat_shape, dtype=int).reshape(-1))
    row_starts = np.asarray(fixture.row_starts, dtype=int).reshape(-1) - 1
    col_starts = np.asarray(fixture.col_starts, dtype=int).reshape(-1) - 1
    np.testing.assert_allclose(
        mat[row_starts[0] : row_starts[1], col_starts[4] : col_starts[5]],
        np.asarray(fixture.block_1_5),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mat[row_starts[4] : row_starts[5], col_starts[0] : col_starts[1]],
        np.asarray(fixture.block_5_1),
        rtol=1e-11,
        atol=1e-12,
    )


def test_chunkgraph_lastlength_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkgraph_lastlength
    graph = chunkgraph_lastlength_graph(fixture)
    np.testing.assert_array_equal([edge.nch for edge in graph.echnks], np.asarray(fixture.initial_nchs, dtype=int).reshape(-1))
    assert np.isnan(np.asarray(fixture.edge2verts_with_closed, dtype=float)[:, -1]).all()
    assert graph.edgesendverts[0, -1] == graph.edgesendverts[1, -1]

    dlist_refined = graph.refine({"nover": 1, "dlist": [1]})
    np.testing.assert_array_equal([edge.nch for edge in dlist_refined.echnks], np.asarray(fixture.dlist_nchs, dtype=int).reshape(-1))

    splitchunks = [[] for _ in graph.echnks]
    splitchunks[2] = [2]
    split_refined = graph.refine({"splitchunks": splitchunks})
    np.testing.assert_array_equal([edge.nch for edge in split_refined.echnks], np.asarray(fixture.splitchunks_nchs, dtype=int).reshape(-1))

    last_len = float(fixture.last_len)
    last_refined = graph.refine({"last_len": last_len})
    arcs, degrees = graph_vertex_endpoint_arcs(
        last_refined,
        int(fixture.ncircedge),
        np.asarray(fixture.last_len_arcs).shape[1],
    )
    np.testing.assert_array_equal(degrees, np.asarray(fixture.last_len_degrees, dtype=int).reshape(-1))
    np.testing.assert_allclose(arcs, np.asarray(fixture.last_len_arcs, dtype=float), atol=5e-11)
    for row, degree in zip(arcs, degrees):
        active = row[:degree]
        np.testing.assert_allclose(active, active[0], atol=5e-11)
        level = np.log2(active[0] / last_len)
        assert abs(level - round(level)) < 1e-10
        assert round(level) <= 0


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
    assert np.isfinite(full).all()
    assert np.isfinite(inner_mat).all()
    np.testing.assert_allclose(full[np.ix_(idslce, idslce)], inner_mat, rtol=1e-12, atol=1e-13)
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
        {"rounded": True, "widths": 0.1 * np.ones(nverts), "eps": 1.0e-8},
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
    np.testing.assert_allclose(truepoly.area(), float(fixture.barb_area), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(truepoly.wts), float(fixture.barb_length), rtol=1e-12, atol=1e-12)
    assert_chunker_geometry_multiset_match(truepoly, fixture.truepoly, atol=1e-11)
    assert truepoly.datadim == np.asarray(fixture.edgevals).shape[0]
    assert rounded.datadim == np.asarray(fixture.edgevals).shape[0]
    assert rounded.nch == 2 * nverts
    assert open_chnkr.nch == np.asarray(fixture.open_verts).shape[1] - 1
    np.testing.assert_array_equal(open_chnkr.adj[:, 0], [-1, 2])
    np.testing.assert_array_equal(open_chnkr.adj[:, -1], [open_chnkr.nch - 1, -1])
    assert np.all(rounded.chunklen() > 0.0)
    assert np.all(open_chnkr.chunklen() > 0.0)


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
    np.testing.assert_array_equal(err_by_pt, np.zeros(chnkr.npt))
    assert chnkr.nch == 2 * int(fixture.nv)
    assert chnkr.checkadjinfo() == 0
    np.testing.assert_allclose(np.linalg.norm(chnkr.n, axis=0), 1.0, atol=1e-14)
    assert np.all(chnkr.chunklen() > 0.0)


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


def test_helm1d_green_devtools_direct_outputs_match_matlab():
    fixture = load_devtools_easy().helm1d_green
    chnkr = chunker_from_fields(fixture.chunker)
    kh = complex(fixture.kh)
    source = np.asarray(fixture.source).reshape(-1)
    xs = chnkr.r[0].reshape(-1, order="F")
    ys = chnkr.r[1].reshape(-1, order="F")
    rr = np.sqrt((xs - source[0]) ** 2 + (ys - source[1]) ** 2)
    u_test = 0.25j * hankel1(0, kh * rr)
    uu = -2.0 * float(fixture.m) * u_test[int(fixture.istart) - 1 : int(fixture.iend)]

    np.testing.assert_allclose(chnkr.r, fixture.chunker.r, atol=1e-13)
    np.testing.assert_allclose(u_test, np.asarray(fixture.u_test).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(uu, np.asarray(fixture.uu).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    assert int(fixture.nchpad) > 0
    assert int(fixture.istart) < int(fixture.iend)

    src = pointinfo_from_mat(fixture.src)
    targ = pointinfo_from_mat(fixture.targ)
    val, grad, hess = helm1d.green(float(fixture.E), src.r, targ.r)
    np.testing.assert_allclose(val, fixture.green_val, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(grad, fixture.green_grad, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(hess, fixture.green_hess, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "s"), fixture.kern_s, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "d"), fixture.kern_d, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "sp"), fixture.kern_sp, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "dp"), fixture.kern_dp, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "c2trans"), fixture.kern_c2trans, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(helm1d.kern(float(fixture.E), src, targ, "all", np.eye(2)), fixture.kern_all, rtol=1e-13, atol=1e-13)

    sweep = helm1d.sweep(
        np.asarray(fixture.uu).reshape(-1, order="F"),
        np.asarray(fixture.inds, dtype=int).reshape(-1, order="F") - 1,
        np.asarray(fixture.ts).reshape(-1, order="F"),
        np.asarray(fixture.wts).reshape(-1, order="F"),
        float(fixture.E),
    )
    np.testing.assert_allclose(sweep, np.asarray(fixture.sweep_uu).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)


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


def test_kernderinterleave_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().kernderinterleave

    helm = fixture.helm
    src = pointinfo_from_mat(helm.src)
    targ = pointinfo_from_mat(helm.targ)
    coefs = np.asarray(helm.coefs).reshape(-1, order="F")
    coefa = np.asarray(helm.coefa)
    zk = complex(helm.zk)
    helm_checks = {
        "s": kernel("helm", "s", zk)(src, targ),
        "d": kernel("helm", "d", zk)(src, targ),
        "c": kernel("helm", "c", zk, coefs)(src, targ),
        "sp": kernel("helm", "sp", zk)(src, targ),
        "dp": kernel("helm", "dp", zk)(src, targ),
        "cp": kernel("helm", "cp", zk, coefs)(src, targ),
        "all": kernel("helm", "all", zk, coefa)(src, targ),
        "trans_rep": kernel("helm", "trans_rep", zk, coefs)(src, targ),
        "trans_rep_prime": kernel("helm", "trans_rep_prime", zk, coefs)(src, targ),
        "c2trans": kernel("helm", "c2trans", zk, coefs)(src, targ),
        "sgrad": helm2d.kern(zk, src, targ, "sgrad"),
        "dgrad": helm2d.kern(zk, src, targ, "dgrad"),
        "cgrad": helm2d.kern(zk, src, targ, "cgrad", coefs),
        "trans_rep_grad": helm2d.kern(zk, src, targ, "trans_rep_grad", coefs),
    }
    for name, actual in helm_checks.items():
        _assert_kernder_field(actual, getattr(helm, name), f"helm {name}")
    _assert_kernder_algebra(helm, coefs, coefa)

    hdiff = fixture.hdiff
    hsrc = pointinfo_from_mat(hdiff.src)
    htarg = pointinfo_from_mat(hdiff.targ)
    zks = np.asarray(hdiff.zks).reshape(-1)
    coefs_diff = np.asarray(hdiff.coefs_diff)
    coefa_diff = np.asarray(hdiff.coefa_diff)
    coefb_diff = np.asarray(hdiff.coefb_diff)
    hdiff_checks = {
        "s": kernel("helmdiff", "s", zks)(hsrc, htarg),
        "d": kernel("helmdiff", "d", zks)(hsrc, htarg),
        "c": kernel("helmdiff", "c", zks, coefs_diff)(hsrc, htarg),
        "sp": kernel("helmdiff", "sp", zks)(hsrc, htarg),
        "dp": kernel("helmdiff", "dp", zks)(hsrc, htarg),
        "cp": kernel("helmdiff", "cp", zks, coefs_diff)(hsrc, htarg),
        "all": kernel("helmdiff", "all", zks, coefa_diff)(hsrc, htarg),
        "trans_rep": kernel("helmdiff", "trans_rep", zks, coefs_diff)(hsrc, htarg),
        "trans_rep_prime": kernel("helmdiff", "trans_rep_prime", zks, coefs_diff)(hsrc, htarg),
        "c2trans": kernel("helmdiff", "c2trans", zks, coefb_diff)(hsrc, htarg),
        "sgrad": helm2d.kern(zks[0], hsrc, htarg, "sgrad_diff") - helm2d.kern(zks[1], hsrc, htarg, "sgrad_diff"),
        "dgrad": helm2d.kern(zks[0], hsrc, htarg, "dgrad_diff") - helm2d.kern(zks[1], hsrc, htarg, "dgrad_diff"),
        "cgrad": helm2d.kern(zks[0], hsrc, htarg, "cgrad_diff", coefs_diff) - helm2d.kern(zks[1], hsrc, htarg, "cgrad_diff", coefs_diff),
        "trans_rep_grad": helm2d.kern(zks[0], hsrc, htarg, "trans_rep_grad_diff", coefb_diff[:, :, 0])
        - helm2d.kern(zks[1], hsrc, htarg, "trans_rep_grad_diff", coefb_diff[:, :, 1]),
    }
    for name, actual in hdiff_checks.items():
        _assert_kernder_field(actual, getattr(hdiff, name), f"helmdiff {name}")
    _assert_kernder_algebra(hdiff, coefs, coefa)

    lap = fixture.lap
    lsrc = pointinfo_from_mat(lap.src)
    ltarg = pointinfo_from_mat(lap.targ)
    lap_coefs = np.asarray(lap.coefs).reshape(-1, order="F")
    lap_checks = {
        "s": kernel("lap", "s")(lsrc, ltarg),
        "d": kernel("lap", "d")(lsrc, ltarg),
        "c": kernel("lap", "c", lap_coefs)(lsrc, ltarg),
        "sp": kernel("lap", "sp")(lsrc, ltarg),
        "dp": kernel("lap", "dp")(lsrc, ltarg),
        "cp": kernel("lap", "cp", lap_coefs)(lsrc, ltarg),
        "sgrad": lap2d.kern(lsrc, ltarg, "sgrad"),
        "dgrad": lap2d.kern(lsrc, ltarg, "dgrad"),
        "cgrad": lap2d.kern(lsrc, ltarg, "cgrad", lap_coefs),
    }
    for name, actual in lap_checks.items():
        _assert_kernder_field(actual, getattr(lap, name), f"lap {name}")
    lap_s = _kernder_array(lap.s)
    lap_d = _kernder_array(lap.d)
    lap_c = _kernder_array(lap.c)
    lap_sp = _kernder_array(lap.sp)
    lap_dp = _kernder_array(lap.dp)
    lap_cp = _kernder_array(lap.cp)
    lap_sgrad = _kernder_array(lap.sgrad)
    lap_dgrad = _kernder_array(lap.dgrad)
    lap_cgrad = _kernder_array(lap.cgrad)
    np.testing.assert_allclose(lap_coefs[0] * lap_d + lap_coefs[1] * lap_s, lap_c, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lap_coefs[0] * lap_dp + lap_coefs[1] * lap_sp, lap_cp, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lap_coefs[0] * lap_dgrad + lap_coefs[1] * lap_sgrad, lap_cgrad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lap_dp.reshape(-1, order="F"), np.asarray(lap.dp_grad_dot).reshape(-1), rtol=1e-12, atol=1e-13)


def test_kernel_interleave_devtools_dense_solve_matches_matlab():
    fixture = load_devtools_easy().kernel_interleave
    assert bool(np.asarray(fixture.invalid_didfail).item())
    with pytest.raises(ValueError, match="nan kernels"):
        kernel([[kernel("lap", "d"), kernel("nan")]])

    chnkr = chunker_from_fields(fixture.chunker)
    system, eval_kernel, skp, sk = kernel_interleave_helmholtz_kernels(fixture)
    np.testing.assert_array_equal(np.asarray(fixture.K_opdims, dtype=int).reshape(-1, order="F"), np.asarray(system.opdims))

    src = PointInfo(r=point_array(fixture.sources))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    weights = chnkr.wts.reshape(-1, order="F")
    sqrt_weights = np.sqrt(weights)
    rowdim = system.opdims[0]
    nsys = rowdim * chnkr.npt

    ubdry = skp(src, pointinfo(chnkr)) @ strengths
    rhs = np.zeros(nsys, dtype=complex)
    rhs[0::rowdim] = ubdry * sqrt_weights
    sys = np.asarray(chunkermat(chnkr, system, {"l2scale": bool(fixture.l2scale)})) + np.eye(nsys)
    sol_scaled = np.linalg.solve(sys, rhs)
    sol = sol_scaled / np.repeat(sqrt_weights, rowdim)
    utarg = sk(src, PointInfo(r=targets)) @ strengths
    dsol = chunkerkerneval(chnkr, eval_kernel, sol, targets, {"forceadap": True}).reshape(-1, order="F")
    relerr = np.linalg.norm(utarg - dsol) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))
    relerr2 = np.linalg.norm(utarg - dsol, ord=np.inf) / np.dot(np.abs(sol), np.repeat(weights, rowdim))

    rows = np.asarray(fixture.entry_rows, dtype=int).reshape(-1, order="F") - 1
    cols = np.asarray(fixture.entry_cols, dtype=int).reshape(-1, order="F") - 1
    probe = np.asarray(fixture.probe)

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(rhs, np.asarray(fixture.rhs).reshape(-1, order="F"), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(sys[np.ix_(rows, cols)], np.asarray(fixture.sys_entries), rtol=5e-8, atol=5e-8)
    np.testing.assert_allclose(sys @ probe, np.asarray(fixture.sys_probe), rtol=5e-8, atol=1e-7)
    np.testing.assert_allclose(sol_scaled, np.asarray(fixture.sol_backslash_scaled).reshape(-1, order="F"), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol).reshape(-1, order="F"), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(dsol, np.asarray(fixture.Dsol).reshape(-1, order="F"), rtol=2e-7, atol=5e-8)
    np.testing.assert_allclose(relerr, float(fixture.relerr), rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(relerr2, float(fixture.relerr2), rtol=2e-7, atol=1e-12)
    assert max(relerr, float(fixture.relerr)) < 1e-4
    assert max(relerr2, float(fixture.relerr2)) < 1e-3
    assert int(fixture.gmres_flag) == 0
    assert float(fixture.gmres_solve_relerr) < 1e-11


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


def test_chunkermat_stok2d_devtools_solve_matches_matlab():
    fixture = load_devtools_easy().chunkermat_stok2d
    chnkr = chunker_from_fields(fixture.chunker)
    mu = float(fixture.mu)
    coefs = np.asarray(fixture.coefs).reshape(-1, order="F")
    sources = PointInfo(r=point_array(fixture.sources), n=point_array(fixture.sources_n))
    targets = PointInfo(r=point_array(fixture.targets), n=point_array(fixture.targets_n))
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    matlab_sol = np.asarray(fixture.sol).reshape(-1, order="F")
    stok_d = kernel("stok", "d", mu)
    cvel = kernel("stok", "cvel", mu, coefs)

    ubdry = stok_d(sources, pointinfo(chnkr)) @ strengths
    utarg = stok_d(sources, targets) @ strengths
    D = chunkermat(chnkr, cvel)
    sys = -0.5 * np.eye(D.shape[0]) + D + chnkr.normonesmat() / np.sum(chnkr.wts)
    rhs = ubdry.reshape(-1, order="F")
    sol = np.linalg.solve(sys, rhs)
    Dsol = chunkerkerneval(chnkr, cvel, matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(D, np.asarray(fixture.D), rtol=5e-8, atol=5e-9)
    np.testing.assert_allclose(sys, np.asarray(fixture.sys), rtol=5e-8, atol=5e-9)
    np.testing.assert_allclose(sol, matlab_sol, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(Dsol, np.asarray(fixture.Dsol).reshape(-1, order="F"), rtol=5e-10, atol=5e-11)
    assert np.linalg.norm(sys @ sol - rhs) / np.linalg.norm(rhs) < 5e-13
    assert np.linalg.norm(sys @ matlab_sol - rhs) / np.linalg.norm(rhs) < 5e-9
    assert np.linalg.norm(utarg - Dsol) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg)) < 2e-10

    svel = kernel("stok", "svel", mu)
    Ssol = chunkerkerneval(chnkr, svel, matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")
    Ssys = chunkerkernevalmat(chnkr, svel, targets)
    np.testing.assert_allclose(Ssys, np.asarray(fixture.Ssys), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(Ssol, np.asarray(fixture.Ssol).reshape(-1, order="F"), rtol=5e-10, atol=5e-11)
    assert np.linalg.norm(Ssol - Ssys @ matlab_sol) < 2e-10

    for selector, field, rtol, atol in [
        ("dvel", "Dvel", 5e-10, 5e-11),
        ("strac", "Strac", 2e-9, 5e-10),
        ("dtrac", "Dtrac", 5e-8, 2e-8),
        ("spres", "Spres", 5e-10, 5e-11),
        ("dpres", "Dpres", 5e-8, 2e-8),
    ]:
        vals = chunkerkerneval(chnkr, kernel("stok", selector, mu), matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")
        np.testing.assert_allclose(vals, np.asarray(getattr(fixture, field)).reshape(-1, order="F"), rtol=rtol, atol=atol)

    pressure = chunkerkerneval(chnkr, kernel("stok", "cpres", mu, coefs), matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")
    pressure = pressure - pressure[0]
    grad = chunkerkerneval(chnkr, kernel("stok", "cgrad", mu, coefs), matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")
    np.testing.assert_allclose(pressure, np.asarray(fixture.pressure_direct).reshape(-1, order="F"), rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(grad, np.asarray(fixture.grad_direct).reshape(-1, order="F"), rtol=5e-9, atol=5e-10)
    assert np.linalg.norm(pressure - np.asarray(fixture.pressure_exact).reshape(-1, order="F")) / np.linalg.norm(fixture.pressure_exact) < 2e-10
    assert np.linalg.norm(grad - np.asarray(fixture.grad_exact).reshape(-1, order="F")) / np.linalg.norm(fixture.grad_exact) < 2e-10


def test_chunkermat_stok_traction_devtools_solve_matches_matlab():
    fixture = load_devtools_easy().chunkermat_stok_traction
    chnkr = chunker_from_fields(fixture.chunker)
    mu = float(fixture.mu)
    coefs = np.asarray(fixture.coefs).reshape(-1, order="F")
    sources = PointInfo(r=point_array(fixture.sources), n=point_array(fixture.sources_n))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    matlab_sol = np.asarray(fixture.sol).reshape(-1, order="F")
    boundary = pointinfo(chnkr)
    dtrac = kernel("stok", "dtrac", mu)
    dvel = kernel("stok", "dvel", mu)
    strac = kernel("stok", "strac", mu, coefs)
    svel = kernel("stok", "svel", mu)

    ubdry = dtrac(sources, boundary) @ strengths
    utarg = dvel(sources, PointInfo(r=targets)) @ strengths
    D = chunkermat(chnkr, strac)
    sys = 0.5 * np.eye(D.shape[0]) + D
    rhs = ubdry.reshape(-1, order="F")
    sol = np.linalg.solve(sys, rhs)
    Dsol = chunkerkerneval(chnkr, svel, matlab_sol, targets, {"acceleration": "fmm", "eps": 1e-11}).reshape(-1, order="F")

    tperp = np.vstack((targets[1], -targets[0])).reshape(-1, order="F")
    nt = targets.shape[1]
    fit = np.column_stack((np.tile(np.eye(2), (nt, 1)), tperp))
    rbd, *_ = np.linalg.lstsq(fit, utarg - Dsol, rcond=None)
    Dsol_corrected = Dsol.reshape(2, nt, order="F").copy()
    Dsol_corrected[0] += rbd[0] + rbd[2] * targets[1]
    Dsol_corrected[1] += rbd[1] - rbd[2] * targets[0]
    relerr = np.linalg.norm(utarg - Dsol_corrected.reshape(-1, order="F")) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=2e-9, atol=2e-8)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(D, np.asarray(fixture.D), rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(sys, np.asarray(fixture.sys), rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(Dsol, np.asarray(fixture.Dsol).reshape(-1, order="F"), rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(rbd, np.asarray(fixture.rbd).reshape(-1, order="F"), rtol=5e-9, atol=5e-10)
    np.testing.assert_allclose(
        Dsol_corrected,
        np.asarray(fixture.Dsol_corrected).reshape(2, -1, order="F"),
        rtol=5e-10,
        atol=5e-11,
    )
    assert np.linalg.norm(sys @ sol - rhs) / np.linalg.norm(rhs) < 1e-12
    assert np.linalg.norm(sys @ matlab_sol - rhs) / np.linalg.norm(rhs) < 1e-10
    assert relerr < 2e-10
    assert float(fixture.relerr) < 2e-10


def test_elastickernels_devtools_direct_diagnostics_match_matlab():
    fixture = load_devtools_easy().elastickernels
    chnkr = chunker_from_fields(fixture.chunker)
    lam = float(fixture.lam)
    mu = float(fixture.mu)
    strengths = np.asarray(fixture.f).reshape(-1, order="F")
    src = pointinfo_from_mat(fixture.src)
    targ = pointinfo_from_mat(fixture.targ)
    diag = fixture.diagnostics

    boundary = pointinfo(chnkr)
    u, trac, *_ = _elasticlet(lam, mu, src, boundary, strengths)
    utarg, tractarg, *_ = _elasticlet(lam, mu, src, targ, strengths)
    wts2 = np.repeat(chnkr.wts.reshape(-1, order="F"), 2)
    uint = elast2d.kern(lam, mu, boundary, targ, "s") @ (wts2 * trac)
    uint -= elast2d.kern(lam, mu, boundary, targ, "d") @ (wts2 * u)
    gid_err = np.linalg.norm(-np.ones_like(utarg) - uint / utarg)

    np.testing.assert_allclose(u, np.asarray(diag.boundary_u).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(trac, np.asarray(diag.boundary_trac).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(utarg, np.asarray(diag.target_u).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tractarg, np.asarray(diag.target_trac).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(uint, np.asarray(diag.green_uint).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gid_err, float(fixture.gid_err), rtol=1e-12, atol=1e-13)
    assert gid_err < 1e-14

    sample_src = pointinfo_from_mat(diag.sample_src)
    sample_targ = pointinfo_from_mat(diag.sample_targ)
    source_indices = np.asarray(diag.source_indices, dtype=int).reshape(-1) - 1
    target_index = int(np.asarray(diag.target_index, dtype=int).reshape(-1)[0]) - 1
    np.testing.assert_allclose(sample_src.r, boundary.r[:, source_indices], atol=1e-14)
    np.testing.assert_allclose(sample_targ.r, boundary.r[:, [target_index]], atol=1e-14)
    sample_values = _elasticlet(lam, mu, sample_src, sample_targ, strengths)
    for actual, expected in zip(
        sample_values,
        (
            diag.sample_u,
            diag.sample_trac,
            diag.sample_double,
            diag.sample_dalt,
            diag.sample_daltgrad,
            diag.sample_dalttrac,
            diag.sample_sgrad,
        ),
    ):
        np.testing.assert_allclose(actual, np.asarray(expected).reshape(-1, order="F"), rtol=1e-12, atol=1e-12)

    errors = _elastic_finite_difference_errors(lam, mu, sample_src, sample_targ, strengths, int(fixture.niter))
    for actual, expected, rtol, atol in zip(
        errors,
        (fixture.pde_errs, fixture.pdedalt_errs, fixture.div_errs, fixture.trac_errs, fixture.daltgrad_errs),
        (6e-2, 2e-2, 1e-4, 3e-2, 1e-2),
        (1e-10, 1e-8, 1e-12, 1e-12, 1e-12),
    ):
        np.testing.assert_allclose(actual[:-1], np.asarray(expected).reshape(-1, order="F")[:-1], rtol=rtol, atol=atol)
    assert np.min(errors[0]) < 1e-5
    assert np.min(errors[1]) < 1e-5
    assert np.min(errors[2]) < 1e-7
    assert np.min(errors[3]) < 1e-7
    assert np.min(errors[4]) < 1e-7


def test_kernelclass_devtools_green_identity_matches_matlab():
    fixture = load_devtools_easy().kernelclass
    chnkr, lap_s, lap_d, densu, densun, utarg = laplace_green_identity_quantities(fixture)

    opts = {"forceadap": True}
    Du = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, opts).reshape(-1, order="F")
    Sun = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, opts).reshape(-1, order="F")
    identity = Sun - Du
    fmm_opts = {"acceleration": "fmm", "forceadap": True, "eps": 1.0e-12}
    Du_fmm = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, fmm_opts).reshape(-1, order="F")
    Sun_fmm = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, fmm_opts).reshape(-1, order="F")
    identity_fmm = Sun_fmm - Du_fmm

    np.testing.assert_allclose(densu, fixture.ubdry, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(densun, fixture.unbdry, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, fixture.utarg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(Du, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Du_fmm, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun_fmm, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity_fmm, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Du_fmm, fixture.Du, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.Sun_fmm, fixture.Sun, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fixture.utarg_identity_fmm, fixture.utarg_identity, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(utarg - identity) / np.linalg.norm(utarg) < 1e-11
    assert np.linalg.norm(utarg - identity_fmm) / np.linalg.norm(utarg) < 1e-11
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
    fmm_opts = {"acceleration": "fmm", "forceadap": True, "eps": 1.0e-12}
    Du_fmm = chunkerkerneval(chnkr, lap_d, densu, fixture.targets, fmm_opts).reshape(-1, order="F")
    Sun_fmm = chunkerkerneval(chnkr, lap_s, densun, fixture.targets, fmm_opts).reshape(-1, order="F")
    identity_fmm = Sun_fmm - Du_fmm
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
    np.testing.assert_allclose(Du_fmm, fixture.Du_fmm, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Sun_fmm, fixture.Sun_fmm, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(identity_fmm, fixture.utarg_identity_fmm, rtol=1e-10, atol=1e-12)
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
    assert np.linalg.norm(utarg - identity_fmm) / np.linalg.norm(utarg) < 1e-11
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


def test_chunkerkerneval_corrections_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerkerneval_corrections
    chnkr = chunker_from_fields(fixture.chunker)
    srcinfo = PointInfo(r=point_array(fixture.sources))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    zk = complex(fixture.zk)
    helm_s = kernel("helm", "s", zk)
    helm_d = kernel("helm", "d", zk)

    rhs = helm_s(srcinfo, pointinfo(chnkr)) @ strengths
    sys = -0.5 * np.eye(chnkr.npt) + chunkermat(chnkr, helm_d)
    sol = np.linalg.solve(sys, rhs)
    utrue = helm_s(srcinfo, PointInfo(r=targets)) @ strengths
    cormat = chunkerkernevalmat(chnkr, helm_d, targets, {"corrections": True})
    assert sparse.issparse(cormat)
    u_eval_cor = chunkerkerneval(chnkr, helm_d, sol, targets, {"forcesmooth": True, "cormat": cormat}).reshape(-1, order="F")
    u_eval = chunkerkerneval(chnkr, helm_d, sol, targets, {"forcesmooth": True}).reshape(-1, order="F")

    np.testing.assert_allclose(rhs, np.asarray(fixture.rhs).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(sys, np.asarray(fixture.sys), rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol).reshape(-1, order="F"), rtol=1e-8, atol=2e-9)
    np.testing.assert_allclose(utrue, np.asarray(fixture.utrue).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    fixture_cormat = fixture.cormat.toarray() if sparse.issparse(fixture.cormat) else np.asarray(fixture.cormat)
    cormat_dense = cormat.toarray() if sparse.issparse(cormat) else np.asarray(cormat)
    np.testing.assert_allclose(cormat_dense, fixture_cormat, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(u_eval_cor, np.asarray(fixture.u_eval_cor).reshape(-1, order="F"), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(u_eval, np.asarray(fixture.u_eval).reshape(-1, order="F"), rtol=1e-9, atol=1e-11)
    assert np.linalg.norm(utrue - u_eval_cor, ord=np.inf) < 1e-11
    assert np.linalg.norm(utrue - u_eval, ord=np.inf) > 1e-10
    assert float(fixture.err_cor) < 1e-11
    assert float(fixture.err_smooth) > 1e-10


def test_chunkerkerneval_gaussid_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkerkerneval_gaussid
    chnkr = chunker_from_fields(fixture.chunker)
    lap_d = kernel("lap", "d")
    dens = np.asarray(fixture.density)
    targets = point_array(fixture.targets)

    values = chunkerkerneval(chnkr, lap_d, dens, targets, {"forceadap": True, "fac": 1.0}).reshape(-1, order="F")
    expected = np.asarray(fixture.values).reshape(-1, order="F")
    identity_err = np.minimum(np.abs(values), np.abs(values + 1.0))
    expected_inside = np.asarray(fixture.inside, dtype=bool).reshape(-1)
    interior = chunkerinterior(chnkr, targets, {"acceleration": "dense"})

    np.testing.assert_allclose(values, expected, rtol=1e-8, atol=5e-8)
    np.testing.assert_allclose(identity_err, np.asarray(fixture.identity_err).reshape(-1, order="F"), rtol=1e-7, atol=5e-8)
    np.testing.assert_array_equal(values < -0.5, expected_inside)
    np.testing.assert_array_equal(interior, expected_inside)
    assert identity_err.max() < 1e-6
    assert float(fixture.max_identity_err) < 1e-6


def test_adapgausswts_devtools_neighbor_block_matches_matlab():
    fixture = load_devtools_easy().adapgausswts
    chnkr = chunker_from_fields(fixture.chunker)
    kern = kernel("helm", "d", complex(np.asarray(fixture.zk).reshape(-1)[0]))
    source_chunk = int(fixture.source_chunk) - 1
    target_chunk = int(fixture.target_chunk) - 1
    targ = PointInfo(
        r=chnkr.r[:, :, target_chunk],
        d=chnkr.d[:, :, target_chunk],
        n=chnkr.n[:, :, target_chunk],
        d2=chnkr.d2[:, :, target_chunk],
    )
    nodes, weights = lege.exps(max(27, chnkr.k + 1))[:2]
    bary = lege.barywts(chnkr.k, chnkr.tstor)

    mat, maxrecs, numints, iers = quadadap.adapgausswts(
        chnkr,
        source_chunk,
        targ,
        kern,
        kern.opdims,
        nodes,
        weights,
        bary,
        {"eps": float(fixture.eps)},
    )
    ggq = chunkermat(chnkr, kern)
    rows = slice(target_chunk * chnkr.k, (target_chunk + 1) * chnkr.k)
    cols = slice(source_chunk * chnkr.k, (source_chunk + 1) * chnkr.k)
    matcomp = ggq[rows, cols]

    np.testing.assert_allclose(mat, fixture.mat, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(matcomp, fixture.matcomp, rtol=1e-10, atol=2e-9)
    np.testing.assert_allclose(matcomp, mat, rtol=1e-10, atol=2e-9)
    np.testing.assert_array_equal(maxrecs, np.asarray(fixture.maxrecs, dtype=int).reshape(-1))
    np.testing.assert_array_equal(numints, np.asarray(fixture.numints, dtype=int).reshape(-1))
    np.testing.assert_array_equal(iers, np.asarray(fixture.iers, dtype=int).reshape(-1))
    assert float(fixture.inferr) < 1e-11
    assert np.linalg.norm(matcomp - mat, ord=np.inf) < 1e-9


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


def test_chunkermat_quadadap_closetotouching_devtools_solve_matches_matlab():
    fixture = load_devtools_easy().chunkermat_quadadap_closetotouching
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    lap_s = kernel("lap", "s")
    combined = kernel("lap", "c", [1.0, float(fixture.eta)])

    ubdry = lap_s(src, pointinfo(chnkr)) @ strengths
    utarg = lap_s(src, PointInfo(r=targets)) @ strengths
    mat_adap = chunkermat(chnkr, combined, {"adaptive_correction": True, "robust": True})
    mat_original = chunkermat(chnkr, combined)
    probe = np.asarray(fixture.sysa_probe_rhs)
    rhs = np.asarray(fixture.rhs).reshape(-1, order="F")
    sys_adap = 0.5 * np.eye(chnkr.npt) + mat_adap
    sys_original = 0.5 * np.eye(chnkr.npt) + mat_original
    sol_adap = np.linalg.solve(sys_adap, rhs)
    sol_original = np.linalg.solve(sys_original, rhs)
    layer_adap = chunkerkerneval(chnkr, combined, sol_adap, targets, {"forceadap": True}).reshape(-1, order="F")
    layer_original = chunkerkerneval(chnkr, combined, sol_original, targets, {"forceadap": True}).reshape(-1, order="F")
    relerr_adap = np.linalg.norm(utarg - layer_adap) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))
    relerr_original = np.linalg.norm(utarg - layer_original) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))
    relerr2_adap = np.linalg.norm(utarg - layer_adap, ord=np.inf) / np.dot(np.abs(sol_adap), chnkr.wts.reshape(-1, order="F"))
    relerr2_original = np.linalg.norm(utarg - layer_original, ord=np.inf) / np.dot(np.abs(sol_original), chnkr.wts.reshape(-1, order="F"))

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(mat_adap @ probe, np.asarray(fixture.mata_probe), rtol=3e-8, atol=1e-8)
    np.testing.assert_allclose(mat_original @ probe, np.asarray(fixture.mato_probe), rtol=3e-8, atol=1e-8)
    np.testing.assert_allclose(sol_adap, np.asarray(fixture.sola).reshape(-1, order="F"), rtol=5e-8, atol=5e-10)
    np.testing.assert_allclose(sol_original, np.asarray(fixture.solo).reshape(-1, order="F"), rtol=5e-8, atol=5e-10)
    np.testing.assert_allclose(layer_adap, np.asarray(fixture.layersola).reshape(-1, order="F"), rtol=5e-8, atol=5e-10)
    np.testing.assert_allclose(layer_original, np.asarray(fixture.layersolo).reshape(-1, order="F"), rtol=5e-8, atol=5e-10)
    assert max(relerr_adap, float(fixture.relerr_adap)) < 1e-10
    assert max(relerr2_adap, float(fixture.relerr2_adap)) < 1e-10
    np.testing.assert_allclose(relerr_original, float(fixture.relerr_original), rtol=5e-8, atol=5e-10)
    np.testing.assert_allclose(relerr2_original, float(fixture.relerr2_original), rtol=5e-8, atol=5e-10)


def test_chunkermatapply_scalar_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermatapply_scalar
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    lap_s = kernel("lap", "s")
    lap_d = kernel("lap", "d")
    dens = (lap_s(src, pointinfo(chnkr)) * float(fixture.strengths)).reshape(-1, order="F")
    sysmat = chunkermat(chnkr, lap_d)
    sys = -0.5 * np.eye(chnkr.npt) + sysmat
    udense = sys @ dens
    u_apply = -0.5 * dens + chunkermatapply(chnkr, lap_d, dens)
    e1 = np.zeros(chnkr.npt)
    e1[0] = 1.0
    sys11 = -0.5 * e1 + chunkermatapply(chnkr, lap_d, e1)
    sol_dense = np.linalg.solve(sys, dens)
    apply_relerr = np.linalg.norm(udense - u_apply) / np.linalg.norm(udense)

    np.testing.assert_allclose(dens, np.asarray(fixture.dens).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(udense, np.asarray(fixture.udense).reshape(-1, order="F"), rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(u_apply, np.asarray(fixture.u_apply).reshape(-1, order="F"), rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(sys11, np.asarray(fixture.sys11).reshape(-1, order="F"), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(sol_dense, np.asarray(fixture.sol_dense).reshape(-1, order="F"), rtol=5e-5, atol=2e-5)
    np.testing.assert_allclose(sol_dense, np.asarray(fixture.sol_gmres).reshape(-1, order="F"), rtol=5e-5, atol=2e-5)
    assert max(apply_relerr, float(fixture.apply_relerr)) < 1e-13
    assert float(fixture.solve_relerr) < 1e-13


def test_chunkermatapply_vector_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermatapply_vector
    chnkr = chunker_from_fields(fixture.chunker)
    kern = transmission_all_kernel_from_fixture(fixture)
    bdry_data = transmission_point_source_boundary_data(chnkr, fixture)
    sysmat = chunkermat(chnkr, kern)
    sys = np.eye(sysmat.shape[0], dtype=complex) + sysmat
    udense = sys @ bdry_data
    u_apply = bdry_data + chunkermatapply(chnkr, kern, bdry_data)
    probe = np.asarray(fixture.probe)
    sys_probe = sys @ probe
    apply_relerr = np.linalg.norm(udense - u_apply) / np.linalg.norm(udense)
    matlab_udense = np.asarray(fixture.udense).reshape(-1, order="F")
    matlab_u_apply = np.asarray(fixture.u_apply).reshape(-1, order="F")
    matlab_sys_probe = np.asarray(fixture.sys_probe)

    np.testing.assert_allclose(bdry_data, np.asarray(fixture.bdry_data).reshape(-1, order="F"), rtol=1e-12, atol=1e-11)
    assert np.linalg.norm(udense - matlab_udense) / np.linalg.norm(matlab_udense) < 1e-5
    assert np.max(np.abs(udense - matlab_udense)) < 2e-3
    assert np.linalg.norm(u_apply - matlab_u_apply) / np.linalg.norm(matlab_u_apply) < 1e-5
    assert np.max(np.abs(u_apply - matlab_u_apply)) < 2e-3
    assert np.linalg.norm(sys_probe - matlab_sys_probe) / np.linalg.norm(matlab_sys_probe) < 3e-5
    assert np.max(np.abs(sys_probe - matlab_sys_probe)) < 1e-2
    assert max(apply_relerr, float(fixture.apply_relerr)) < 1e-13


def test_chunkermatapply_graph_scalar_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermatapply_graph_scalar
    cgrph = chunkermatapply_graph_from_fixture(fixture)
    src = PointInfo(r=point_array(fixture.sources))
    lap_s = kernel("lap", "s")
    lap_d = -2 * kernel("lap", "d")
    dens = (lap_s(src, pointinfo(cgrph)) * float(fixture.strengths)).reshape(-1, order="F")
    sysmat = chunkermat(cgrph, lap_d)
    sys = np.eye(cgrph.npt) + sysmat
    udense = sys @ dens
    u_apply = dens + chunkermatapply(cgrph, lap_d, dens)
    probe = np.asarray(fixture.probe)
    sys_probe = sys @ probe
    apply_relerr = np.linalg.norm(udense - u_apply) / np.linalg.norm(udense)
    matlab_udense = np.asarray(fixture.udense).reshape(-1, order="F")
    matlab_u_apply = np.asarray(fixture.u_apply).reshape(-1, order="F")
    matlab_sys_probe = np.asarray(fixture.sys_probe)

    np.testing.assert_allclose(dens, np.asarray(fixture.dens).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(u_apply, udense, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(matlab_u_apply, matlab_udense, rtol=1e-12, atol=1e-13)
    assert np.linalg.norm(udense - matlab_udense) / np.linalg.norm(matlab_udense) < 3.5e-2
    assert np.linalg.norm(sys_probe - matlab_sys_probe) / np.linalg.norm(matlab_sys_probe) < 6e-2
    assert max(apply_relerr, float(fixture.apply_relerr)) < 1e-13


def test_chunkermatapply_graph_vector_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermatapply_graph_vector
    cgrph = chunkermatapply_graph_from_fixture(fixture)
    kern = transmission_all_kernel_from_fixture(fixture)
    bdry_data = transmission_point_source_boundary_data(cgrph, fixture)
    sysmat = chunkermat(cgrph, kern)
    sys = np.eye(sysmat.shape[0], dtype=complex) + sysmat
    udense = sys @ bdry_data
    u_apply = bdry_data + chunkermatapply(cgrph, kern, bdry_data)
    probe = np.asarray(fixture.probe)
    sys_probe = sys @ probe
    apply_relerr = np.linalg.norm(udense - u_apply) / np.linalg.norm(udense)
    matlab_udense = np.asarray(fixture.udense).reshape(-1, order="F")
    matlab_u_apply = np.asarray(fixture.u_apply).reshape(-1, order="F")
    matlab_sys_probe = np.asarray(fixture.sys_probe)

    np.testing.assert_allclose(bdry_data, np.asarray(fixture.bdry_data).reshape(-1, order="F"), rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(u_apply, udense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matlab_u_apply, matlab_udense, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(udense - matlab_udense) / np.linalg.norm(matlab_udense) < 1e-3
    assert np.linalg.norm(sys_probe - matlab_sys_probe) / np.linalg.norm(matlab_sys_probe) < 1e-2
    assert max(apply_relerr, float(fixture.apply_relerr)) < 1e-13


def test_chunkermat_laplace_solve_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermat_laplace
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    lap_s = kernel("lap", "s")
    lap_d = kernel("lap", "d")

    ubdry = lap_s(src, pointinfo(chnkr)) @ strengths
    utarg = lap_s(src, PointInfo(r=targets)) @ strengths
    dmat = chunkermat(chnkr, lap_d)
    sys = -0.5 * np.eye(chnkr.npt) + dmat
    rhs = ubdry.reshape(-1, order="F")
    sol = np.linalg.solve(sys, rhs)
    dsol = chunkerkerneval(chnkr, lap_d, sol, targets, {"forceadap": True}).reshape(-1, order="F")
    relerr = np.linalg.norm(utarg - dsol) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))
    relerr2 = np.linalg.norm(utarg - dsol, ord=np.inf) / np.dot(np.abs(sol), chnkr.wts.reshape(-1, order="F"))

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dmat, np.asarray(fixture.D), rtol=1e-8, atol=3e-9)
    np.testing.assert_allclose(sys, np.asarray(fixture.sys), rtol=1e-8, atol=3e-9)
    np.testing.assert_allclose(rhs, np.asarray(fixture.rhs).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol_backslash).reshape(-1, order="F"), rtol=5e-9, atol=3e-9)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol_gmres).reshape(-1, order="F"), rtol=5e-9, atol=3e-9)
    np.testing.assert_allclose(dsol, np.asarray(fixture.Dsol).reshape(-1, order="F"), rtol=1e-9, atol=1e-11)
    assert max(relerr, float(fixture.relerr)) < 1e-10
    assert max(relerr2, float(fixture.relerr2)) < 1e-10
    assert float(fixture.solve_relerr) < 1e-12


def test_chunkermat_helm2d_solve_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermat_helm2d
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    targets = point_array(fixture.targets)
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    zk = complex(fixture.zk)
    helm_s = kernel("helm", "s", zk)
    helm_d = kernel("helm", "d", zk)

    ubdry = helm_s(src, pointinfo(chnkr)) @ strengths
    utarg = helm_s(src, PointInfo(r=targets)) @ strengths
    dmat = chunkermat(chnkr, helm_d)
    sys = -0.5 * np.eye(chnkr.npt) + dmat
    rhs = ubdry.reshape(-1, order="F")
    sol = np.linalg.solve(sys, rhs)
    dsol = chunkerkerneval(chnkr, helm_d, sol, targets, {"forceadap": True}).reshape(-1, order="F")
    relerr = np.linalg.norm(utarg - dsol) / (np.sqrt(chnkr.nch) * np.linalg.norm(utarg))
    relerr2 = np.linalg.norm(utarg - dsol, ord=np.inf) / np.dot(np.abs(sol), chnkr.wts.reshape(-1, order="F"))

    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(utarg, np.asarray(fixture.utarg).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dmat, np.asarray(fixture.D), rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(sys, np.asarray(fixture.sys), rtol=1e-8, atol=5e-9)
    np.testing.assert_allclose(rhs, np.asarray(fixture.rhs).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol_backslash).reshape(-1, order="F"), rtol=5e-9, atol=3e-9)
    np.testing.assert_allclose(sol, np.asarray(fixture.sol_gmres).reshape(-1, order="F"), rtol=5e-9, atol=3e-9)
    np.testing.assert_allclose(dsol, np.asarray(fixture.Dsol).reshape(-1, order="F"), rtol=1e-9, atol=1e-11)
    assert max(relerr, float(fixture.relerr)) < 1e-10
    assert max(relerr2, float(fixture.relerr2)) < 1e-10
    assert float(fixture.solve_relerr) < 1e-12


def test_chunkermat_l2scale_devtools_outputs_match_matlab():
    fixture = load_devtools_easy().chunkermat_l2scale
    zk0 = float(fixture.zk0)
    zk1 = float(fixture.zk1)
    modes = np.array([float(fixture.modes)])
    chnkr = chunkerfuncuni(
        lambda t: curves.bymode(t, modes, [0.0, 0.0], [1.2, 1.0]),
        int(fixture.nch),
        {"nover": 0, "ifclosed": True, "eps": 1.0e-10},
    ).sort()[0]
    assert chnkr.npt == int(fixture.npt)

    srcinfo0 = {"r": np.array([[0.0], [0.0]])}
    srcinfo1 = {"r": np.array([[-10.0], [5.0]])}
    tinfo = pointinfo(chnkr)
    u0bdr = helm2d.kern(zk0, srcinfo0, tinfo, "s").reshape(-1)
    u0nbdr = helm2d.kern(zk0, srcinfo0, tinfo, "sprime").reshape(-1)
    u1bdr = helm2d.kern(zk1, srcinfo1, tinfo, "s").reshape(-1)
    u1nbdr = helm2d.kern(zk1, srcinfo1, tinfo, "sprime").reshape(-1)
    weights = chnkr.wts.reshape(-1, order="F")
    sqrt_weights = np.sqrt(weights)
    npt = chnkr.npt
    nn = 2 * npt
    rhs = np.zeros(nn, dtype=complex)
    rhs[0::2] = -(-u0bdr + u1bdr) * sqrt_weights
    rhs[1::2] = -(-u0nbdr + u1nbdr) * sqrt_weights / (1j * zk0 + 1j * zk1) * 2

    kd = kernel("helmdiff", "d", [zk0, zk1], [1, 1])
    ks = kernel("helmdiff", "s", [zk0, zk1], [-1j * zk0, -1j * zk1])
    kdp = kernel("helmdiff", "dprime", [zk0, zk1], [1, 1])
    ksp = kernel("helmdiff", "sprime", [zk0, zk1], [-1j * zk0, -1j * zk1])

    manual = np.zeros((nn, nn), dtype=complex)
    dd = np.diag(sqrt_weights)
    ddinv = np.diag(1.0 / sqrt_weights)
    manual[0::2, 0::2] = dd @ (chunkermat(chnkr, kd) + np.eye(npt)) @ ddinv
    manual[0::2, 1::2] = dd @ chunkermat(chnkr, ks) @ ddinv
    manual[1::2, 0::2] = dd @ (chunkermat(chnkr, kdp) / (1j * zk0 + 1j * zk1) * 2) @ ddinv
    manual[1::2, 1::2] = dd @ (chunkermat(chnkr, ksp) / (1j * zk0 + 1j * zk1) * 2 + np.eye(npt)) @ ddinv

    opts = {"l2scale": "true"}
    scaled = np.zeros_like(manual)
    scaled[0::2, 0::2] = chunkermat(chnkr, kd, opts) + np.eye(npt)
    scaled[0::2, 1::2] = chunkermat(chnkr, ks, opts)
    scaled[1::2, 0::2] = chunkermat(chnkr, kdp, opts) / (1j * zk0 + 1j * zk1) * 2
    scaled[1::2, 1::2] = chunkermat(chnkr, ksp, opts) / (1j * zk0 + 1j * zk1) * 2 + np.eye(npt)

    err_matrix = np.linalg.norm(manual - scaled, "fro")
    err_density = np.linalg.norm(np.linalg.solve(manual, rhs) - np.linalg.solve(scaled, rhs))
    assert err_matrix < 1e-10
    assert err_density < 1e-11
    assert float(fixture.err_matrix) < 1e-10
    assert float(fixture.err_density) < 1e-11
    assert err_matrix <= max(1e-10, 10 * float(fixture.err_matrix))
    assert err_density <= max(1e-11, 10 * float(fixture.err_density))


def test_singularkernel_devtools_pv_hs_outputs_match_matlab():
    fixture = load_devtools_easy().singularkernel
    chnkr = chunker_from_fields(fixture.chunker)
    src = PointInfo(r=point_array(fixture.sources))
    strengths = np.asarray(fixture.strengths).reshape(-1, order="F")
    boundary = pointinfo(chnkr)
    lap_s = kernel("lap", "s")
    lap_sgrad = kernel("lap", "sg")
    lap_sp = kernel("lap", "sp")
    lap_stau = kernel("lap", "stau")
    lap_d = kernel("lap", "d")
    lap_dp = kernel("lap", "dp")

    ubdry = lap_s(src, boundary) @ strengths
    grad = (lap_sgrad(src, boundary) @ strengths).reshape(2, chnkr.npt, order="F")
    normals = chnkr.n.reshape(2, chnkr.npt, order="F")
    tangents = np.vstack((-normals[1], normals[0]))
    unbdry = np.sum(normals * grad, axis=0)
    utbdry = np.sum(tangents * grad, axis=0)
    sprime = chunkermat(chnkr, lap_sp)
    stau = chunkermat(chnkr, lap_stau)
    dmat = chunkermat(chnkr, lap_d)
    dprime = chunkermat(chnkr, lap_dp)
    probe = np.asarray(fixture.probe)
    sys_pv = 0.5 * np.eye(chnkr.npt) + sprime + np.ones((chnkr.npt, 1)) @ chnkr.wts.reshape(1, -1, order="F")
    mu_pv = np.linalg.solve(sys_pv, unbdry)
    utau = stau @ mu_pv
    sys_hs = -0.5 * np.eye(chnkr.npt) + dmat
    mu_hs = np.linalg.solve(sys_hs, ubdry)
    un = dprime @ mu_hs
    relerr_pv = np.linalg.norm(utau - utbdry) / np.linalg.norm(utbdry)
    relerr_hs = np.linalg.norm(un - unbdry) / np.linalg.norm(unbdry)

    assert np.isfinite(sprime).all()
    assert np.isfinite(stau).all()
    assert np.isfinite(dprime).all()
    np.testing.assert_allclose(ubdry, np.asarray(fixture.ubdry).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(unbdry, np.asarray(fixture.unbdry).reshape(-1, order="F"), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(utbdry, np.asarray(fixture.utbdry).reshape(-1, order="F"), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(sprime @ probe, np.asarray(fixture.sprime_probe), rtol=5e-9, atol=1e-9)
    np.testing.assert_allclose(stau @ probe, np.asarray(fixture.stau_probe), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dmat @ probe, np.asarray(fixture.d_probe), rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(dprime @ probe, np.asarray(fixture.dprime_probe), rtol=5e-7, atol=5e-4)
    np.testing.assert_allclose(mu_pv, np.asarray(fixture.mu_pv).reshape(-1, order="F"), rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(utau, np.asarray(fixture.utau).reshape(-1, order="F"), rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(mu_hs, np.asarray(fixture.mu_hs).reshape(-1, order="F"), rtol=5e-8, atol=5e-9)
    np.testing.assert_allclose(un, np.asarray(fixture.un).reshape(-1, order="F"), rtol=2e-3, atol=3e-4)
    assert max(relerr_pv, float(fixture.relerr_pv)) < 1e-9
    assert relerr_hs < 2e-4
    assert float(fixture.relerr_hs) < 2e-4


def test_datafield_devtools_target_data_flam_matches_matlab():
    fixture = load_devtools_easy().datafield
    chnkr = chunker_from_fields(fixture.chunker)
    srcinfo = PointInfo(r=point_array(fixture.srcinfo.r))
    targinfo = PointInfo(r=point_array(fixture.targinfo.r), data=np.asarray(fixture.targinfo.data))
    v = np.asarray(fixture.v).reshape(-1, order="F")

    def directional_derivative_single_layer(src, targ):
        _, grad, _ = lap2d.green(src.r, targ.r)
        target_data = np.asarray(targ.data)
        if target_data.ndim == 1:
            target_data = target_data.reshape(2, -1)
        return grad[:, :, 0] * target_data[0, :, None] + grad[:, :, 1] * target_data[1, :, None]

    directional_derivative_kernel = Kernel(
        name="directional_derivative_single_layer",
        type="custom",
        eval=directional_derivative_single_layer,
        opdims=(1, 1),
        sing="pv",
    )
    spkern = kernel("lap", "sp")

    unbdry = spkern(srcinfo, pointinfo(chnkr)).reshape(-1, order="F")
    mu = np.asarray(fixture.mu).reshape(-1, order="F")
    deru = chunkerkerneval(chnkr, directional_derivative_kernel, mu, targinfo).reshape(-1, order="F")
    deru_adap = chunkerkerneval(chnkr, directional_derivative_kernel, mu, targinfo, {"forceadap": True}).reshape(-1, order="F")
    flam_opts = {"acceleration": "flam", "occ": 32, "rank_or_tol": 1.0e-10}
    deru_flam = chunkerkerneval(chnkr, directional_derivative_kernel, mu, targinfo, flam_opts).reshape(-1, order="F")
    gradutrue = kernel("lap", "sg")(srcinfo, targinfo).reshape(2, -1, order="F")
    derutrue = np.sum(gradutrue * v[:, None], axis=0)

    np.testing.assert_allclose(unbdry, fixture.unbdry, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(deru, np.asarray(fixture.deru).reshape(-1, order="F"), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(deru_adap, np.asarray(fixture.deru_adap).reshape(-1, order="F"), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(deru_flam, np.asarray(fixture.deru_flam).reshape(-1, order="F"), rtol=1e-8, atol=2e-9)
    np.testing.assert_allclose(derutrue, np.asarray(fixture.derutrue).reshape(-1, order="F"), rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(deru - derutrue) / np.linalg.norm(derutrue) < 1e-10
    assert np.linalg.norm(deru_adap - deru) / np.linalg.norm(deru) < 1e-10
    assert np.linalg.norm(deru_flam - deru) / np.linalg.norm(deru) < 1e-8
    assert float(fixture.err_direct) < 1e-10
    assert float(fixture.err_adap) < 1e-10
    assert float(fixture.err_flam) < 1e-10


def test_datafield_devtools_hilbert_data_flam_matches_matlab():
    fixture = load_devtools_easy().datafield.hilbert
    chnkr = chunker_from_fields(fixture.chunker)
    data = np.asarray(fixture.data).reshape(1, chnkr.npt, order="F")
    chnkr.makedatarows(data.shape[0])
    chnkr.data = data.reshape(data.shape[0], chnkr.k, chnkr.nch, order="F")
    length = float(fixture.L)

    def cotangent_kernel(src, targ):
        theta = targ.data.reshape(-1, order="F")[:, None] - src.data.reshape(-1, order="F")[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.tan(theta / 2.0)

    hkern = Kernel(
        name="cotan",
        type="cot",
        eval=cotangent_kernel,
        opdims=(1, 1),
        sing="pv",
    )

    f1 = np.asarray(fixture.f1).reshape(-1, order="F")
    f2 = (chunkermat(chnkr, hkern) / length) @ f1
    f2_flam = chunkermat(
        chnkr,
        hkern,
        {"acceleration": "flam", "occ": 64, "rank_or_tol": 1.0e-10, "useproxy": False},
    ) @ f1 / length

    np.testing.assert_allclose(f2, np.asarray(fixture.f2).reshape(-1, order="F"), rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(f2_flam, np.asarray(fixture.f2_flam).reshape(-1, order="F") / length, rtol=2e-6, atol=1e-8)
    assert np.linalg.norm(f1**2 + f2**2 - 1.0) / np.linalg.norm(f1**2) < 1e-8
    assert np.linalg.norm(f2 - f2_flam) / np.linalg.norm(f2) < 1e-8
    assert float(fixture.err_circle) < 1e-8
    assert float(fixture.err_flam) < 1e-10


def test_flam_proxy_geometry_helpers_match_matlab_fixture():
    fixture = load_devtools_easy().flam_helpers

    pr, ptau, pw, pin = flam.proxy_square_pts(64)
    proxy, pnorm, cpw = flam.proxy_circ_pts(16)

    np.testing.assert_allclose(pr, fixture.square64_pr, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(ptau, fixture.square64_ptau, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(pw, np.asarray(fixture.square64_pw).reshape(-1), rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(
        pin(np.array([[0.0, 2.0, -1.49, 1.51], [0.0, 0.0, 1.49, 0.0]])),
        np.asarray(fixture.square64_inside, dtype=bool).reshape(-1),
    )
    np.testing.assert_allclose(proxy, fixture.circle16_proxy, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(pnorm, fixture.circle16_pnorm, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cpw, np.asarray(fixture.circle16_pw).reshape(-1), rtol=1e-14, atol=1e-14)

    rpr, rptau, rpw, _ = flam.proxy_rect_pts([2.0, 3.0], [4, 6])
    np.testing.assert_allclose(rpr, fixture.rect_pr, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(rptau, fixture.rect_ptau, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(rpw, np.asarray(fixture.rect_pw).reshape(-1), rtol=1e-14, atol=1e-14)
    assert flam.nproxy_square(
        kernel("lap", "s"),
        float(fixture.nproxy_lap_s_width),
        {"nsrc": int(fixture.nproxy_lap_s_nsrc), "rank_or_tol": float(fixture.nproxy_lap_s_tol)},
    ) == int(fixture.nproxy_lap_s)

    chnkr = chunker_from_fields(fixture.kernbyindex_chunker)
    lap_s = kernel("lap", "s")
    rows = np.asarray(fixture.kbi_rows, dtype=int).reshape(-1) - 1
    cols = np.asarray(fixture.kbi_cols, dtype=int).reshape(-1) - 1
    np.testing.assert_allclose(
        flam.kernbyindex(rows, cols, chnkr, lap_s, (1, 1)),
        fixture.kbi_mat,
        rtol=1e-13,
        atol=1e-13,
    )
    overwrite = sparse.csr_matrix((np.array([9.0, -4.0]), (np.array([2, 7]), np.array([4, 8]))), shape=(chnkr.npt, chnkr.npt))
    np.testing.assert_allclose(
        flam.kernbyindex(rows, cols, chnkr, lap_s, (1, 1), overwrite),
        fixture.kbi_overwrite,
        rtol=1e-13,
        atol=1e-13,
    )

    rrows = np.asarray(fixture.kbir_rows, dtype=int).reshape(-1) - 1
    rcols = np.asarray(fixture.kbir_cols, dtype=int).reshape(-1) - 1
    targets = np.asarray(fixture.kbir_targets)
    np.testing.assert_allclose(
        flam.kernbyindexr(rrows, rcols, targets, chnkr, lap_s, (1, 1)),
        fixture.kbir_mat,
        rtol=1e-13,
        atol=1e-13,
    )
    roverwrite = sparse.csr_matrix((np.array([7.0, -3.0]), (np.array([1, 2]), np.array([2, 5]))), shape=(targets.shape[1], chnkr.npt))
    np.testing.assert_allclose(
        flam.kernbyindexr(rrows, rcols, targets, chnkr, lap_s, (1, 1), roverwrite),
        fixture.kbir_overwrite,
        rtol=1e-13,
        atol=1e-13,
    )

    pslf = np.asarray(fixture.proxyfun_slf, dtype=int).reshape(-1) - 1
    pnbr = np.asarray(fixture.proxyfun_nbr, dtype=int).reshape(-1) - 1
    pK, pnbr_out = flam.proxyfun(
        pslf,
        pnbr,
        np.asarray(fixture.proxyfun_l, dtype=float).reshape(-1),
        np.asarray(fixture.proxyfun_ctr, dtype=float).reshape(2),
        chnkr,
        lap_s,
        (1, 1),
        pr,
        ptau,
        pw,
        pin,
    )
    np.testing.assert_allclose(pK, fixture.proxyfun_K, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(pnbr_out + 1, np.asarray(fixture.proxyfun_nbr_out, dtype=int).reshape(-1))

    pKc, pnbr_c = flam.proxyfunr("c", targets, chnkr.r.reshape(2, -1, order="F"), rcols, rrows, [1.0, 1.0], [0.0, 0.0], chnkr, lap_s, (1, 1), pr, ptau, pw, pin)
    pKr, pnbr_r = flam.proxyfunr("r", targets, chnkr.r.reshape(2, -1, order="F"), rrows, rcols, [1.0, 1.0], [0.0, 0.0], chnkr, lap_s, (1, 1), pr, ptau, pw, pin)
    np.testing.assert_allclose(pKc, fixture.proxyfunr_c_K, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(pKr, fixture.proxyfunr_r_K, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(pnbr_c + 1, np.asarray(fixture.proxyfunr_c_nbr, dtype=int).reshape(-1))
    np.testing.assert_array_equal(pnbr_r + 1, np.asarray(fixture.proxyfunr_r_nbr, dtype=int).reshape(-1))
