"""Generate Python-side snapshots for the easy MATLAB devtools fixture.

The pytest suite recomputes these values live. This script is for manual
side-by-side inspection when regenerating MATLAB data from
``scripts/matlab/generate_devtools_easy_fixture.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from chunkie import Chunker, chunkerfit, chunkerfunc, chunkerfuncuni, chunkerintegral, chunkerinterior, chunkerkerneval, chunkerkernevalmat, chunkermat, chunkermatapply, chunkerpoly, chunkgraph, kernel, lege, tochunkgraph
from chunkie.chnk import arcparam, curves, flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself, helm2d, quadadap, smoother, spcl
from chunkie.operators import PointInfo


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
MATLAB_FIXTURE = GOLDEN / "devtools_easy.mat"
PYTHON_SNAPSHOT = GOLDEN / "devtools_easy_python.npz"


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


def pointinfo_from_chunker(chnkr: Chunker) -> PointInfo:
    return PointInfo(
        r=chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F"),
        d=chnkr.d.reshape(chnkr.dim, chnkr.npt, order="F"),
        d2=chnkr.d2.reshape(chnkr.dim, chnkr.npt, order="F"),
        n=chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F"),
    )


def transmission_all_kernel_from_fixture(fixture):
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
    return kernel("helmdiff", "all", [ks[d1], ks[d2]], np.stack((-cc1, -cc2), axis=2))


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
    targ = pointinfo_from_chunker(chnkr)
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


def build_snapshot() -> dict[str, np.ndarray]:
    fixture = loadmat(MATLAB_FIXTURE, squeeze_me=True, struct_as_record=False)["devtools_easy"]
    out: dict[str, np.ndarray] = {}

    acg = fixture.absconvgauss
    acg_val, acg_der, acg_der2 = spcl.absconvgauss(acg.x, float(acg.m), float(acg.offset), float(acg.h))
    out["absconvgauss_val"] = acg_val
    out["absconvgauss_der"] = acg_der
    out["absconvgauss_der2"] = acg_der2

    leg = fixture.legeexpsunit
    k = int(leg.k)
    x, w, u, v = lege.exps(k)
    dmat = lege.dermat(k, u, v)
    imat, *_ = lege.intmat(k, u, v)
    out["lege_x"] = x
    out["lege_w"] = w
    out["lege_u"] = u
    out["lege_v"] = v
    out["lege_dmat"] = dmat
    out["lege_imat"] = imat
    out["lege_dpv"] = dmat @ np.sin(x)
    out["lege_ipv"] = imat @ np.sin(x)
    out["lege_cfsint"] = lege.intpol(leg.cfs)
    out["lege_cfsint_original"] = lege.intpol(leg.cfs, "original")
    out["lege_integral_exev"] = lege.exev(x, lege.intpol(leg.cfs))

    arc = fixture.arclengthfun
    out["arclength_single"] = chunker_from_fields(arc.chunker_single).arclengthfun()
    out["arclength_merged"] = chunker_from_fields(arc.chunker_merged).arclengthfun()

    cap = fixture.chunkerarcparam
    cap_chunker = chunker_from_fields(cap.chunker)
    cap_data = arcparam.init(cap_chunker)
    cap_r, cap_d, cap_d2 = arcparam.eval(np.asarray(cap.s_nodes).reshape(-1, order="F"), cap_data)
    cap_sample_r, cap_sample_d, cap_sample_d2 = arcparam.eval(cap.sample_s, cap_data)
    cap_resampled, cap_eps = cap_chunker.arcresample({"mv_bdries": 0})
    out["chunkerarcparam_r_nodes"] = cap_r
    out["chunkerarcparam_d_nodes"] = cap_d
    out["chunkerarcparam_d2_nodes"] = cap_d2
    out["chunkerarcparam_sample_r"] = cap_sample_r
    out["chunkerarcparam_sample_d"] = cap_sample_d
    out["chunkerarcparam_sample_d2"] = cap_sample_d2
    out["chunkerarcparam_resampled_r"] = cap_resampled.r
    out["chunkerarcparam_resampled_eps"] = np.asarray(cap_eps)

    dimat = fixture.chunker_diffintmat
    ellipse = chunker_from_fields(dimat.ellipse)
    ellipse_d = ellipse.diffmat()
    ellipse_c = ellipse.intmat()
    ellipse_x = ellipse.r[0].reshape(-1, order="F")
    ellipse_y = ellipse.r[1].reshape(-1, order="F")
    ellipse_dx = ellipse_d @ ellipse_x
    ellipse_dy = ellipse_d @ ellipse_y
    ellipse_x_int = ellipse_c @ ellipse_dx
    ellipse_y_int = ellipse_c @ ellipse_dy
    out["chunker_diffintmat_ellipse_D"] = ellipse_d
    out["chunker_diffintmat_ellipse_C"] = ellipse_c
    out["chunker_diffintmat_ellipse_dx"] = ellipse_dx
    out["chunker_diffintmat_ellipse_dy"] = ellipse_dy
    out["chunker_diffintmat_ellipse_x_int"] = ellipse_x_int
    out["chunker_diffintmat_ellipse_y_int"] = ellipse_y_int
    out["chunker_diffintmat_ellipse_tangent_residual"] = ellipse_dx**2 + ellipse_dy**2 - 1.0
    out["chunker_diffintmat_ellipse_x_residual"] = ellipse_x_int - ellipse_x_int[0] - ellipse_x + ellipse_x[0]
    out["chunker_diffintmat_ellipse_y_residual"] = ellipse_y_int - ellipse_y_int[0] - ellipse_y + ellipse_y[0]

    circle = chunker_from_fields(dimat.circle)
    circle_d = circle.diffmat()
    circle_x = circle.r[0].reshape(-1, order="F")
    circle_y = circle.r[1].reshape(-1, order="F")
    out["chunker_diffintmat_circle_D"] = circle_d
    out["chunker_diffintmat_circle_test_quant"] = circle_d @ circle_x + circle_y

    near = fixture.chunker_nearest
    near_circle = chunker_from_fields(near.circle)
    rn, dn, d2n, dist, tn, ichn = near_circle.nearest(near.targs)
    out["chunker_nearest_rn"] = rn
    out["chunker_nearest_dn"] = dn
    out["chunker_nearest_d2n"] = d2n
    out["chunker_nearest_dist"] = dist
    out["chunker_nearest_tn"] = tn
    out["chunker_nearest_ichn"] = ichn
    theta_targ = np.arctan2(near.targs[1], near.targs[0])
    theta_near = np.arctan2(rn[1], rn[0])
    out["chunker_nearest_angle_err"] = np.abs(np.angle(np.exp(1j * (theta_targ - theta_near))))

    cint = fixture.chunkerintegral
    cint_chunker = chunker_from_fields(cint.chunker)
    out["chunkerintegral_fvals"] = np.cos(cint_chunker.r.reshape(2, cint_chunker.npt, order="F")[0] - 1.0) + np.sin(
        cint_chunker.r.reshape(2, cint_chunker.npt, order="F")[1] - 0.5
    )
    out["chunkerintegral_value"] = np.asarray(chunkerintegral(cint_chunker, cint.fvals))
    out["chunkerintegral_callable"] = np.asarray(
        chunkerintegral(
            cint_chunker,
            lambda xx: np.cos(xx[0] - 1.0) + np.sin(xx[1] - 0.5),
        )
    )

    cfu = fixture.chunkerfuncuni
    cfu_starfish = chunkerfuncuni(lambda t: curves.starfish(t, int(cfu.narms), float(cfu.amp)), int(cfu.nch))
    cfu_bymode = chunkerfuncuni(lambda t: curves.bymode(t, cfu.modes, cfu.mode_ctr), int(cfu.nch)).reverse()
    cfu_circle = chunkerfuncuni(
        lambda t: (
            np.vstack(
                (
                    cfu.circle_ctr[0] + float(cfu.circle_radius) * np.cos(t),
                    cfu.circle_ctr[1] + float(cfu.circle_radius) * np.sin(t),
                )
            ),
            np.vstack((-float(cfu.circle_radius) * np.sin(t), float(cfu.circle_radius) * np.cos(t))),
            np.vstack((-float(cfu.circle_radius) * np.cos(t), -float(cfu.circle_radius) * np.sin(t))),
        ),
        int(cfu.nch),
    )
    out["chunkerfuncuni_starfish_r"] = cfu_starfish.r
    out["chunkerfuncuni_bymode_reversed_r"] = cfu_bymode.r
    out["chunkerfuncuni_circle_area"] = np.asarray(cfu_circle.area())

    cfunc = fixture.chunkerfunc
    cfunc_starfish, _ = chunkerfunc(
        lambda t: curves.starfish(t, int(cfunc.narms), float(cfunc.amp)),
        {"eps": 1.0e-4},
        {"k": 16},
    )
    cfunc_starfish_nout, _ = chunkerfunc(
        lambda t: curves.starfish(t, int(cfunc.narms), float(cfunc.amp)),
        {"eps": 1.0e-4, "nout": 3},
        {"k": 16},
    )
    cfunc_bymode, _ = chunkerfunc(
        lambda t: curves.bymode(t, cfunc.modes, cfunc.mode_ctr),
        {"eps": 1.0e-4, "nout": 3},
    )
    cfunc_circle, _ = chunkerfunc(
        lambda t: np.vstack(
            (
                cfunc.circle_ctr[0] + float(cfunc.circle_radius) * np.cos(t),
                cfunc.circle_ctr[1] + float(cfunc.circle_radius) * np.sin(t),
            )
        ),
        {"eps": 1.0e-4, "nout": 3},
    )
    out["chunkerfunc_starfish_r"] = cfunc_starfish.r
    out["chunkerfunc_starfish_nout_r"] = cfunc_starfish_nout.r
    out["chunkerfunc_bymode_r"] = cfunc_bymode.r
    out["chunkerfunc_bymode_reversed_r"] = cfunc_bymode.reverse().r
    out["chunkerfunc_circle_r"] = cfunc_circle.r
    out["chunkerfunc_circle_refined_r"] = cfunc_circle.refine({"nover": 1}).r

    ccls = fixture.chunkerclassunit
    ccls_chunker = chunker_from_fields(ccls.chunker)
    out["chunkerclass_plus_left_r"] = (ccls.v + ccls_chunker).r
    out["chunkerclass_plus_right_r"] = (ccls_chunker + ccls.v).r
    out["chunkerclass_mat_left_r"] = (ccls.A @ ccls_chunker).r
    out["chunkerclass_scale_left_area"] = np.asarray((float(ccls.s) * ccls_chunker).area())
    out["chunkerclass_scale_right_area"] = np.asarray((ccls_chunker * float(ccls.s)).area())

    cfit = fixture.chunkerfit
    cfit_r = curves.bymode(cfit.tt, cfit.modes)[0]
    cfit_closed = chunkerfit(cfit_r, {"ifclosed": True, "cparams": {"eps": 1.0e-6}, "pref": {"k": 16}})
    cfit_open = chunkerfit(cfit_r[:, :10], {"ifclosed": False, "cparams": {"eps": 1.0e-6}, "pref": {"k": 16}})
    out["chunkerfit_r"] = cfit_r
    out["chunkerfit_closed_ier"] = np.asarray(cfit_closed.checkadjinfo())
    out["chunkerfit_open_ier"] = np.asarray(cfit_open.checkadjinfo())

    tcg = fixture.tochunkgraph
    tcg_total = chunker_from_fields(tcg.total)
    tcg_graph = tochunkgraph(tcg_total)
    tcg_manual = chunkgraph(
        tcg.manual_verts,
        tcg.manual_edge2verts,
        [chunker_from_fields(tcg.arc), chunker_from_fields(tcg.circle)],
    )
    out["tochunkgraph_verts"] = tcg_graph.verts
    out["tochunkgraph_edgesendverts"] = tcg_graph.edgesendverts
    out["tochunkgraph_manual_verts"] = tcg_manual.verts
    out["tochunkgraph_manual_edgesendverts"] = tcg_manual.edgesendverts

    slc = fixture.slicegraph
    slc_graph = chunkgraph(slc.verts, np.asarray(slc.edge_2_verts, dtype=int) - 1)
    slc_inner_edges = np.asarray(slc.ichs_inner, dtype=int).reshape(-1) - 1
    slc_lap_d = -2 * kernel("lap", "d")
    slc_full = chunkermat(slc_graph, slc_lap_d)
    slc_inner = slc_graph.slicegraph(slc_inner_edges)
    out["slicegraph_mixed_r"] = slc_graph.slicegraph(np.asarray(slc.ichs_mixed, dtype=int).reshape(-1) - 1).r
    out["slicegraph_inner_sysmat"] = chunkermat(slc_inner, slc_lap_d)
    out["slicegraph_full_inner_sysmat"] = slc_full[np.ix_(np.asarray(slc.idslce, dtype=int).reshape(-1) - 1, np.asarray(slc.idslce, dtype=int).reshape(-1) - 1)]
    out["slicegraph_edgeids_inner"] = slc_graph.edgeids(slc_inner_edges)

    cint2 = fixture.chunkerinterior
    cint2_chunker = chunker_from_fields(cint2.chunker)
    out["chunkerinterior_in"] = chunkerinterior(cint2_chunker, cint2.targs, {"acceleration": "dense"})
    out["chunkerinterior_in_flam"] = chunkerinterior(cint2_chunker, cint2.targs, {"acceleration": "flam", "useproxy": False})
    out["chunkerinterior_in_fmm"] = chunkerinterior(cint2_chunker, cint2.targs, {"acceleration": "fmm"})
    out["chunkerinterior_in_chunker"] = chunkerinterior(cint2_chunker, chunker_from_fields(cint2.inner_chunker), {"acceleration": "fmm"})
    out["chunkerinterior_axis"] = chunkerinterior(
        chunker_from_fields(cint2.axis_chunker),
        cint2.axis_targs,
        {"axissym": True},
    )
    out["chunkerinterior_stress"] = chunkerinterior(
        chunker_from_fields(cint2.stress_chunker),
        [cint2.stress_x, cint2.stress_x],
    )

    cpoly = fixture.chunkerpoly
    cpoly_rounded = chunkerpoly(
        cpoly.verts,
        {"widths": 0.1 * np.ones(np.asarray(cpoly.verts).shape[1]), "eps": 1.0e-8},
        {"k": 16, "dim": 2},
        cpoly.edgevals,
    ).sort()[0]
    cpoly_true = chunkerpoly(
        cpoly.verts,
        {"rounded": False, "depth": 8},
        {"k": 16, "dim": 2},
        cpoly.edgevals,
    ).sort()[0]
    cpoly_open = chunkerpoly(
        cpoly.open_verts,
        {
            "widths": 0.1 * np.ones(np.asarray(cpoly.open_verts).shape[1]),
            "autowidths": True,
            "autowidthsfac": 0.1,
            "ifclosed": False,
            "eps": 1.0e-3,
        },
        {"k": 16, "dim": 2},
    )
    out["chunkerpoly_rounded_ier"] = np.asarray(cpoly_rounded.checkadjinfo())
    out["chunkerpoly_truepoly_ier"] = np.asarray(cpoly_true.checkadjinfo())
    out["chunkerpoly_open_ier"] = np.asarray(cpoly_open.checkadjinfo())

    smth = fixture.smoother
    _, smth_err, smth_err_by_pt = smoother.smooth(smth.verts, {"lam": float(smth.opts.lam), "return_error": True})
    out["smoother_err"] = np.asarray(smth_err)
    out["smoother_err_by_pt"] = smth_err_by_pt

    fs = fixture.flagself
    out["flagself_pairs"] = flagself(fs.srcs, fs.targs)

    fn = fixture.flagnear
    out["flagnear_flags"] = flagnear(chunker_from_fields(fn.chunker), fn.targs, {"fac": float(fn.fac)})

    fr = fixture.flagrect
    fr_chunker = chunker_from_fields(fr.chunker)
    out["flagrect_flags"] = flagnear_rectangle(fr_chunker, fr.targets)
    out["flagrect_grid_flags"] = flagnear_rectangle_grid(fr_chunker, fr.x, fr.y)

    h2g = fixture.helm2d_green
    h2g_val, h2g_grad, h2g_hess = helm2d.green(h2g.zk, h2g.src, h2g.trg)
    out["helm2d_green_val"] = h2g_val
    out["helm2d_green_grad"] = h2g_grad
    out["helm2d_green_hess"] = h2g_hess

    kop = fixture.kernelop
    src = pointinfo_from_mat(kop.src)
    targ = pointinfo_from_mat(kop.targ)
    skern = kernel("lap", "s")
    dkern = kernel("helm", "d", 1)
    fkern1 = kernel([[skern], [dkern]])
    scalar = kop.a
    out["kernelop_skern"] = skern(src, targ)
    out["kernelop_dkern"] = dkern(src, targ)
    out["kernelop_fkern1"] = fkern1(src, targ)
    out["kernelop_fkern2"] = (scalar * fkern1)(src, targ)
    out["kernelop_fkern3"] = (fkern1 / scalar)(src, targ)
    out["kernelop_nkern"] = (-skern)(src, targ)
    out["kernelop_ckern1"] = (skern + dkern)(src, targ)
    out["kernelop_ckern2"] = (skern - dkern)(src, targ)
    out["kernelop_conj_dkern"] = dkern.conj()(src, targ)

    sdtr = fixture.stokes_dtrac
    src = pointinfo_from_mat(sdtr.srcinfo)
    targ = pointinfo_from_mat(sdtr.targinfo)
    strengths = np.asarray(sdtr.strengths).reshape(-1)
    kt = kernel("stok", "dtrac", float(sdtr.mu))(src, targ) @ strengths
    kg = kernel("stok", "dgrad", float(sdtr.mu))(src, targ) @ strengths
    kp = kernel("stok", "dpres", float(sdtr.mu))(src, targ) @ strengths
    du = kg.reshape(2, 2, 1, order="F")
    eu = du + np.transpose(du, (1, 0, 2))
    reconstructed = np.zeros(2)
    reconstructed[0::2] = -kp * targ.n[0] + (eu[0, 0] * targ.n[0] + eu[0, 1] * targ.n[1]) * float(sdtr.mu)
    reconstructed[1::2] = -kp * targ.n[1] + (eu[0, 1] * targ.n[0] + eu[1, 1] * targ.n[1]) * float(sdtr.mu)
    out["stokes_dtrac_Kt"] = kt
    out["stokes_dtrac_Kg"] = kg
    out["stokes_dtrac_Kp"] = kp
    out["stokes_dtrac_reconstructed"] = reconstructed

    stok = fixture.chunkermat_stok2d
    stok_chunker = chunker_from_fields(stok.chunker)
    stok_mu = float(stok.mu)
    stok_coefs = np.asarray(stok.coefs).reshape(-1, order="F")
    stok_sources = PointInfo(r=point_array(stok.sources), n=point_array(stok.sources_n))
    stok_targets = PointInfo(r=point_array(stok.targets), n=point_array(stok.targets_n))
    stok_strengths = np.asarray(stok.strengths).reshape(-1, order="F")
    stok_cvel = kernel("stok", "cvel", stok_mu, stok_coefs)
    stok_d = kernel("stok", "d", stok_mu)
    stok_D = chunkermat(stok_chunker, stok_cvel)
    stok_sys = -0.5 * np.eye(stok_D.shape[0]) + stok_D + stok_chunker.normonesmat() / np.sum(stok_chunker.wts)
    stok_rhs = (stok_d(stok_sources, pointinfo_from_chunker(stok_chunker)) @ stok_strengths).reshape(-1, order="F")
    stok_sol = np.linalg.solve(stok_sys, stok_rhs)
    out["chunkermat_stok2d_D"] = stok_D
    out["chunkermat_stok2d_sys"] = stok_sys
    out["chunkermat_stok2d_sol"] = stok_sol
    out["chunkermat_stok2d_Dsol"] = chunkerkerneval(
        stok_chunker,
        stok_cvel,
        np.asarray(stok.sol).reshape(-1, order="F"),
        stok_targets,
        {"acceleration": "fmm", "eps": 1e-11},
    )
    out["chunkermat_stok2d_Ssys"] = chunkerkernevalmat(stok_chunker, kernel("stok", "svel", stok_mu), stok_targets)

    stoktr = fixture.chunkermat_stok_traction
    stoktr_chunker = chunker_from_fields(stoktr.chunker)
    stoktr_mu = float(stoktr.mu)
    stoktr_coefs = np.asarray(stoktr.coefs).reshape(-1, order="F")
    stoktr_sources = PointInfo(r=point_array(stoktr.sources), n=point_array(stoktr.sources_n))
    stoktr_targets = point_array(stoktr.targets)
    stoktr_strengths = np.asarray(stoktr.strengths).reshape(-1, order="F")
    stoktr_strac = kernel("stok", "strac", stoktr_mu, stoktr_coefs)
    stoktr_dtrac = kernel("stok", "dtrac", stoktr_mu)
    stoktr_D = chunkermat(stoktr_chunker, stoktr_strac)
    stoktr_sys = 0.5 * np.eye(stoktr_D.shape[0]) + stoktr_D
    stoktr_rhs = (stoktr_dtrac(stoktr_sources, pointinfo_from_chunker(stoktr_chunker)) @ stoktr_strengths).reshape(-1, order="F")
    stoktr_sol = np.linalg.solve(stoktr_sys, stoktr_rhs)
    out["chunkermat_stok_traction_D"] = stoktr_D
    out["chunkermat_stok_traction_sys"] = stoktr_sys
    out["chunkermat_stok_traction_sol"] = stoktr_sol
    out["chunkermat_stok_traction_Dsol"] = chunkerkerneval(
        stoktr_chunker,
        kernel("stok", "svel", stoktr_mu),
        np.asarray(stoktr.sol).reshape(-1, order="F"),
        stoktr_targets,
        {"acceleration": "fmm", "eps": 1e-11},
    )

    cqa = fixture.chunkermat_quadadap
    cqa_chunker = chunker_from_fields(cqa.chunker)
    cqa_kern = kernel("helm", "d", cqa.zk)
    out["chunkermat_quadadap_ggq"] = chunkermat(cqa_chunker, cqa_kern)
    out["chunkermat_quadadap_adap"] = quadadap.buildmat(cqa_chunker, cqa_kern, cqa_kern.opdims, {"sing": "log", "robust": False})
    cqac = fixture.chunkermat_quadadap_closetotouching
    cqac_chunker = chunker_from_fields(cqac.chunker)
    cqac_kern = kernel("lap", "c", [1.0, float(cqac.eta)])
    out["chunkermat_quadadap_closetouching_mata_probe"] = (
        chunkermat(cqac_chunker, cqac_kern, {"adaptive_correction": True, "robust": True})
        @ np.asarray(cqac.sysa_probe_rhs)
    )
    out["chunkermat_quadadap_closetouching_mato_probe"] = (
        chunkermat(cqac_chunker, cqac_kern)
        @ np.asarray(cqac.sysa_probe_rhs)
    )
    cma = fixture.chunkermatapply_scalar
    cma_chunker = chunker_from_fields(cma.chunker)
    cma_kern = kernel("lap", "d")
    out["chunkermatapply_scalar_apply"] = chunkermatapply(
        cma_chunker,
        cma_kern,
        np.asarray(cma.dens).reshape(-1, order="F"),
    )
    cmav = fixture.chunkermatapply_vector
    cmav_chunker = chunker_from_fields(cmav.chunker)
    cmav_kern = transmission_all_kernel_from_fixture(cmav)
    cmav_bdry_data = transmission_point_source_boundary_data(cmav_chunker, cmav)
    cmav_sysmat = chunkermat(cmav_chunker, cmav_kern)
    out["chunkermatapply_vector_bdry_data"] = cmav_bdry_data
    out["chunkermatapply_vector_apply"] = chunkermatapply(cmav_chunker, cmav_kern, cmav_bdry_data)
    out["chunkermatapply_vector_probe"] = (np.eye(cmav_sysmat.shape[0], dtype=complex) + cmav_sysmat) @ np.asarray(cmav.probe)
    cmag = fixture.chunkermatapply_graph_scalar
    cmag_graph = chunkermatapply_graph_from_fixture(cmag)
    cmag_kern = -2 * kernel("lap", "d")
    cmag_src = PointInfo(r=point_array(cmag.sources))
    cmag_dens = (kernel("lap", "s")(cmag_src, pointinfo_from_chunker(cmag_graph)) * float(cmag.strengths)).reshape(-1, order="F")
    cmag_sysmat = chunkermat(cmag_graph, cmag_kern)
    out["chunkermatapply_graph_scalar_dens"] = cmag_dens
    out["chunkermatapply_graph_scalar_apply"] = chunkermatapply(cmag_graph, cmag_kern, cmag_dens)
    out["chunkermatapply_graph_scalar_probe"] = (np.eye(cmag_graph.npt) + cmag_sysmat) @ np.asarray(cmag.probe)
    cmavg = fixture.chunkermatapply_graph_vector
    cmavg_graph = chunkermatapply_graph_from_fixture(cmavg)
    cmavg_kern = transmission_all_kernel_from_fixture(cmavg)
    cmavg_bdry_data = transmission_point_source_boundary_data(cmavg_graph, cmavg)
    cmavg_sysmat = chunkermat(cmavg_graph, cmavg_kern)
    out["chunkermatapply_graph_vector_bdry_data"] = cmavg_bdry_data
    out["chunkermatapply_graph_vector_apply"] = chunkermatapply(cmavg_graph, cmavg_kern, cmavg_bdry_data)
    out["chunkermatapply_graph_vector_probe"] = (np.eye(cmavg_sysmat.shape[0], dtype=complex) + cmavg_sysmat) @ np.asarray(cmavg.probe)
    sk = fixture.singularkernel
    sk_chunker = chunker_from_fields(sk.chunker)
    sk_probe = np.asarray(sk.probe)
    out["singularkernel_sprime_probe"] = chunkermat(sk_chunker, kernel("lap", "sp")) @ sk_probe
    out["singularkernel_stau_probe"] = chunkermat(sk_chunker, kernel("lap", "stau")) @ sk_probe
    out["singularkernel_d_probe"] = chunkermat(sk_chunker, kernel("lap", "d")) @ sk_probe
    out["singularkernel_dprime_probe"] = chunkermat(sk_chunker, kernel("lap", "dp")) @ sk_probe
    return out


def main() -> None:
    PYTHON_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PYTHON_SNAPSHOT, **build_snapshot())
    print(f"Wrote {PYTHON_SNAPSHOT}")


if __name__ == "__main__":
    main()
