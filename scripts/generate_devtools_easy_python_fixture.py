"""Generate Python-side snapshots for the easy MATLAB devtools fixture.

The pytest suite recomputes these values live. This script is for manual
side-by-side inspection when regenerating MATLAB data from
``scripts/matlab/generate_devtools_easy_fixture.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from chunkie import Chunker, chunkerfunc, chunkerfuncuni, chunkerintegral, kernel, lege
from chunkie.chnk import curves, flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself, helm2d, spcl
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

    ccls = fixture.chunkerclassunit
    ccls_chunker = chunker_from_fields(ccls.chunker)
    out["chunkerclass_plus_left_r"] = (ccls.v + ccls_chunker).r
    out["chunkerclass_plus_right_r"] = (ccls_chunker + ccls.v).r
    out["chunkerclass_mat_left_r"] = (ccls.A @ ccls_chunker).r
    out["chunkerclass_scale_left_area"] = np.asarray((float(ccls.s) * ccls_chunker).area())
    out["chunkerclass_scale_right_area"] = np.asarray((ccls_chunker * float(ccls.s)).area())

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
    return out


def main() -> None:
    PYTHON_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PYTHON_SNAPSHOT, **build_snapshot())
    print(f"Wrote {PYTHON_SNAPSHOT}")


if __name__ == "__main__":
    main()
