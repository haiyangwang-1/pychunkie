"""Generate Python-side snapshots for the easy MATLAB devtools fixture.

The pytest suite recomputes these values live. This script is for manual
side-by-side inspection when regenerating MATLAB data from
``scripts/matlab/generate_devtools_easy_fixture.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from chunkie import Chunker, kernel, lege
from chunkie.chnk import spcl
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
        r=np.asarray(obj.r),
        d=np.asarray(obj.d) if hasattr(obj, "d") else None,
        d2=np.asarray(obj.d2) if hasattr(obj, "d2") else None,
        n=np.asarray(obj.n) if hasattr(obj, "n") else None,
    )


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
    return out


def main() -> None:
    PYTHON_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PYTHON_SNAPSHOT, **build_snapshot())
    print(f"Wrote {PYTHON_SNAPSHOT}")


if __name__ == "__main__":
    main()
