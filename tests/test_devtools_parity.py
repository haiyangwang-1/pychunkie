from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from chunkie import Chunker, kernel, lege
from chunkie.chnk import spcl
from chunkie.operators import PointInfo


GOLDEN = Path(__file__).parent / "golden"


def load_devtools_easy():
    return loadmat(GOLDEN / "devtools_easy.mat", squeeze_me=True, struct_as_record=False)["devtools_easy"]


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


@pytest.mark.xfail(strict=True, reason="Python arclengthfun does not yet reset arclength per merged component.")
def test_arclengthfun_merged_components_devtools_output_matches_matlab():
    fixture = load_devtools_easy().arclengthfun
    chnkr = chunker_from_fields(fixture.chunker_merged)

    np.testing.assert_allclose(chnkr.arclengthfun(), fixture.s_merged, atol=1e-13)


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
