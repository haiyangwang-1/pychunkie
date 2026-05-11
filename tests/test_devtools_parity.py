from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from chunkie import Chunker, kernel, lege
from chunkie.chnk import flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself, helm2d, spcl
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


def sorted_pairs(pairs: np.ndarray) -> np.ndarray:
    arr = np.asarray(pairs, dtype=int)
    if arr.size == 0:
        return arr.reshape(2, 0)
    order = np.lexsort((arr[1], arr[0]))
    return arr[:, order]


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
