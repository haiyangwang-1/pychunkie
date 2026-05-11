from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from chunkie import Chunker, chunkerfit, chunkerfunc, chunkerfuncuni, chunkerintegral, kernel, lege
from chunkie.chnk import curves, flagnear, flagnear_rectangle, flagnear_rectangle_grid, flagself, helm2d, spcl
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
