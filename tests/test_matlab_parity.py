from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import loadmat

from chunkie import Chunker, chunkerkerneval, chunkerkernevalmat, chunkermat, kernel, lege
from chunkie.chnk import elast2d, helm1d, helm2d, lap2d, quadggq, quadnative, rcip, stok2d


GOLDEN = Path(__file__).parent / "golden"


def load_fixture(name: str):
    path = GOLDEN / name
    if not path.exists():
        pytest.skip(f"optional MATLAB parity fixture is not present: {name}")
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def mat_fields(obj) -> dict:
    return {name: getattr(obj, name) for name in obj._fieldnames}


def pointinfo_dict(obj) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(obj, name)) for name in obj._fieldnames}


def dense_array(obj) -> np.ndarray:
    return obj.toarray() if sparse.issparse(obj) else np.asarray(obj)


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


def assert_chunker_matches_fields(chnkr: Chunker, fields, label: str) -> None:
    np.testing.assert_allclose(chnkr.r, fields.r, atol=1e-13, err_msg=f"{label}: r")
    np.testing.assert_allclose(chnkr.d, fields.d, atol=1e-13, err_msg=f"{label}: d")
    np.testing.assert_allclose(chnkr.d2, fields.d2, atol=1e-13, err_msg=f"{label}: d2")
    np.testing.assert_allclose(chnkr.n, fields.n, atol=1e-13, err_msg=f"{label}: n")
    np.testing.assert_allclose(chnkr.wts, fields.wts, atol=1e-13, err_msg=f"{label}: wts")
    np.testing.assert_array_equal(chnkr.adj, fields.adj, err_msg=f"{label}: adj")
    np.testing.assert_allclose(chnkr.area(), fields.area, atol=1e-13, err_msg=f"{label}: area")
    np.testing.assert_allclose(chnkr.chunklen(), fields.chunklen, atol=1e-13, err_msg=f"{label}: chunklen")


def test_extended_legendre_helpers_match_matlab_fixture():
    fixture = load_fixture("lege_extended.mat")
    k = int(fixture["k_ext"])
    xs = fixture["xs_ext"]
    coeff = fixture["coeff_ext"]
    x, w, u, v = lege.exps(k)
    pols, ders = lege.pols(xs, k - 1)
    matrin, *_ = lege.matrin(k, xs, u)
    intmat, *_ = lege.intmat(k, u, v)

    np.testing.assert_allclose(x, fixture["x_ext"], atol=1e-14)
    np.testing.assert_allclose(w, fixture["w_ext"], atol=1e-14)
    np.testing.assert_allclose(u, fixture["u_ext"], atol=1e-13)
    np.testing.assert_allclose(v, fixture["v_ext"], atol=1e-13)
    np.testing.assert_allclose(pols, fixture["pol_ext"], atol=1e-13)
    np.testing.assert_allclose(ders, fixture["der_ext"], atol=1e-12)
    np.testing.assert_allclose(matrin, fixture["matrin_ext"], atol=1e-13)
    np.testing.assert_allclose(intmat, fixture["intmat_ext"], atol=1e-13)
    np.testing.assert_allclose(lege.exev(xs, coeff), fixture["exev_ext"], atol=1e-13)
    np.testing.assert_allclose(lege.intpol(coeff, "true"), fixture["intpol_true_ext"], atol=1e-13)
    np.testing.assert_allclose(lege.intpol(coeff, "original"), fixture["intpol_original_ext"], atol=1e-13)
    np.testing.assert_allclose(lege.derpol(coeff), fixture["derpol_ext"], atol=1e-13)
    np.testing.assert_allclose(lege.barywts(k, x), fixture["barywts_ext"], atol=1e-13)
    np.testing.assert_allclose(
        lege.bernstein_ellipse(int(fixture["bernstein_ntheta_ext"]), float(fixture["bernstein_rho_ext"])),
        fixture["bernstein_ellipse_ext"],
        atol=1e-14,
    )
    polsum_pol, polsum_der, polsum_tot = lege.polsum(xs, int(fixture["polsum_n_ext"]))
    np.testing.assert_allclose(polsum_pol, fixture["polsum_pol_ext"], atol=1e-13)
    np.testing.assert_allclose(polsum_der, fixture["polsum_der_ext"], atol=1e-12)
    np.testing.assert_allclose(polsum_tot, fixture["polsum_tot_ext"], atol=1e-13)
    tayl_pol, tayl_der = lege.tayl(
        fixture["tayl_pol0_ext"],
        fixture["tayl_der0_ext"],
        fixture["tayl_x_ext"],
        fixture["tayl_h_ext"],
        int(fixture["tayl_n_ext"]),
        int(fixture["tayl_k_ext"]),
    )
    np.testing.assert_allclose(tayl_pol, fixture["tayl_pol_ext"], atol=1e-13)
    np.testing.assert_allclose(tayl_der, fixture["tayl_der_ext"], atol=1e-12)


def test_chunker_geometry_and_transforms_match_matlab_fixture():
    ops = load_fixture("chunker_ops.mat")["chunker_ops"]
    base = chunker_from_fields(ops.base)

    assert_chunker_matches_fields(base, ops.base, "base chunker")
    transformed = base.move([0.35, -0.2], [0.1, 0.2], 0.45, 1.3).transform(ops.mat)
    assert_chunker_matches_fields(transformed, ops.transformed, "transformed chunker")

    np.testing.assert_allclose(dense_array(base.diffmat()), dense_array(ops.diffmat1), atol=1e-13)
    np.testing.assert_allclose(dense_array(base.diffmat(2)), dense_array(ops.diffmat2), atol=1e-13)
    np.testing.assert_allclose(base.intmat(), ops.intmat, atol=1e-13)
    np.testing.assert_allclose(base.onesmat(), ops.onesmat, atol=1e-13)
    np.testing.assert_allclose(base.normonesmat(), ops.normonesmat, atol=1e-13)
    np.testing.assert_allclose(base.centroids(), ops.centroids, atol=1e-13)


@pytest.mark.parametrize(
    "kind",
    [
        "s",
        "d",
        "sp",
        "stau",
        "hilb",
        "sgrad",
        "dgrad",
        "dp",
        "c",
        "cp",
        "cgrad",
    ],
)
def test_laplace_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    coefs = fixture["lap_coefs"] if kind in {"c", "cp", "cgrad"} else None

    actual = lap2d.kern(src, targ, kind, coefs)
    expected = getattr(fixture["lap"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize(
    "kind",
    [
        "s",
        "d",
        "sp",
        "stau",
        "sgrad",
        "dgrad",
        "dp",
        "c",
        "cp",
    ],
)
def test_helmholtz_2d_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    coefs = fixture["helm_coefs"] if kind in {"c", "cp"} else None

    actual = helm2d.kern(fixture["helm_zk"], src, targ, kind, coefs)
    expected = getattr(fixture["helm2d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "kind",
    ["s", "d", "sp", "stau", "dp", "c", "cp", "c2trans", "all", "trans_rep", "trans_rep_prime", "trans_rep_grad"],
)
def test_helmholtz_1d_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    if kind == "all":
        coefs = fixture["helm_all_coefs"]
    elif kind in {"c", "cp", "c2trans", "trans_rep", "trans_rep_prime", "trans_rep_grad"}:
        coefs = fixture["helm_coefs"]
    else:
        coefs = None

    actual = helm1d.kern(fixture["helm1d_zk"], src, targ, kind, coefs)
    expected = getattr(fixture["helm1d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["s", "spres", "strac", "d", "dpres", "dtrac", "sgrad", "dgrad", "c"])
def test_stokes_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    coefs = fixture["stok_coefs"] if kind == "c" else None

    actual = stok2d.kern(fixture["stok_mu"], src, targ, kind, coefs)
    expected = getattr(fixture["stok2d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", ["s", "strac", "d", "dalt"])
def test_elasticity_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])

    actual = elast2d.kern(fixture["elast_lam"], fixture["elast_mu"], src, targ, kind)
    expected = getattr(fixture["elast2d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


def test_dense_native_operator_paths_match_matlab_fixture():
    fixture = load_fixture("operator_parity.mat")["operator_parity"]
    chnkr = chunker_from_fields(fixture.chunker)
    density_scalar = np.asarray(fixture.density_scalar).reshape(-1, order="F")
    density_stokes = np.asarray(fixture.density_stokes).reshape(-1, order="F")
    targets = np.asarray(fixture.targets)
    lap_d = lambda s, t: lap2d.kern(s, t, "d")
    lap_s = lambda s, t: lap2d.kern(s, t, "s")
    stok_d = lambda s, t: stok2d.kern(fixture.stok_mu, s, t, "d")

    lap_d_mat = chunkermat(chnkr, lap_d)
    np.testing.assert_allclose(lap_d_mat, fixture.lap_d_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lap_d_mat @ density_scalar, fixture.lap_d_apply, rtol=1e-12, atol=1e-13)

    evalmat = chunkerkernevalmat(chnkr, lap_s, targets)
    values = chunkerkerneval(chnkr, lap_s, density_scalar, targets).reshape(-1, order="F")
    np.testing.assert_allclose(evalmat, fixture.lap_s_evalmat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(values, np.asarray(fixture.lap_s_eval).reshape(-1, order="F"), rtol=1e-12, atol=1e-13)

    stok_d_mat = quadnative.buildmat(chnkr, stok_d, (2, 2))
    np.testing.assert_allclose(stok_d_mat, fixture.stok_d_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(stok_d_mat @ density_stokes, fixture.stok_d_apply, rtol=1e-12, atol=1e-13)


def test_quadggq_special_quadrature_matches_matlab_fixture():
    fixture = load_fixture("quadggq.mat")["quadggq"]
    chnkr = chunker_from_fields(fixture.chunker)

    log_aux = quadggq.setup(chnkr.k, "log")
    np.testing.assert_allclose(log_aux.xs1, fixture.log_xs1, atol=0.0)
    np.testing.assert_allclose(log_aux.wts1, fixture.log_wts1, atol=0.0)
    for actual, expected in zip(log_aux.xs0, fixture.log_xs0):
        np.testing.assert_allclose(actual, expected, atol=0.0)
    for actual, expected in zip(log_aux.wts0, fixture.log_wts0):
        np.testing.assert_allclose(actual, expected, atol=0.0)

    pv_aux = quadggq.setup(chnkr.k, "pv")
    hs_aux = quadggq.setup(chnkr.k, "hs")
    for actual, expected in zip(pv_aux.xs0, fixture.pv_xs0):
        np.testing.assert_allclose(actual, expected, atol=0.0)
    for actual, expected in zip(pv_aux.wts0, fixture.pv_wts0):
        np.testing.assert_allclose(actual, expected, atol=0.0)
    for actual, expected in zip(hs_aux.xs0, fixture.hs_xs0):
        np.testing.assert_allclose(actual, expected, atol=0.0)
    for actual, expected in zip(hs_aux.wts0, fixture.hs_wts0):
        np.testing.assert_allclose(actual, expected, atol=0.0)

    lap_s = kernel("lap", "s")
    lap_sgrad = kernel("lap", "sgrad")
    lap_dgrad = kernel("lap", "dgrad")
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_s, lap_s.opdims, "log"), fixture.log_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_sgrad, lap_sgrad.opdims, "pv"), fixture.pv_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_dgrad, lap_dgrad.opdims, "hs"), fixture.hs_mat, rtol=2e-7, atol=5e-8)

    skipped = quadggq.buildmat(chnkr, lap_s, lap_s.opdims, "log", ilist=[0, 1])
    np.testing.assert_array_equal(np.isinf(skipped), np.isinf(fixture.log_mat_skip))
    finite = np.isfinite(fixture.log_mat_skip)
    np.testing.assert_allclose(skipped[finite], fixture.log_mat_skip[finite], rtol=1e-12, atol=1e-13)


def test_rcip_recursive_compression_matches_matlab_fixture():
    fixture = load_fixture("rcip.mat")["rcip_fixture"]
    edge1 = chunker_from_fields(fixture.edge1)
    edge2 = chunker_from_fields(fixture.edge2)
    sbclmat, sbcrmat, lvmat, rvmat, u = rcip.shiftedlegbasismats(edge1.k)

    np.testing.assert_allclose(sbclmat, fixture.sbclmat, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(sbcrmat, fixture.sbcrmat, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(lvmat, fixture.lvmat, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(rvmat, fixture.rvmat, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(u, fixture.u, rtol=1e-13, atol=1e-13)

    rmat, saved = rcip.Rcompchunk(
        [edge1, edge2],
        np.asarray(fixture.iedgechunks0, dtype=int),
        kernel("lap", "d"),
        1,
        fixture.vert0,
        opts={"nsub": 2, "rcip_savedepth": 2},
    )

    np.testing.assert_allclose(rmat, fixture.R, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(saved.R[-1], fixture.saved_R_final, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(saved.MAT[-1], fixture.saved_MAT_last, rtol=1e-12, atol=1e-13)
    assert_chunker_matches_fields(saved.chnkrlocals[-1], fixture.saved_local_last, "rcip saved local chunker")

    rhohatinterp, srcinfo, wts = rcip.rhohatInterp(fixture.rhohat, saved, 2)
    for iedge in range(saved.nedge):
        np.testing.assert_allclose(rhohatinterp[iedge], fixture.rhohatinterp[iedge], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].r, fixture.srcinfo[iedge].r, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].d, fixture.srcinfo[iedge].d, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].d2, fixture.srcinfo[iedge].d2, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].n, fixture.srcinfo[iedge].n, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(wts[iedge], fixture.wts[iedge], rtol=1e-12, atol=1e-13)
