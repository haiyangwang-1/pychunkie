import numpy as np
import pytest
from scipy import sparse

from chunkie import (
    Chunker,
    ChunkerFLAMMatrix,
    ChunkerFMMMatrix,
    PointInfo,
    chunkerinterior,
    chunkerintegral,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    chunkerpref,
    kernel,
    lege,
)
from chunkie.kernels import biharmonic as biharm2d
from chunkie.kernels import elasticity as elast2d
from chunkie.kernels import helmholtz as helm2d
from chunkie.kernels import helmholtz_1d as helm1d
from chunkie.kernels import laplace as lap2d
from chunkie.kernels import stokes as stok2d

pointinfo = PointInfo.from_any
from chunkie.quadrature import adaptive as quadadap
from chunkie.quadrature import ggq as quadggq
from chunkie.quadrature import native as quadnative
from chunkie.quadrature import rcip
from _fixture_generation import assert_chunker_matches_fields, chunker_from_fields, load_generated_mat_fixture


def load_fixture(name: str):
    return load_generated_mat_fixture(name, squeeze_me=True, struct_as_record=False)


def mat_fields(obj) -> dict:
    return {name: getattr(obj, name) for name in obj._fieldnames}


def pointinfo_dict(obj) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(obj, name)) for name in obj._fieldnames}


def dense_array(obj) -> np.ndarray:
    return obj.toarray() if sparse.issparse(obj) else np.asarray(obj)


def matlab_string(value) -> str:
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.reshape(-1).astype(str))
    return str(value)


def assert_cell_arrays_allclose(actual, expected, *, rtol: float = 1e-13, atol: float = 1e-13, label: str) -> None:
    assert len(actual) == len(expected), f"{label}: cell count"
    for idx, (actual_cell, expected_cell) in enumerate(zip(actual, expected)):
        np.testing.assert_allclose(
            np.asarray(actual_cell),
            np.asarray(expected_cell),
            rtol=rtol,
            atol=atol,
            err_msg=f"{label}: cell {idx}",
        )


def block_22_kernel(src, targ):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    base = 1.0 + dx**2 + 0.5 * dy**2
    ntarget, nsource = base.shape
    out = np.zeros((2 * ntarget, 2 * nsource), dtype=base.dtype)
    out[0::2, 0::2] = base
    out[1::2, 1::2] = 2.0 + 0.25 * base
    out[0::2, 1::2] = 0.1 * dx
    out[1::2, 0::2] = -0.2 * dy
    return out


block_22_kernel.opdims = (2, 2)


def block_21_kernel(src, targ):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    base = 0.75 + 0.2 * dx - 0.1 * dy + 0.05 * dx * dy
    ntarget, nsource = base.shape
    out = np.zeros((2 * ntarget, nsource), dtype=base.dtype)
    out[0::2, :] = base
    out[1::2, :] = 1.25 - 0.15 * dx + 0.3 * dy**2
    return out


block_21_kernel.opdims = (2, 1)


def block_12_kernel(src, targ):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    ntarget, nsource = dx.shape
    out = np.zeros((ntarget, 2 * nsource), dtype=dx.dtype)
    out[:, 0::2] = -0.4 + 0.35 * dx**2 + 0.1 * dy
    out[:, 1::2] = 0.6 + 0.2 * dx - 0.25 * dy
    return out


block_12_kernel.opdims = (1, 2)


def block_scalar_kernel(src, targ):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    return 1.0 + dx**2 + 0.5 * dy**2


block_scalar_kernel.opdims = (1, 1)


KERNEL_OBJECT_CASES = [
    ("lap_s", lambda f: kernel("lap", "s")),
    ("lap_d", lambda f: kernel("lap", "d")),
    ("lap_sp", lambda f: kernel("lap", "sp")),
    ("lap_stau", lambda f: kernel("lap", "stau")),
    ("lap_sgrad", lambda f: kernel("lap", "sgrad")),
    ("lap_dgrad", lambda f: kernel("lap", "dgrad")),
    ("lap_dp", lambda f: kernel("lap", "dp")),
    ("lap_c", lambda f: kernel("lap", "c", f["lap_coefs"])),
    ("lap_cp", lambda f: kernel("lap", "cp", f["lap_coefs"])),
    ("lap_cgrad", lambda f: kernel("lap", "cgrad", f["lap_coefs"])),
    ("helm2d_s", lambda f: kernel("helm", "s", f["helm_zk"])),
    ("helm2d_d", lambda f: kernel("helm", "d", f["helm_zk"])),
    ("helm2d_sp", lambda f: kernel("helm", "sp", f["helm_zk"])),
    ("helm2d_dp", lambda f: kernel("helm", "dp", f["helm_zk"])),
    ("helm2d_c", lambda f: kernel("helm", "c", f["helm_zk"], f["helm_coefs"])),
    ("helm2d_cp", lambda f: kernel("helm", "cp", f["helm_zk"], f["helm_coefs"])),
    ("helm1d_s", lambda f: kernel("helm1d", "s", f["helm1d_zk"])),
    ("stok_s", lambda f: kernel("stok", "s", float(f["stok_mu"]))),
    ("stok_spres", lambda f: kernel("stok", "spres", float(f["stok_mu"]))),
    ("stok_strac", lambda f: kernel("stok", "strac", float(f["stok_mu"]))),
    ("stok_sgrad", lambda f: kernel("stok", "sgrad", float(f["stok_mu"]))),
    ("stok_d", lambda f: kernel("stok", "d", float(f["stok_mu"]))),
    ("stok_dpres", lambda f: kernel("stok", "dpres", float(f["stok_mu"]))),
    ("stok_dtrac", lambda f: kernel("stok", "dtrac", float(f["stok_mu"]))),
    ("stok_dgrad", lambda f: kernel("stok", "dgrad", float(f["stok_mu"]))),
    ("stok_c", lambda f: kernel("stok", "c", float(f["stok_mu"]), f["stok_coefs"])),
    ("stok_cpres", lambda f: kernel("stok", "cpres", float(f["stok_mu"]), f["stok_coefs"])),
    ("stok_ctrac", lambda f: kernel("stok", "ctrac", float(f["stok_mu"]), f["stok_coefs"])),
    ("stok_cgrad", lambda f: kernel("stok", "cgrad", float(f["stok_mu"]), f["stok_coefs"])),
    ("elast_s", lambda f: kernel("elast", "s", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_sgrad", lambda f: kernel("elast", "sgrad", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_strac", lambda f: kernel("elast", "strac", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_d", lambda f: kernel("elast", "d", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_dalt", lambda f: kernel("elast", "dalt", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_dalttrac", lambda f: kernel("elast", "dalttrac", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("elast_daltgrad", lambda f: kernel("elast", "daltgrad", float(f["elast_lam"]), float(f["elast_mu"]))),
    ("zeros_2_3", lambda f: kernel("zero", 2, 3)),
    ("nans_2_3", lambda f: kernel("nan", 2, 3)),
    (
        "custom",
        lambda f: kernel(
            lambda s, t: (1.0 + 2.0j) * np.ones((t.r.shape[1], s.r.shape[1]))
            + 0.1 * (t.r[0, :, None] - s.r[0, None, :])
        ),
    ),
]


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
    rts_x, rts_w = lege.rts(k)
    rts_stab_x, rts_stab_w = lege.rts_stab(k)
    np.testing.assert_allclose(rts_x, fixture["rts_x_ext"], atol=1e-14)
    np.testing.assert_allclose(rts_w, fixture["rts_w_ext"], atol=1e-14)
    np.testing.assert_allclose(rts_stab_x, fixture["rts_stab_x_ext"], atol=1e-14)
    np.testing.assert_allclose(rts_stab_w, fixture["rts_stab_w_ext"], atol=1e-14)
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
    pol_direct, der_direct = lege.pol(fixture["tayl_x_ext"], int(fixture["tayl_n_ext"]))
    np.testing.assert_allclose(pol_direct, fixture["tayl_pol0_ext"], atol=1e-13)
    np.testing.assert_allclose(der_direct, fixture["tayl_der0_ext"], atol=1e-12)
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
    adap_val, adap_maxrec, adap_numint, adap_ier = lege.adapgauss(
        lambda x: x**4,
        float(fixture["adap_poly_a_ext"]),
        float(fixture["adap_poly_b_ext"]),
    )
    np.testing.assert_allclose(adap_val, fixture["adap_poly_val_ext"], atol=1e-13)
    assert adap_maxrec == int(fixture["adap_poly_maxrec_ext"])
    assert adap_numint == int(fixture["adap_poly_numint_ext"])
    assert adap_ier == int(fixture["adap_poly_ier_ext"])


def test_chunker_geometry_and_transforms_match_matlab_fixture():
    ops = load_fixture("chunker_ops.mat")["chunker_ops"]
    base = chunker_from_fields(ops.base)

    assert_chunker_matches_fields(base, ops.base, "base chunker", atol=1e-13)
    transformed = base.move([0.35, -0.2], [0.1, 0.2], 0.45, 1.3).transform(ops.mat)
    assert_chunker_matches_fields(transformed, ops.transformed, "transformed chunker", atol=1e-13)

    np.testing.assert_allclose(dense_array(base.diffmat()), dense_array(ops.diffmat1), atol=1e-13)
    np.testing.assert_allclose(dense_array(base.diffmat(2)), dense_array(ops.diffmat2), atol=1e-13)
    np.testing.assert_allclose(base.intmat(), ops.intmat, atol=1e-13)
    np.testing.assert_allclose(base.onesmat(), ops.onesmat, atol=1e-13)
    np.testing.assert_allclose(base.normonesmat(), ops.normonesmat, atol=1e-13)
    np.testing.assert_allclose(base.centroids(), ops.centroids, atol=1e-13)


def test_chunker_storage_and_data_helpers_match_matlab_fixture():
    ops = load_fixture("chunker_ops.mat")["chunker_ops"]
    pref = chunkerpref(
        {
            "k": int(ops.pref.k),
            "dim": int(ops.pref.dim),
            "nchmax": int(ops.pref.nchmax),
            "nchstor": int(ops.pref.nchstor),
            "verttol": float(ops.pref.verttol),
        }
    )
    assert pref.k == int(ops.pref.k)
    assert pref.dim == int(ops.pref.dim)
    assert pref.nchmax == int(ops.pref.nchmax)
    assert pref.nchstor == int(ops.pref.nchstor)
    assert pref.verttol == float(ops.pref.verttol)

    storage = ops.storage
    chnkr = Chunker(
        {
            "k": int(storage.k),
            "dim": int(storage.dim),
            "nchmax": int(storage.nchmax),
            "nchstor": int(storage.nchstor_initial),
        }
    )
    chnkr.addchunk(int(storage.nchadd))
    chnkr.r = storage.r_input
    chnkr.d = storage.d_input
    chnkr.d2 = storage.d2_input
    chnkr.n = storage.n_input
    chnkr.wts = storage.wts_input
    chnkr.adj = np.asarray(storage.adj_input, dtype=int)
    assert chnkr.checkadjinfo() == int(storage.checkadjinfo)

    chnkr.resize(int(storage.nchstor_resized))
    np.testing.assert_allclose(chnkr.rstor, storage.rstor_resized, atol=0.0)
    np.testing.assert_allclose(chnkr.dstor, storage.dstor_resized, atol=0.0)
    np.testing.assert_allclose(chnkr.d2stor, storage.d2stor_resized, atol=0.0)
    np.testing.assert_allclose(chnkr.nstor, storage.nstor_resized, atol=0.0)
    np.testing.assert_allclose(chnkr.wtsstor, storage.wtsstor_resized, atol=0.0)
    np.testing.assert_array_equal(chnkr.adjstor, np.asarray(storage.adjstor_resized, dtype=int))

    chnkr.makedatarows(2)
    assert chnkr.datadim == int(storage.datadim_after_make)
    chnkr.data = storage.data_input
    chnkr.makedatarows(1)
    assert chnkr.datadim == int(storage.datadim_after_expand)
    np.testing.assert_allclose(chnkr.data, storage.data_after_expand, atol=0.0)
    np.testing.assert_allclose(chnkr.datastor, storage.datastor_after_expand, atol=0.0)
    chnkr.cleardata()
    assert chnkr.datadim == int(storage.datadim_after_clear)
    assert chnkr.data.size == int(storage.data_numel_after_clear)

    base = chunker_from_fields(ops.base)
    copied = base.copy()
    copied.rstor[0, 0, 0] += float(ops.copy_delta)
    np.testing.assert_allclose(base.rstor[0, 0, 0], ops.copy_source_first, atol=0.0)
    np.testing.assert_allclose(copied.rstor[0, 0, 0], ops.copy_mutated_first, atol=0.0)


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
        "cgrad",
        "c2trans",
        "all",
        "trans_rep",
        "trans_rep_prime",
        "trans_rep_grad",
    ],
)
def test_helmholtz_2d_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    if kind == "all":
        coefs = fixture["helm_all_coefs"]
    elif kind in {"c", "cp", "cgrad", "c2trans", "trans_rep", "trans_rep_prime", "trans_rep_grad"}:
        coefs = fixture["helm_coefs"]
    else:
        coefs = None

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


@pytest.mark.parametrize(
    "kind",
    [
        "s",
        "spres",
        "strac",
        "d",
        "dpres",
        "dtrac",
        "sgrad",
        "dgrad",
        "c",
        "cpres",
        "ctrac",
        "cgrad",
    ],
)
def test_stokes_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    coefs = fixture["stok_coefs"] if kind in {"c", "cpres", "ctrac", "cgrad"} else None

    actual = stok2d.kern(fixture["stok_mu"], src, targ, kind, coefs)
    if kind == "cgrad":
        # MATLAB's saved cgrad fixture combines sgrad twice; keep the Python
        # reference tied to MATLAB's individual dgrad/sgrad component blocks.
        expected = coefs[0] * fixture["stok2d"].dgrad + coefs[1] * fixture["stok2d"].sgrad
    else:
        expected = getattr(fixture["stok2d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("kind", ["s", "sgrad", "strac", "d", "dalt", "dalttrac", "daltgrad"])
def test_elasticity_point_kernels_match_matlab_fixture(kind):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])

    actual = elast2d.kern(fixture["elast_lam"], fixture["elast_mu"], src, targ, kind)
    expected = getattr(fixture["elast2d"], kind)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case_name,kernel_factory", KERNEL_OBJECT_CASES)
def test_kernel_objects_match_matlab_fixture(case_name, kernel_factory):
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    expected = getattr(fixture["kernel_objects"], case_name)
    expected_meta = expected.meta
    actual_kernel = kernel_factory(fixture)
    src_info = pointinfo(src)
    targ_info = pointinfo(targ)

    if case_name != "elast_sgrad":
        assert actual_kernel.opdims == tuple(np.asarray(expected_meta.opdims, dtype=int).reshape(-1))
    assert actual_kernel.sing == matlab_string(expected_meta.sing)
    assert bool(actual_kernel.iszero) == bool(np.asarray(expected_meta.iszero).item())
    assert bool(actual_kernel.isnan) == bool(np.asarray(expected_meta.isnan).item())

    actual = actual_kernel(src_info, targ_info)
    if np.isnan(np.asarray(expected.eval)).all():
        assert np.isnan(actual).all()
    else:
        np.testing.assert_allclose(actual, expected.eval, rtol=1e-11, atol=1e-12)


def test_kernel_algebra_and_interleave_match_matlab_fixture():
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    expected = fixture["kernel_algebra"]
    lap_d = kernel("lap", "d")
    lap_s = kernel("lap", "s")
    helm_s = kernel("helm", "s", fixture["helm_zk"])

    cases = {
        "add": 2.0 * lap_d + lap_s,
        "sub": lap_d - lap_s,
        "neg": -lap_d,
        "div": helm_s / 2.0,
        "conj": helm_s.conj(),
        "interleave": kernel([[lap_d, -lap_s], [lap_s, kernel("zero")]]),
    }
    for name, actual_kernel in cases.items():
        np.testing.assert_allclose(actual_kernel(src, targ), getattr(expected, name), rtol=1e-11, atol=1e-12)

    meta = expected.interleave_meta
    mixed = cases["interleave"]
    assert mixed.opdims == tuple(np.asarray(meta.opdims, dtype=int).reshape(-1))
    assert mixed.sing == matlab_string(meta.sing)


def test_green_helpers_match_matlab_fixture():
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    greens = fixture["greens"]

    lap_val, lap_grad, lap_hess = lap2d.green(src["r"], targ["r"])
    np.testing.assert_allclose(lap_val, greens.lap.val, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(lap_grad, greens.lap.grad, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(lap_hess, greens.lap.hess, rtol=1e-13, atol=1e-13)

    helm_val, helm_grad, helm_hess = helm2d.green(fixture["helm_zk"], src["r"], targ["r"])
    np.testing.assert_allclose(helm_val, greens.helm2d.val, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(helm_grad, greens.helm2d.grad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(helm_hess, greens.helm2d.hess, rtol=1e-12, atol=1e-13)

    helm1d_val, helm1d_grad, helm1d_hess = helm1d.green(fixture["helm1d_zk"], src["r"], targ["r"])
    np.testing.assert_allclose(helm1d_val, greens.helm1d.val, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(helm1d_grad, greens.helm1d.grad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(helm1d_hess, greens.helm1d.hess, rtol=1e-12, atol=1e-13)

    sweep = greens.helm1d_sweep
    actual_sweep = helm1d.sweep(sweep.uin, np.asarray(sweep.inds, dtype=int) - 1, sweep.ts, sweep.wts, fixture["helm1d_zk"])
    np.testing.assert_allclose(actual_sweep, sweep.out, rtol=1e-12, atol=1e-13)


def test_biharmonic_helpers_match_matlab_bhgreen_fixture():
    fixture = load_fixture("kernel_pointinfo.mat")
    src = pointinfo_dict(fixture["srcinfo"])
    targ = pointinfo_dict(fixture["targinfo"])
    expected = fixture["biharm2d"]

    val, grad, hess, lap = biharm2d.green(src["r"], targ["r"])
    np.testing.assert_allclose(val, expected.green.val, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(grad, expected.green.grad, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(hess, expected.green.hess, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(lap, expected.green.lap, rtol=1e-12, atol=1e-13)

    for field, selector in [
        ("s", "s"),
        ("lap", "lap"),
        ("d", "d"),
        ("sp", "sp"),
        ("sgrad", "sgrad"),
        ("shess", "shess"),
    ]:
        np.testing.assert_allclose(
            biharm2d.kern(src, targ, selector),
            getattr(expected.kern, field),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            kernel("biharm", selector)(src, targ),
            getattr(expected.kern, field),
            rtol=1e-12,
            atol=1e-13,
        )


def test_dense_native_operator_paths_match_matlab_fixture():
    fixture = load_fixture("operator_parity.mat")["operator_parity"]
    chnkr = chunker_from_fields(fixture.chunker)
    density_scalar = np.asarray(fixture.density_scalar).reshape(-1, order="F")
    density_stokes = np.asarray(fixture.density_stokes).reshape(-1, order="F")
    targets = np.asarray(fixture.targets)
    lap_d = lambda s, t: lap2d.kern(s, t, "d")
    lap_s = lambda s, t: lap2d.kern(s, t, "s")
    stok_d = lambda s, t: stok2d.kern(fixture.stok_mu, s, t, "d")
    smooth = kernel(
        lambda s, t: (t.r[0, :, None] - s.r[0, None, :]) ** 2
        + 0.5 * (t.r[1, :, None] - s.r[1, None, :]) ** 2
    )

    srcinfo = pointinfo(chnkr)
    np.testing.assert_allclose(srcinfo.r, fixture.pointinfo.r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(srcinfo.d, fixture.pointinfo.d, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(srcinfo.d2, fixture.pointinfo.d2, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(srcinfo.n, fixture.pointinfo.n, rtol=1e-13, atol=1e-13)

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

    smooth_mat = chunkermat(chnkr, smooth)
    np.testing.assert_allclose(smooth_mat, fixture.smooth_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(chunkermatapply(chnkr, smooth, density_scalar), fixture.smooth_apply, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(chunkermat(chnkr, kernel("zero")), fixture.zero_mat, atol=0.0)

    np.testing.assert_allclose(chunkerintegral(chnkr, density_scalar), fixture.integral_values, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        chunkerintegral(chnkr, lambda r: r[0] ** 2 + 2.0 * r[1] ** 2),
        fixture.integral_callable,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        chunkerinterior(chnkr, fixture.interior_points),
        np.asarray(fixture.interior_point_flags, dtype=bool).reshape(-1, order="F"),
    )
    np.testing.assert_array_equal(
        chunkerinterior(chnkr, (fixture.interior_grid_x, fixture.interior_grid_y)).reshape(-1, order="F"),
        np.asarray(fixture.interior_grid_flags, dtype=bool).reshape(-1, order="F"),
    )


def test_accelerated_operator_paths_match_matlab_fixture():
    fixture = load_fixture("operator_parity.mat")["operator_parity"]
    chnkr = chunker_from_fields(fixture.chunker)
    rhs = np.asarray(fixture.compressed_rhs).reshape(-1, order="F")
    lap_s = kernel("lap", "s")

    dense = chunkermat(chnkr, lap_s)
    np.testing.assert_allclose(dense, fixture.lap_s_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dense @ rhs, fixture.lap_s_apply, rtol=1e-12, atol=1e-13)

    fmm_opts = {"acceleration": "fmm", "eps": 1e-12}
    fmm_op = chunkermat(chnkr, lap_s, fmm_opts)
    fmm_apply = chunkermatapply(chnkr, lap_s, rhs, fmm_opts)
    assert isinstance(fmm_op, ChunkerFMMMatrix)
    np.testing.assert_allclose(fmm_op @ rhs, fixture.lap_s_fmm_apply, rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(fmm_apply, fixture.lap_s_fmm_apply, rtol=5e-10, atol=5e-11)
    np.testing.assert_allclose(fmm_op @ rhs, fixture.lap_s_apply, rtol=5e-10, atol=5e-11)

    flam_opts = {
        "acceleration": "flam",
        "dval": 1.0,
        "occ": 8,
        "rank_or_tol": 1e-10,
        "useproxy": False,
    }
    flam_op = chunkermat(chnkr, lap_s, flam_opts)
    shifted = dense + np.eye(chnkr.npt)
    assert isinstance(flam_op, ChunkerFLAMMatrix)
    np.testing.assert_allclose(flam_op @ rhs, fixture.lap_s_flam_apply, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(flam_op @ rhs, shifted @ rhs, rtol=2e-8, atol=2e-10)
    sol = flam_op.solve(rhs)
    np.testing.assert_allclose(sol, fixture.lap_s_flam_solve, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(shifted @ sol, rhs, rtol=2e-8, atol=2e-10)

    flam_proxy_opts = {
        "acceleration": "flam",
        "dval": 1.0,
        "occ": 8,
        "rank_or_tol": 1e-10,
        "proxybylevel": True,
    }
    flam_proxy_op = chunkermat(chnkr, lap_s, flam_proxy_opts)
    assert isinstance(flam_proxy_op, ChunkerFLAMMatrix)
    np.testing.assert_allclose(flam_proxy_op @ rhs, fixture.lap_s_flam_proxy_apply, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(flam_proxy_op @ rhs, shifted @ rhs, rtol=2e-8, atol=2e-10)
    proxy_sol = flam_proxy_op.solve(rhs)
    np.testing.assert_allclose(proxy_sol, fixture.lap_s_flam_proxy_solve, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(shifted @ proxy_sol, rhs, rtol=2e-8, atol=2e-10)
    proxy_eval = chunkerkerneval(
        chnkr,
        lap_s,
        np.asarray(fixture.density_scalar).reshape(-1, order="F"),
        fixture.targets,
        {"acceleration": "flam", "rank_or_tol": 1e-10, "proxybylevel": True},
    )
    np.testing.assert_allclose(
        np.asarray(proxy_eval).reshape(-1, order="F"),
        np.asarray(fixture.lap_s_flam_proxy_eval).reshape(-1, order="F"),
        rtol=5e-6,
        atol=2e-8,
    )

    block_chunkers = [chunker_from_fields(fixture.block_chunker1), chunker_from_fields(fixture.block_chunker2)]
    block_kernels = [
        [block_22_kernel, block_21_kernel],
        [block_12_kernel, block_scalar_kernel],
    ]
    block_rhs = np.asarray(fixture.block_rhs).reshape(-1, order="F")
    block_dense = chunkermat(block_chunkers, block_kernels, {"quad": "native"})
    block_shifted = block_dense + float(fixture.block_dval) * np.eye(block_dense.shape[0])

    np.testing.assert_allclose(block_dense, fixture.block_dense, rtol=1e-12, atol=1e-13)
    block_flam = chunkermat(
        block_chunkers,
        block_kernels,
        {
            "acceleration": "flam",
            "dval": float(fixture.block_dval),
            "quad": "native",
            "occ": 1000,
            "rank_or_tol": 1e-12,
            "useproxy": False,
        },
    )
    assert isinstance(block_flam, ChunkerFLAMMatrix)
    np.testing.assert_allclose(block_flam @ block_rhs, fixture.block_flam_apply, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(block_flam @ block_rhs, block_shifted @ block_rhs, rtol=2e-8, atol=2e-10)
    block_sol = block_flam.solve(block_rhs)
    np.testing.assert_allclose(block_sol, fixture.block_flam_solve, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(block_shifted @ block_sol, block_rhs, rtol=2e-8, atol=2e-10)


def test_section_iii_quadratures_match_matlab_fixture():
    fixture = load_fixture("quadggq.mat")["quadggq"]
    chnkr = chunker_from_fields(fixture.chunker)

    np.testing.assert_array_equal(quadggq.logavail(), np.asarray(fixture.log_orders, dtype=int).reshape(-1))
    np.testing.assert_array_equal(quadggq.hqsuppavail(), np.asarray(fixture.hqsupp_orders, dtype=int).reshape(-1))

    log_xs1, log_wts1, log_xs0, log_wts0 = quadggq.getlogquad(chnkr.k, 2)
    np.testing.assert_allclose(log_xs1, fixture.log_xs1, atol=0.0)
    np.testing.assert_allclose(log_wts1, fixture.log_wts1, atol=0.0)
    assert_cell_arrays_allclose(log_xs0, fixture.log_xs0, atol=0.0, label="getlogquad xs0")
    assert_cell_arrays_allclose(log_wts0, fixture.log_wts0, atol=0.0, label="getlogquad wts0")

    log_aux = quadggq.setup(chnkr.k, "log")
    np.testing.assert_allclose(log_aux.xs1, fixture.log_xs1, atol=0.0)
    np.testing.assert_allclose(log_aux.wts1, fixture.log_wts1, atol=0.0)
    assert_cell_arrays_allclose(log_aux.xs0, fixture.log_xs0, atol=0.0, label="log setup xs0")
    assert_cell_arrays_allclose(log_aux.wts0, fixture.log_wts0, atol=0.0, label="log setup wts0")

    removable_xs0, removable_wts0 = quadggq.getremovablequad(chnkr.k, 1)
    removable_aux = quadggq.setup(chnkr.k, "removable")
    assert_cell_arrays_allclose(removable_xs0, fixture.removable_xs0, label="getremovablequad xs0")
    assert_cell_arrays_allclose(removable_wts0, fixture.removable_wts0, label="getremovablequad wts0")
    assert_cell_arrays_allclose(removable_aux.xs0, fixture.setup_removable_xs0, label="removable setup xs0")
    assert_cell_arrays_allclose(removable_aux.wts0, fixture.setup_removable_wts0, label="removable setup wts0")

    pv_aux = quadggq.setup(chnkr.k, "pv")
    hs_aux = quadggq.setup(chnkr.k, "hs")
    pv_xs0, pv_wts0 = quadggq.getpvquad(chnkr.k)
    hs_xs0, hs_wts0 = quadggq.gethsquad(chnkr.k)
    assert_cell_arrays_allclose(pv_aux.xs0, fixture.pv_xs0, atol=0.0, label="pv setup xs0")
    assert_cell_arrays_allclose(pv_aux.wts0, fixture.pv_wts0, atol=0.0, label="pv setup wts0")
    assert_cell_arrays_allclose(pv_xs0, fixture.pv_xs0, atol=0.0, label="getpvquad xs0")
    assert_cell_arrays_allclose(pv_wts0, fixture.pv_wts0, atol=0.0, label="getpvquad wts0")
    assert_cell_arrays_allclose(hs_aux.xs0, fixture.hs_xs0, atol=0.0, label="hs setup xs0")
    assert_cell_arrays_allclose(hs_aux.wts0, fixture.hs_wts0, atol=0.0, label="hs setup wts0")
    assert_cell_arrays_allclose(hs_xs0, fixture.hs_xs0, atol=0.0, label="gethsquad xs0")
    assert_cell_arrays_allclose(hs_wts0, fixture.hs_wts0, atol=0.0, label="gethsquad wts0")

    lap_s = kernel("lap", "s")
    lap_d = kernel("lap", "d")
    lap_sgrad = kernel("lap", "sgrad")
    lap_dgrad = kernel("lap", "dgrad")
    np.testing.assert_allclose(quadnative.buildmat(chnkr, lap_d, lap_d.opdims), fixture.native_lap_d_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_s, lap_s.opdims, "log"), fixture.log_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_sgrad, lap_sgrad.opdims, "pv"), fixture.pv_mat, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(quadggq.buildmat(chnkr, lap_dgrad, lap_dgrad.opdims, "hs"), fixture.hs_mat, rtol=2e-7, atol=5e-8)

    np.testing.assert_allclose(quadggq.buildmattd(chnkr, lap_s, lap_s.opdims, "log").toarray(), fixture.log_td_mat, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        quadggq.buildmattd(chnkr, lap_s, lap_s.opdims, "log", ilist=[0, 1]).toarray(),
        fixture.log_td_mat_skip,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        quadggq.buildmattd(chnkr, lap_s, lap_s.opdims, "log", corrections=True).toarray(),
        fixture.log_td_mat_corrections,
        rtol=1e-12,
        atol=1e-13,
    )

    diag_chunk = int(fixture.log_diag_chunk) - 1
    near_source_chunk = int(fixture.log_near_source_chunk) - 1
    near_target_chunk = int(fixture.log_near_target_chunk) - 1
    np.testing.assert_allclose(
        quadggq.diagbuildmat(chnkr, diag_chunk, lap_s, lap_s.opdims, log_aux),
        fixture.log_diag_mat,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        quadggq.diagbuildmat(chnkr, diag_chunk, lap_s, lap_s.opdims, log_aux, corrections=True),
        fixture.log_diag_mat_corrections,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        quadggq.nearbuildmat(chnkr, near_target_chunk, near_source_chunk, lap_s, lap_s.opdims, log_aux),
        fixture.log_near_mat,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        quadggq.nearbuildmat(chnkr, near_target_chunk, near_source_chunk, lap_s, lap_s.opdims, log_aux, corrections=True),
        fixture.log_near_mat_corrections,
        rtol=1e-12,
        atol=1e-13,
    )

    skipped = quadggq.buildmat(chnkr, lap_s, lap_s.opdims, "log", ilist=[0, 1])
    np.testing.assert_array_equal(np.isinf(skipped), np.isinf(fixture.log_mat_skip))
    finite = np.isfinite(fixture.log_mat_skip)
    np.testing.assert_allclose(skipped[finite], fixture.log_mat_skip[finite], rtol=1e-12, atol=1e-13)

    np.testing.assert_allclose(
        quadadap.buildmat(chnkr, lap_s, lap_s.opdims, {"sing": "log"}),
        fixture.adap_log_mat,
        rtol=5e-10,
        atol=1e-11,
    )
    close_chnkr = chunker_from_fields(fixture.adap_close_chunker)
    np.testing.assert_allclose(
        quadadap.buildmat(close_chnkr, lap_s, lap_s.opdims, {"sing": "log"}),
        fixture.adap_close_mat,
        rtol=5e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        quadadap.buildmat(close_chnkr, lap_s, lap_s.opdims, {"sing": "log", "robust": True}),
        fixture.adap_close_robust_mat,
        rtol=5e-10,
        atol=1e-11,
    )


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
    assert_chunker_matches_fields(saved.chnkrlocals[-1], fixture.saved_local_last, "rcip saved local chunker", atol=1e-13)

    rhohatinterp, srcinfo, wts = rcip.rhohatInterp(fixture.rhohat, saved, 2)
    for iedge in range(saved.nedge):
        np.testing.assert_allclose(rhohatinterp[iedge], fixture.rhohatinterp[iedge], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].r, fixture.srcinfo[iedge].r, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].d, fixture.srcinfo[iedge].d, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].d2, fixture.srcinfo[iedge].d2, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(srcinfo[iedge].n, fixture.srcinfo[iedge].n, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(wts[iedge], fixture.wts[iedge], rtol=1e-12, atol=1e-13)
