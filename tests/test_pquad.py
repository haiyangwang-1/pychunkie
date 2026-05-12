from time import perf_counter

import numpy as np
import pytest

from chunkie import PointInfo, chunkerfunc, kernel, lege
from chunkie.chnk import pquad


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_low_level_pquad_weights_match_oversampled_legendre_moments(record_property):
    nsrc = 16
    nodes, weights = lege.exps(nsrc)[:2]
    source = nodes.astype(complex)
    normal = -1j * np.ones(nsrc)
    wxp = weights.astype(complex)
    target = np.array([0.0 + 0.2j])
    start = perf_counter()
    special = pquad.sd_special_quad(target, source, normal, wxp, -1.0, 1.0, "i", nout=4)
    elapsed = perf_counter() - start

    ref_nodes, ref_weights = lege.exps(800)[:2]
    ref_z = ref_nodes.astype(complex)
    max_error = 0.0
    for degree in (0, 1, 3, 7):
        node_values = nodes**degree
        ref_values = ref_nodes**degree
        expected = (
            np.sum(-np.log(np.abs(ref_z - target[0])) / (2.0 * np.pi) * ref_values * ref_weights),
            np.sum(1j / (2.0 * np.pi) * ref_values / (ref_z - target[0]) * ref_weights),
            np.sum(1j / (2.0 * np.pi) * ref_values / (ref_z - target[0]) ** 2 * ref_weights),
            np.sum(1j / (2.0 * np.pi) * ref_values / (ref_z - target[0]) ** 3 * ref_weights),
        )
        for weights0, expected0 in zip(special, expected, strict=True):
            actual0 = weights0 @ node_values
            max_error = max(max_error, float(np.max(np.abs(actual0 - expected0))))
            np.testing.assert_allclose(actual0, expected0, rtol=1e-12, atol=1e-12)
    _report_metrics(record_property, "low_level_moments", elapsed, max_error)


def test_pquad_panel_weights_can_compose_to_original_nodes(record_property):
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    src_chunk = 0
    product_nodes, product_weights = lege.exps(2 * chnkr.k)[:2]
    interp = lege.matrin(chnkr.k, product_nodes)[0]
    mid = lege.matrin(chnkr.k, [0.0])[0]
    rmid = (mid @ chnkr.r[:, :, src_chunk].T).T[:, 0]
    nmid = (mid @ chnkr.n[:, :, src_chunk].T).T[:, 0]
    targ = PointInfo(r=(rmid + 0.04 * nmid).reshape(2, 1))

    start = perf_counter()
    upsampled = pquad.pquadwts(
        chnkr,
        src_chunk,
        targ,
        (pquad.LOG, pquad.CAUCHY),
        "e",
        nodes=product_nodes,
        weights=product_weights,
        intp=interp,
        ifup=True,
    )
    original = pquad.pquadwts(
        chnkr,
        src_chunk,
        targ,
        (pquad.LOG, pquad.CAUCHY),
        "e",
        nodes=product_nodes,
        weights=product_weights,
        intp=interp,
        ifup=False,
    )
    elapsed = perf_counter() - start

    max_error = 0.0
    for upsampled_weights, original_weights in zip(upsampled, original, strict=True):
        composed = upsampled_weights @ interp
        max_error = max(max_error, float(np.max(np.abs(original_weights - composed))))
        np.testing.assert_allclose(original_weights, composed, atol=1e-14)
    _report_metrics(record_property, "panel_weight_composition", elapsed, max_error)


@pytest.mark.parametrize(
    ("kernel_args", "side", "normal_sign"),
    [
        (("lap", "s"), "e", 1.0),
        (("lap", "d"), "e", 1.0),
        (("helm", "s", 1.7), "e", 1.0),
        (("helm", "d", 1.7), "e", 1.0),
        (("helm", "d", 1.7), "i", -1.0),
    ],
)
def test_pquad_split_panel_matrix_matches_oversampled_legendre(kernel_args, side, normal_sign, record_property):
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    src_chunk = 0
    mid = lege.matrin(chnkr.k, [0.0])[0]
    rmid = (mid @ chnkr.r[:, :, src_chunk].T).T[:, 0]
    nmid = (mid @ chnkr.n[:, :, src_chunk].T).T[:, 0]
    targ = PointInfo(r=(rmid + normal_sign * 0.03 * nmid).reshape(2, 1))
    kern = kernel(*kernel_args)
    splitinfo = pquad.splitinfo_for_kernel(kern)

    assert splitinfo is not None
    start = perf_counter()
    actual = pquad.panel_matrix(chnkr, src_chunk, targ, splitinfo, side)
    elapsed = perf_counter() - start
    expected = _oversampled_panel_matrix(chnkr, src_chunk, targ, kern, nref=350)
    max_error = float(np.max(np.abs(actual - expected)))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=3e-7)
    _report_metrics(record_property, f"panel_matrix_{kern.name}_{kern.type}_{side}", elapsed, max_error)


def _oversampled_panel_matrix(chnkr, src_chunk, targ, kern, nref=350):
    nodes, weights = lege.exps(nref)[:2]
    interp = lege.matrin(chnkr.k, nodes)[0]
    r = (interp @ chnkr.r[:, :, src_chunk].T).T
    d = (interp @ chnkr.d[:, :, src_chunk].T).T
    d2 = (interp @ chnkr.d2[:, :, src_chunk].T).T
    speed = np.sqrt(np.sum(np.abs(d) ** 2, axis=0))
    normal = np.vstack((d[1], -d[0])) / speed[None, :]
    src = PointInfo(r=r, d=d, d2=d2, n=normal)
    values = kern(src, targ)
    op1 = kern.opdims[1]
    mat = values * np.repeat(weights * speed, op1)[None, :]
    return mat @ np.kron(interp, np.eye(op1))


def _report_metrics(record_property, label, elapsed, max_error):
    record_property("pquad_label", label)
    record_property("pquad_time_s", elapsed)
    record_property("pquad_max_abs_error", max_error)
    print(f"pquad metric: {label} time={elapsed:.6e}s max_abs_error={max_error:.6e}")
