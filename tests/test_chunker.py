import numpy as np
import pytest

from chunkie import Chunker, chunker, chunkerfunc, chunkerpoints, chunkerpoly, chunkerpref, lege, merge
from chunkie.geometry import curves


def circle_chunker(k=16):
    chnkr = Chunker({"k": k, "nchstor": 1, "nchmax": 8}).addchunk()
    t = chnkr.tstor
    theta = np.pi * (t + 1.0)
    scale = np.pi
    chnkr.r = np.stack([np.cos(theta), np.sin(theta)], axis=0)[:, :, None]
    chnkr.d = scale * np.stack([-np.sin(theta), np.cos(theta)], axis=0)[:, :, None]
    chnkr.d2 = scale**2 * np.stack([-np.cos(theta), -np.sin(theta)], axis=0)[:, :, None]
    chnkr.adj = np.array([[1], [1]])
    chnkr.recompute_geometry()
    return chnkr


def test_chunker_constructor_defaults_and_validation():
    pref = chunkerpref({"k": 8, "nchmax": 3, "nchstor": 1})
    chnkr = chunker(pref)

    assert chnkr.k == 8
    assert chnkr.dim == 2
    assert chnkr.nch == 0
    assert chnkr.npt == 0

    with pytest.raises(ValueError):
        Chunker({"k": 1})


def test_addchunk_resizes_storage_and_exposes_live_slices():
    chnkr = Chunker({"k": 4, "nchstor": 1, "nchmax": 3})
    chnkr.addchunk(2)

    assert chnkr.nch == 2
    assert chnkr.nchstor >= 2
    assert chnkr.r.shape == (2, 4, 2)
    assert chnkr.wts.shape == (4, 2)

    chnkr.r = np.ones((2, 4, 2))
    chnkr.d = 2.0 * np.ones((2, 4, 2))
    chnkr.d2 = 3.0 * np.ones((2, 4, 2))
    chnkr.n = 4.0 * np.ones((2, 4, 2))
    chnkr.wts = 5.0 * np.ones((4, 2))
    np.testing.assert_allclose(chnkr.rstor[:, :, :2], 1.0)
    np.testing.assert_allclose(chnkr.dstor[:, :, :2], 2.0)
    np.testing.assert_allclose(chnkr.d2stor[:, :, :2], 3.0)
    np.testing.assert_allclose(chnkr.nstor[:, :, :2], 4.0)
    np.testing.assert_allclose(chnkr.wtsstor[:, :2], 5.0)


def test_resize_chunkends_min_max_and_cleardata_helpers():
    chnkr = circle_chunker(12)
    chnkr.resize(4)
    assert chnkr.nchstor == 4

    rend, tauend = chnkr.chunkends()
    np.testing.assert_allclose(rend[:, 0, 0], [1.0, 0.0], atol=5e-6)
    np.testing.assert_allclose(rend[:, 1, 0], [1.0, 0.0], atol=5e-6)
    np.testing.assert_allclose(np.sqrt(np.sum(tauend[:, :, 0] ** 2, axis=0)), 1.0, atol=1e-13)
    np.testing.assert_allclose(chnkr.min(), np.min(chnkr.r.reshape(2, -1, order="F"), axis=1))
    np.testing.assert_allclose(chnkr.max(), np.max(chnkr.r.reshape(2, -1, order="F"), axis=1))

    chnkr.makedatarows(2)
    assert chnkr.datadim == 2
    chnkr.cleardata()
    assert chnkr.datadim == 0
    assert chnkr.data.shape == (0, chnkr.k, 0)


def test_circle_weights_normals_tangents_area_and_length():
    chnkr = circle_chunker(24)

    np.testing.assert_allclose(chnkr.chunklen(), [2 * np.pi], atol=1e-13)
    np.testing.assert_allclose(chnkr.area(), np.pi, atol=1e-13)
    np.testing.assert_allclose(chnkr.arclengthdens(), np.pi)
    np.testing.assert_allclose(chnkr.signed_curvature(), 1.0)
    np.testing.assert_allclose(np.sqrt(np.sum(chnkr.tangents() ** 2, axis=0)), 1.0)
    np.testing.assert_allclose(np.sum(chnkr.n * chnkr.tangents(), axis=0), 0.0, atol=1e-14)


def test_translation_and_scaling_match_matlab_style_operations():
    chnkr = circle_chunker()
    center = chnkr.r.reshape(2, -1) @ chnkr.wts.ravel() / np.sum(chnkr.wts)

    moved = np.array([1.0, -2.0]) + chnkr
    moved_center = moved.r.reshape(2, -1) @ moved.wts.ravel() / np.sum(moved.wts)
    np.testing.assert_allclose(moved_center - center, [1.0, -2.0], atol=1e-14)
    np.testing.assert_allclose(moved.r, chnkr.r + np.array([1.0, -2.0])[:, None, None])
    np.testing.assert_allclose(moved.d, chnkr.d)
    np.testing.assert_allclose(moved.d2, chnkr.d2)
    np.testing.assert_allclose(moved.n, chnkr.n)
    np.testing.assert_allclose(moved.wts, chnkr.wts)

    scaled = chnkr * 2.0
    np.testing.assert_allclose(scaled.r, 2.0 * chnkr.r)
    np.testing.assert_allclose(scaled.d, 2.0 * chnkr.d)
    np.testing.assert_allclose(scaled.d2, 2.0 * chnkr.d2)
    np.testing.assert_allclose(scaled.n, chnkr.n)
    np.testing.assert_allclose(scaled.wts, 2.0 * chnkr.wts)
    np.testing.assert_allclose(scaled.area(), 4.0 * chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(scaled.chunklen(), 2.0 * chnkr.chunklen(), atol=1e-13)

    negative_moved = chnkr.move(scale=-2.0)
    negative_scaled = chnkr * -2.0
    np.testing.assert_allclose(negative_moved.r, negative_scaled.r)
    np.testing.assert_allclose(negative_moved.d, negative_scaled.d)
    np.testing.assert_allclose(negative_moved.d2, negative_scaled.d2)
    np.testing.assert_allclose(negative_moved.n, negative_scaled.n)
    np.testing.assert_allclose(negative_moved.wts, negative_scaled.wts)
    np.testing.assert_allclose(negative_moved.area(), 4.0 * chnkr.area(), atol=1e-13)


def test_matrix_transform_updates_derivatives_normals_and_weights():
    chnkr = circle_chunker()
    mat = np.array([[1.0, 2.0], [2.0, 3.0]])

    transformed = mat @ chnkr
    expected_r = np.einsum("ij,jkl->ikl", mat, chnkr.r)
    expected_d = np.einsum("ij,jkl->ikl", mat, chnkr.d)
    expected_d2 = np.einsum("ij,jkl->ikl", mat, chnkr.d2)
    expected_speed = np.sqrt(np.sum(expected_d**2, axis=0))
    expected_n = np.stack((expected_d[1] / expected_speed, -expected_d[0] / expected_speed), axis=0)

    np.testing.assert_allclose(transformed.r, expected_r)
    np.testing.assert_allclose(transformed.d, expected_d)
    np.testing.assert_allclose(transformed.d2, expected_d2)
    np.testing.assert_allclose(transformed.n, expected_n)
    np.testing.assert_allclose(transformed.wts, expected_speed * chnkr.wstor[:, None])
    np.testing.assert_allclose(transformed.area(), np.linalg.det(mat) * chnkr.area(), atol=1e-13)

    with pytest.raises(TypeError):
        _ = chnkr * mat


def test_rotate_and_reflect_match_matlab_transform_formulas():
    chnkr = circle_chunker()
    theta = np.pi / 3.0
    r0 = np.array([0.25, -0.5])
    r1 = np.array([1.0, 2.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotated = chnkr.rotate(theta, r0, r1)
    np.testing.assert_allclose(
        rotated.r,
        np.einsum("ij,jkl->ikl", rot, chnkr.r - r0[:, None, None]) + r1[:, None, None],
    )
    np.testing.assert_allclose(rotated.d, np.einsum("ij,jkl->ikl", rot, chnkr.d))
    np.testing.assert_allclose(rotated.d2, np.einsum("ij,jkl->ikl", rot, chnkr.d2))
    np.testing.assert_allclose(rotated.n, np.einsum("ij,jkl->ikl", rot, chnkr.n))
    np.testing.assert_allclose(rotated.wts, chnkr.wts)

    angle = np.pi / 4.0
    refmat = np.array(
        [[np.cos(2.0 * angle), np.sin(2.0 * angle)], [np.sin(2.0 * angle), -np.cos(2.0 * angle)]]
    )
    reflected = chnkr.reflect(angle, r0, r1)
    np.testing.assert_allclose(
        reflected.r,
        np.einsum("ij,jkl->ikl", refmat, chnkr.r - r0[:, None, None]) + r1[:, None, None],
    )
    np.testing.assert_allclose(reflected.d, np.einsum("ij,jkl->ikl", refmat, chnkr.d))
    np.testing.assert_allclose(reflected.d2, np.einsum("ij,jkl->ikl", refmat, chnkr.d2))
    np.testing.assert_allclose(reflected.n, np.einsum("ij,jkl->ikl", refmat, chnkr.n))
    np.testing.assert_allclose(reflected.wts, chnkr.wts)


def test_chunker_spectral_helpers_on_circle():
    chnkr = circle_chunker(20)
    rc, dc, d2c = chnkr.exps()

    assert rc.shape == (2, 20, 1)
    assert dc.shape == (2, 20, 1)
    assert d2c.shape == (2, 20, 1)
    _, _, _, vals = lege.exps(20)
    np.testing.assert_allclose(np.einsum("ij,djn->din", vals, rc), chnkr.r, atol=1e-13)
    np.testing.assert_allclose(np.einsum("ij,djn->din", vals, dc), chnkr.d, atol=1e-13)
    np.testing.assert_allclose(np.einsum("ij,djn->din", vals, d2c), chnkr.d2, atol=1e-13)

    s = chnkr.arclengthfun()
    np.testing.assert_allclose(s[:, 0], np.pi * (chnkr.tstor + 1.0), atol=1e-13)

    vals = np.sin(s)
    np.testing.assert_allclose(chnkr.arclengthder(vals), np.cos(s), atol=1e-11)
    np.testing.assert_allclose(
        chnkr.diffmat() @ vals.reshape(-1, order="F"),
        np.cos(s).reshape(-1, order="F"),
        atol=1e-11,
    )


def test_intmat_integrates_in_chunk_order():
    chnkr = circle_chunker(20).refine({"nover": 1})
    imat = chnkr.intmat()
    ones = np.ones(chnkr.npt)
    integrated = imat @ ones

    expected = chnkr.arclengthfun().reshape(-1, order="F")
    np.testing.assert_allclose(integrated, expected, atol=1e-12)


def test_onesmat_and_normonesmat_shapes():
    chnkr = circle_chunker(8)
    weights = chnkr.wts.reshape(-1, order="F")
    normals = chnkr.n.reshape(-1, order="F")

    np.testing.assert_allclose(chnkr.onesmat(), np.ones((chnkr.npt, 1)) @ weights[None, :])
    np.testing.assert_allclose(
        chnkr.normonesmat(),
        normals[:, None] @ (np.repeat(weights, chnkr.dim) * normals)[None, :],
    )


def test_centroids_and_adjacency_info():
    chnkr = circle_chunker(8).refine({"nover": 1})
    ctrs = chnkr.centroids()
    inds, adjs, info = chnkr.sortinfo()
    expected_ctrs = np.sum(chnkr.r * chnkr.wstor[None, :, None], axis=1) / 2.0

    np.testing.assert_allclose(ctrs, expected_ctrs)
    np.testing.assert_array_equal(inds, [0, 1])
    np.testing.assert_array_equal(adjs, chnkr.adj)
    assert info["ncomp"] == 1
    np.testing.assert_array_equal(info["nchs"], [2])
    np.testing.assert_array_equal(info["ifclosed"], [True])
    assert chnkr.checkadjinfo() == 0


def test_upsample_preserves_circle_geometry_and_density_values():
    chnkr = circle_chunker(16)
    sigma = (1.0 + chnkr.tstor - 2.0 * chnkr.tstor**3).reshape(1, chnkr.k, chnkr.nch)

    up, sigmaup = chnkr.upsample(24, sigma)

    assert up.k == 24
    assert up.nch == chnkr.nch
    theta = np.pi * (up.tstor + 1.0)
    expected_r = np.stack([np.cos(theta), np.sin(theta)], axis=0)[:, :, None]
    expected_d = np.pi * np.stack([-np.sin(theta), np.cos(theta)], axis=0)[:, :, None]
    expected_d2 = np.pi**2 * np.stack([-np.cos(theta), -np.sin(theta)], axis=0)[:, :, None]

    np.testing.assert_allclose(up.r, expected_r, atol=1e-9)
    np.testing.assert_allclose(up.d, expected_d, atol=1e-9)
    np.testing.assert_allclose(up.d2, expected_d2, atol=2e-9)
    np.testing.assert_allclose(up.n, expected_r, atol=1e-10)
    np.testing.assert_allclose(up.wts, np.pi * up.wstor[:, None], atol=5e-11)
    np.testing.assert_allclose(up.area(), chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(sigmaup[0, :, 0], 1.0 + up.tstor - 2.0 * up.tstor**3, atol=1e-12)


def test_refine_oversamples_by_splitting_chunks():
    chnkr = circle_chunker(16)
    refined = chnkr.refine({"nover": 1})
    h = np.pi / 2.0

    assert refined.nch == 2
    np.testing.assert_array_equal(refined.adj, [[2, 1], [2, 1]])
    for ich in range(refined.nch):
        theta = ich * np.pi + h * (refined.tstor + 1.0)
        expected_r = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        expected_d = h * np.stack([-np.sin(theta), np.cos(theta)], axis=0)
        expected_d2 = h * h * np.stack([-np.cos(theta), -np.sin(theta)], axis=0)

        np.testing.assert_allclose(refined.r[:, :, ich], expected_r, atol=5e-10)
        np.testing.assert_allclose(refined.d[:, :, ich], expected_d, atol=5e-10)
        np.testing.assert_allclose(refined.d2[:, :, ich], expected_d2, atol=5e-10)
        np.testing.assert_allclose(refined.n[:, :, ich], expected_r, atol=1e-10)
        np.testing.assert_allclose(refined.wts[:, ich], h * refined.wstor, atol=5e-11)
    np.testing.assert_allclose(refined.area(), chnkr.area(), atol=1e-12)
    np.testing.assert_allclose(np.sum(refined.chunklen()), np.sum(chnkr.chunklen()), atol=1e-12)


def test_refine_enforces_arc_length_level_restriction():
    chnkr = chunkerfunc(
        lambda t: curves.linefunc(t, [0.0, 0.0], [1.0, 0.0]),
        {"ta": 0.0, "tb": 1.0, "ifclosed": False, "tsplits": [0.05, 0.1], "ifrefine": False, "lvlr": "n"},
        {"k": 8, "nchmax": 64},
    )[0]

    refined = chnkr.refine({"lvlr": "a", "lvlrfac": 2.1, "stype": "t"})
    lengths = refined.chunklen()

    assert refined.nch > chnkr.nch
    np.testing.assert_allclose(refined.r[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(refined.d[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(refined.d2, 0.0, atol=1e-14)
    expected_n = np.repeat(np.repeat([[[0.0]], [[-1.0]]], refined.k, axis=1), refined.nch, axis=2)
    np.testing.assert_allclose(refined.n, expected_n, atol=1e-14)
    np.testing.assert_allclose(refined.wts, refined.d[0] * refined.wstor[:, None], atol=1e-14)
    for idx, length in enumerate(lengths):
        left, right = refined.adj[:, idx]
        left_len = lengths[left - 1] if left > 0 else length
        right_len = lengths[right - 1] if right > 0 else length
        assert length <= 2.1 * left_len + 1e-13
        assert length <= 2.1 * right_len + 1e-13


def test_chunkerpoints_builds_from_nodes_and_optional_derivatives():
    base = circle_chunker(18)
    rebuilt = chunkerpoints(base.r, {"ifclosed": True})

    np.testing.assert_allclose(rebuilt.r, base.r)
    np.testing.assert_allclose(rebuilt.d, base.d, atol=1e-11)
    np.testing.assert_allclose(rebuilt.d2, base.d2, atol=1e-10)
    np.testing.assert_allclose(rebuilt.area(), base.area(), atol=1e-12)
    np.testing.assert_array_equal(rebuilt.adj, [[1], [1]])

    explicit = chunkerpoints({"r": base.r, "d": 2.0 * base.d, "d2": 3.0 * base.d2})
    np.testing.assert_allclose(explicit.d, 2.0 * base.d)
    np.testing.assert_allclose(explicit.d2, 3.0 * base.d2)


def test_datares_flags_high_order_data_coefficients():
    chnkr = circle_chunker(12)
    chnkr.makedatarows(2)
    _, _, _, v = lege.exps(chnkr.k)
    chnkr.data = np.stack(
        [
            1.0 + chnkr.tstor**2,
            v[:, -1],
        ],
        axis=0,
    )[:, :, None]

    flags = chnkr.datares({"tol": 1e-10})

    np.testing.assert_array_equal(flags, [[True], [False]])
    np.testing.assert_array_equal(chnkr.datares({"idata": [1], "tol": 1e-10}), [[False]])


def test_merge_combines_chunkers_and_pads_data_rows():
    first = chunkerpoly(np.array([[0.0, 1.0], [0.0, 0.0]]), {"ifclosed": False}, {"k": 8})
    first.makedatarows(1)
    first.data = np.ones((1, first.k, first.nch))

    second = chunkerpoly(np.array([[2.0, 2.0], [0.0, 1.0]]), {"ifclosed": False}, {"k": 8})
    second.makedatarows(2)
    second.data = 2.0 * np.ones((2, second.k, second.nch))

    combined = merge([first, second])

    assert combined.nch == first.nch + second.nch
    assert combined.datadim == 2
    np.testing.assert_allclose(combined.r[:, :, 0], first.r[:, :, 0])
    np.testing.assert_allclose(combined.r[:, :, 1], second.r[:, :, 0])
    np.testing.assert_allclose(combined.d[:, :, 0], first.d[:, :, 0])
    np.testing.assert_allclose(combined.d[:, :, 1], second.d[:, :, 0])
    np.testing.assert_allclose(combined.d2[:, :, 0], first.d2[:, :, 0])
    np.testing.assert_allclose(combined.d2[:, :, 1], second.d2[:, :, 0])
    np.testing.assert_allclose(combined.n[:, :, 0], first.n[:, :, 0])
    np.testing.assert_allclose(combined.n[:, :, 1], second.n[:, :, 0])
    np.testing.assert_allclose(combined.wts[:, 0], first.wts[:, 0])
    np.testing.assert_allclose(combined.wts[:, 1], second.wts[:, 0])
    np.testing.assert_array_equal(combined.adj, [[-1, -1], [-1, -1]])
    np.testing.assert_allclose(combined.data[0, :, 0], 1.0)
    np.testing.assert_allclose(combined.data[1, :, 0], 0.0)
    np.testing.assert_allclose(combined.data[:, :, 1], 2.0)
