import numpy as np
import pytest

from chunkie import Chunker, chunker, chunkerpoints, chunkerpref, lege


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
    np.testing.assert_allclose(chnkr.rstor[:, :, :2], 1.0)


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

    scaled = chnkr * 2.0
    np.testing.assert_allclose(scaled.area(), 4.0 * chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(scaled.chunklen(), 2.0 * chnkr.chunklen(), atol=1e-13)


def test_matrix_transform_updates_derivatives_normals_and_weights():
    chnkr = circle_chunker()
    mat = np.array([[1.0, 2.0], [2.0, 3.0]])

    transformed = mat @ chnkr

    np.testing.assert_allclose(transformed.r, np.einsum("ij,jkl->ikl", mat, chnkr.r))
    np.testing.assert_allclose(transformed.d, np.einsum("ij,jkl->ikl", mat, chnkr.d))
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
    np.testing.assert_allclose(rotated.n, np.einsum("ij,jkl->ikl", rot, chnkr.n))

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
    np.testing.assert_allclose(reflected.n, np.einsum("ij,jkl->ikl", refmat, chnkr.n))


def test_chunker_spectral_helpers_on_circle():
    chnkr = circle_chunker(20)
    rc, dc, d2c = chnkr.exps()

    assert rc.shape == (2, 20, 1)
    assert dc.shape == (2, 20, 1)
    assert d2c.shape == (2, 20, 1)

    s = chnkr.arclengthfun()
    np.testing.assert_allclose(s[:, 0], np.pi * (chnkr.tstor + 1.0), atol=1e-13)

    vals = np.sin(s)
    np.testing.assert_allclose(chnkr.arclengthder(vals), np.cos(s), atol=1e-11)
    np.testing.assert_allclose(chnkr.diffmat() @ vals.reshape(-1), np.cos(s).reshape(-1), atol=1e-11)


def test_intmat_integrates_in_chunk_order():
    chnkr = circle_chunker(20).refine({"nover": 1})
    imat = chnkr.intmat()
    ones = np.ones(chnkr.npt)
    integrated = imat @ ones

    expected = chnkr.arclengthfun().T.reshape(-1)
    np.testing.assert_allclose(integrated, expected, atol=1e-12)


def test_onesmat_and_normonesmat_shapes():
    chnkr = circle_chunker(8)
    assert chnkr.onesmat().shape == (chnkr.npt, chnkr.npt)
    assert chnkr.normonesmat().shape == (2 * chnkr.npt, 2 * chnkr.npt)


def test_centroids_and_adjacency_info():
    chnkr = circle_chunker(8).refine({"nover": 1})
    ctrs = chnkr.centroids()
    inds, adjs, info = chnkr.sortinfo()

    assert ctrs.shape == (2, chnkr.nch)
    np.testing.assert_array_equal(inds, [0, 1])
    np.testing.assert_array_equal(adjs, chnkr.adj)
    assert info["ncomp"] == 1
    np.testing.assert_array_equal(info["nchs"], [2])
    np.testing.assert_array_equal(info["ifclosed"], [True])
    assert chnkr.checkadjinfo() == 0


def test_upsample_preserves_circle_geometry_and_density_values():
    chnkr = circle_chunker(8)
    sigma = (1.0 + chnkr.tstor - 2.0 * chnkr.tstor**3).reshape(1, chnkr.k, chnkr.nch)

    up, sigmaup = chnkr.upsample(16, sigma)

    assert up.k == 16
    assert up.nch == chnkr.nch
    np.testing.assert_allclose(up.area(), chnkr.area(), atol=1e-13)
    np.testing.assert_allclose(sigmaup[0, :, 0], 1.0 + up.tstor - 2.0 * up.tstor**3, atol=1e-12)


def test_refine_oversamples_by_splitting_chunks():
    chnkr = circle_chunker(16)
    refined = chnkr.refine({"nover": 1})

    assert refined.nch == 2
    np.testing.assert_array_equal(refined.adj, [[2, 1], [2, 1]])
    np.testing.assert_allclose(refined.area(), chnkr.area(), atol=1e-12)
    np.testing.assert_allclose(np.sum(refined.chunklen()), np.sum(chnkr.chunklen()), atol=1e-12)


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
