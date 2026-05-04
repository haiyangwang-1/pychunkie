import numpy as np
import pytest

from chunkie import Chunker, chunker, chunkerpref


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


def test_onesmat_and_normonesmat_shapes():
    chnkr = circle_chunker(8)
    assert chnkr.onesmat().shape == (chnkr.npt, chnkr.npt)
    assert chnkr.normonesmat().shape == (2 * chnkr.npt, 2 * chnkr.npt)
