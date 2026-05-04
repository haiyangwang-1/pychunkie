import numpy as np

from chunkie.chnk import smoother


def test_smoother_uniform_mesh_matches_polygon_edges():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    umesh = smoother.get_umesh(verts)

    np.testing.assert_allclose(umesh.lengths, 1.0)
    np.testing.assert_allclose(umesh.centroids[:, 0], [0.5, 0.0])
    np.testing.assert_allclose(umesh.face_normals[:, 0], [0.0, -1.0])
    np.testing.assert_allclose(np.linalg.norm(umesh.pseudo_normals, axis=0), 1.0)


def test_smoother_get_mesh_expands_legendre_panels():
    verts = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    umesh = smoother.get_umesh(verts)
    mesh = smoother.get_mesh(umesh, 2, 5)

    assert mesh.r.shape == (2, 3 * 2 * 5)
    assert mesh.n.shape == mesh.r.shape
    assert mesh.pseudo_normals.shape == mesh.r.shape
    assert mesh.wts.shape == (3 * 2 * 5,)
    np.testing.assert_allclose(np.sum(mesh.wts), np.sum(umesh.lengths))


def test_smoother_smooth_returns_rounded_chunker_and_error_outputs():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr, err, err_by_pt = smoother.smooth(verts, {"k": 8, "widths": 0.1, "return_error": True})

    assert chnkr.nch == 8
    assert err == 0.0
    assert err_by_pt.shape == (chnkr.npt,)
    assert chnkr.area() > 0.9
