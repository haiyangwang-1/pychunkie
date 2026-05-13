import numpy as np

from chunkie import lege
from chunkie.misc import smoother


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
    x, w, _, _ = lege.exps(5)
    first_panel_u = (x + 1.0) / 4.0

    assert mesh.r.shape == (2, 3 * 2 * 5)
    assert mesh.n.shape == mesh.r.shape
    assert mesh.pseudo_normals.shape == mesh.r.shape
    assert mesh.wts.shape == (3 * 2 * 5,)
    np.testing.assert_allclose(mesh.r[:, :5], np.vstack((first_panel_u, np.zeros_like(first_panel_u))))
    np.testing.assert_allclose(mesh.n[:, :10], np.repeat([[0.0], [-1.0]], 10, axis=1))
    np.testing.assert_allclose(mesh.wts[:5], w / 4.0)
    np.testing.assert_allclose(np.sum(mesh.wts), np.sum(umesh.lengths))


def test_smoother_smooth_returns_rounded_chunker_and_error_outputs():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr, err, err_by_pt = smoother.smooth(verts, {"k": 8, "widths": 0.1, "return_error": True})
    u = (chnkr.tstor + 1.0) / 2.0

    assert chnkr.nch == 8
    assert err == 0.0
    np.testing.assert_array_equal(err_by_pt, np.zeros(chnkr.npt))
    np.testing.assert_allclose(chnkr.r[:, :, 0], np.vstack((0.1 + 0.8 * u, np.zeros_like(u))))
    np.testing.assert_allclose(np.linalg.norm(chnkr.n, axis=0), 1.0, atol=1e-14)
    assert chnkr.checkadjinfo() == 0
    assert chnkr.area() > 0.9
    assert chnkr.area() < 1.0
