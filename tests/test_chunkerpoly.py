import numpy as np

from chunkie import chunkerpoly


def assert_line_panel(chnkr, ich, start, end, atol=1e-14):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    u = (chnkr.tstor + 1.0) / 2.0
    tangent = end - start
    half_tangent = tangent / 2.0
    speed = np.linalg.norm(half_tangent)

    np.testing.assert_allclose(chnkr.r[:, :, ich], start[:, None] + tangent[:, None] * u[None, :], atol=atol)
    np.testing.assert_allclose(chnkr.d[:, :, ich], np.repeat(half_tangent[:, None], chnkr.k, axis=1), atol=atol)
    np.testing.assert_allclose(chnkr.d2[:, :, ich], 0.0, atol=atol)
    np.testing.assert_allclose(
        chnkr.n[:, :, ich],
        np.repeat([[half_tangent[1]], [-half_tangent[0]]], chnkr.k, axis=1) / speed,
        atol=atol,
    )
    np.testing.assert_allclose(chnkr.wts[:, ich], speed * chnkr.wstor, atol=atol)


def test_chunkerpoly_closed_square_area_length_and_adjacency():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr = chunkerpoly(verts, {"rounded": False}, {"k": 8})

    assert chnkr.nch == 4
    np.testing.assert_allclose(chnkr.area(), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 4.0, atol=1e-14)
    np.testing.assert_array_equal(chnkr.adj[:, 0], [4, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [3, 1])
    for ich in range(4):
        assert_line_panel(chnkr, ich, verts[:, ich], verts[:, (ich + 1) % 4])


def test_chunkerpoly_open_polyline_and_edge_data():
    verts = np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 3.0]])
    chnkr = chunkerpoly(
        verts,
        {"rounded": False, "ifclosed": False},
        {"k": 6},
        edgevals=np.array([[10.0, 20.0], [30.0, 40.0]]),
    )

    assert chnkr.nch == 2
    assert chnkr.datadim == 2
    np.testing.assert_array_equal(chnkr.adj, [[-1, 1], [2, -1]])
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 5.0)
    assert_line_panel(chnkr, 0, verts[:, 0], verts[:, 1])
    assert_line_panel(chnkr, 1, verts[:, 1], verts[:, 2])
    np.testing.assert_allclose(chnkr.data[:, :, 0], np.array([[10.0], [30.0]]) * np.ones((1, chnkr.k)))
    np.testing.assert_allclose(chnkr.data[:, :, 1], np.array([[20.0], [40.0]]) * np.ones((1, chnkr.k)))


def test_chunkerpoly_rounded_builds_trimmed_edges_and_corner_panels():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr = chunkerpoly(verts, {"rounded": True, "widths": 0.1}, {"k": 8})
    u = (chnkr.tstor + 1.0) / 2.0
    first_edge = np.vstack((0.1 + 0.8 * u, np.zeros_like(u)))
    first_corner = (
        (1.0 - u)[None, :] ** 2 * np.array([[0.9], [0.0]])
        + 2.0 * (1.0 - u)[None, :] * u[None, :] * np.array([[1.0], [0.0]])
        + u[None, :] ** 2 * np.array([[1.0], [0.1]])
    )

    assert chnkr.nch == 8
    assert chnkr.checkadjinfo() == 0
    np.testing.assert_allclose(chnkr.r[:, :, 0], first_edge)
    np.testing.assert_allclose(chnkr.d[:, :, 0], np.repeat([[0.4], [0.0]], chnkr.k, axis=1))
    np.testing.assert_allclose(chnkr.d2[:, :, 0], 0.0)
    np.testing.assert_allclose(chnkr.r[:, :, 1], first_corner)
    np.testing.assert_allclose(chnkr.d2[:, :, 1], np.repeat([[-0.05], [0.05]], chnkr.k, axis=1))
    assert chnkr.area() > 0.9
    assert chnkr.area() < 1.0
    assert np.all(chnkr.chunklen() > 0.0)


def test_chunkerpoly_rounded_open_polyline_and_edge_data():
    verts = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    chnkr = chunkerpoly(
        verts,
        {"rounded": True, "ifclosed": False, "widths": [0.0, 0.2, 0.0]},
        {"k": 6},
        edgevals=np.array([2.0, 4.0]),
    )

    assert chnkr.nch == 3
    assert chnkr.datadim == 1
    u = (chnkr.tstor + 1.0) / 2.0
    np.testing.assert_allclose(chnkr.r[:, :, 0], np.vstack((0.8 * u, np.zeros_like(u))))
    np.testing.assert_allclose(
        chnkr.r[:, :, 1],
        (1.0 - u)[None, :] ** 2 * np.array([[0.8], [0.0]])
        + 2.0 * (1.0 - u)[None, :] * u[None, :] * np.array([[1.0], [0.0]])
        + u[None, :] ** 2 * np.array([[1.0], [0.2]]),
    )
    np.testing.assert_allclose(chnkr.r[:, :, 2], np.vstack((np.ones_like(u), 0.2 + 0.8 * u)))
    np.testing.assert_array_equal(chnkr.adj[:, 0], [-1, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [2, -1])
    np.testing.assert_allclose(chnkr.data[:, :, 0], 2.0)
    np.testing.assert_allclose(chnkr.data[:, :, -1], 4.0)
    np.testing.assert_allclose(chnkr.data[:, :, 1], (2.0 * (1.0 - u) + 4.0 * u)[None, :])
    assert np.min(chnkr.data[:, :, 1]) >= 2.0
    assert np.max(chnkr.data[:, :, 1]) <= 4.0


def test_reverse_and_move_preserve_expected_geometry():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr = chunkerpoly(verts, {"rounded": False}, {"k": 8})

    rev = chnkr.reverse()
    np.testing.assert_allclose(rev.area(), -1.0, atol=1e-14)
    np.testing.assert_allclose(rev.r, chnkr.r[:, ::-1, :])
    np.testing.assert_allclose(rev.d, -chnkr.d[:, ::-1, :])
    np.testing.assert_allclose(rev.n, -chnkr.n[:, ::-1, :])
    np.testing.assert_allclose(rev.wts, chnkr.wts[::-1, :])

    moved = chnkr.move(r1=[2.0, -1.0], trotat=np.pi / 2.0, scale=3.0)
    rot = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)], [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
    np.testing.assert_allclose(
        moved.r,
        3.0 * np.einsum("ij,jkl->ikl", rot, chnkr.r) + np.array([2.0, -1.0])[:, None, None],
    )
    np.testing.assert_allclose(moved.d, 3.0 * np.einsum("ij,jkl->ikl", rot, chnkr.d))
    np.testing.assert_allclose(moved.d2, 3.0 * np.einsum("ij,jkl->ikl", rot, chnkr.d2))
    np.testing.assert_allclose(moved.n, np.einsum("ij,jkl->ikl", rot, chnkr.n), atol=1e-15)
    np.testing.assert_allclose(moved.wts, 3.0 * chnkr.wts)
    np.testing.assert_allclose(moved.area(), 9.0, atol=1e-13)
