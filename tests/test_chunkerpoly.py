import numpy as np
import pytest

from chunkie import chunkerpoly


def test_chunkerpoly_closed_square_area_length_and_adjacency():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr = chunkerpoly(verts, {"rounded": False}, {"k": 8})

    assert chnkr.nch == 4
    np.testing.assert_allclose(chnkr.area(), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 4.0, atol=1e-14)
    np.testing.assert_array_equal(chnkr.adj[:, 0], [4, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [3, 1])


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
    np.testing.assert_allclose(chnkr.data[:, :, 0], np.array([[10.0], [30.0]]) * np.ones((1, chnkr.k)))
    np.testing.assert_allclose(chnkr.data[:, :, 1], np.array([[20.0], [40.0]]) * np.ones((1, chnkr.k)))


def test_chunkerpoly_rounded_is_explicitly_deferred():
    verts = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NotImplementedError):
        chunkerpoly(verts, {"rounded": True})


def test_reverse_and_move_preserve_expected_geometry():
    verts = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    chnkr = chunkerpoly(verts, {"rounded": False}, {"k": 8})

    rev = chnkr.reverse()
    np.testing.assert_allclose(rev.area(), -1.0, atol=1e-14)
    moved = chnkr.move(r1=[2.0, -1.0], trotat=np.pi / 2.0, scale=3.0)
    np.testing.assert_allclose(moved.area(), 9.0, atol=1e-13)
