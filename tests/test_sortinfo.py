import numpy as np

from chunkie import chunkerpoly


def test_sort_reorders_two_open_segments_and_remaps_adjacency():
    verts = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    chnkr = chunkerpoly(verts, {"rounded": False, "ifclosed": False}, {"k": 4})
    shuffled = chnkr.copy()
    shuffled.r = chnkr.r[:, :, [1, 0]]
    shuffled.d = chnkr.d[:, :, [1, 0]]
    shuffled.d2 = chnkr.d2[:, :, [1, 0]]
    shuffled.n = chnkr.n[:, :, [1, 0]]
    shuffled.wts = chnkr.wts[:, [1, 0]]
    shuffled.adj = np.array([[2, -1], [-1, 1]])

    sorted_chnkr, info = shuffled.sort()

    assert info["ier"] == 0
    np.testing.assert_array_equal(sorted_chnkr.adj, chnkr.adj)
    np.testing.assert_allclose(sorted_chnkr.r, chnkr.r)
    np.testing.assert_allclose(sorted_chnkr.d, chnkr.d)
    np.testing.assert_allclose(sorted_chnkr.d2, chnkr.d2)
    np.testing.assert_allclose(sorted_chnkr.n, chnkr.n)
    np.testing.assert_allclose(sorted_chnkr.wts, chnkr.wts)
