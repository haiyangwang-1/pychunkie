import numpy as np

from chunkie import chunkerfunc, chunkerpoly, tochunkgraph


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_flagnear_matches_bruteforce_chunk_node_distance():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    pts = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    fac = 0.75

    flag = chnkr.flagnear(pts, fac=fac)

    expected = np.zeros_like(flag)
    lens = chnkr.chunklen()
    for ich in range(chnkr.nch):
        for ipt in range(pts.shape[1]):
            dists = np.sqrt(np.sum((chnkr.r[:, :, ich] - pts[:, ipt : ipt + 1]) ** 2, axis=0))
            expected[ipt, ich] = np.any(dists < fac * lens[ich])
    np.testing.assert_array_equal(flag, expected)


def test_flagnear_rectangle_grid_matches_direct_meshgrid_order():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=8)
    x = np.linspace(-1.5, 1.5, 21)
    y = np.linspace(-1.25, 1.25, 17)
    xx, yy = np.meshgrid(x, y)
    pts = np.vstack((xx.ravel(order="F"), yy.ravel(order="F")))

    direct = chnkr.flagnear_rectangle(pts, rho=1.5)
    grid = chnkr.flagnear_rectangle_grid(x, y, rho=1.5)

    np.testing.assert_array_equal(grid, direct)


def test_flagnear_rectangle_uses_per_chunk_padding_and_chunkgraph_delegates():
    chnkr = chunkerpoly(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        closed=False,
        order=8,
    )
    pts = np.array([[1.0, 1.0, 2.25, 2.25], [0.1, 1.6, 0.5, 1.6]])

    tight = chnkr.flagnear_rectangle(pts, rho=1.0)
    padded = chnkr.flagnear_rectangle(pts, rho=1.8)
    graph = tochunkgraph(chnkr)
    expected_tight = np.array(
        [
            [False, False],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    expected_padded = np.array(
        [
            [True, False],
            [False, False],
            [False, True],
            [False, False],
        ]
    )

    np.testing.assert_array_equal(tight, expected_tight)
    np.testing.assert_array_equal(padded, expected_padded)
    np.testing.assert_array_equal(graph.flagnear(pts, fac=0.5), chnkr.flagnear(pts, fac=0.5))
    np.testing.assert_array_equal(graph.flagnear_rectangle(pts, rho=1.8), padded)


def test_chunker_nearest_selects_point_and_chunk():
    chnkr = chunkerpoly(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        closed=False,
        order=12,
    )

    rn, dn, d2n, dist, tn, ichn = chnkr.nearest(np.array([1.25, 0.6]))

    np.testing.assert_allclose(rn, [1.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(dn, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d2n, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dist, 0.6, atol=1e-12)
    np.testing.assert_allclose(tn, 0.25, atol=1e-12)
    assert ichn == 0
