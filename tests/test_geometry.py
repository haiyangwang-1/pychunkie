import numpy as np

from chunkie import chunkerfunc, chunkerpoly
from chunkie.chnk import chunk_nearparam, curvature2d, flagnear, flagself, normal2d, perp


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_flagnear_matches_bruteforce_chunk_node_distance():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    pts = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    fac = 0.75

    flag = flagnear(chnkr, pts, {"fac": fac})

    expected = np.zeros_like(flag)
    lens = chnkr.chunklen()
    for ich in range(chnkr.nch):
        for ipt in range(pts.shape[1]):
            dists = np.sqrt(np.sum((chnkr.r[:, :, ich] - pts[:, ipt : ipt + 1]) ** 2, axis=0))
            expected[ipt, ich] = np.any(dists < fac * lens[ich])
    np.testing.assert_array_equal(flag, expected)


def test_flagself_reports_close_source_target_pairs():
    src = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    targ = np.array([[2.0, 0.0, 3.0], [2.0, 0.0, 3.0]])

    pairs = flagself(src, targ)

    np.testing.assert_array_equal(pairs, np.array([[0, 2], [1, 0]]))


def test_basic_2d_geometry_helpers():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 2}, {"k": 8})
    d = chnkr.d.reshape(2, -1)
    d2 = chnkr.d2.reshape(2, -1)

    np.testing.assert_allclose(perp(np.array([[1.0, 0.0], [0.0, 1.0]])), [[0.0, 1.0], [-1.0, -0.0]])
    np.testing.assert_allclose(normal2d({"d": d}), chnkr.n.reshape(2, -1), atol=1e-14)
    np.testing.assert_allclose(curvature2d({"d": d, "d2": d2}), 1.0, atol=1e-12)


def test_chunk_nearparam_on_line_segment():
    chnkr = chunkerpoly(np.array([[0.0, 2.0], [0.0, 0.0]]), {"ifclosed": False}, {"k": 12})
    ts, rs, ds, d2s, dist2s = chunk_nearparam(
        chnkr.r[:, :, 0], np.array([[0.5, 1.75], [1.0, -0.25]]), t=chnkr.tstor
    )

    np.testing.assert_allclose(ts, [-0.5, 0.75], atol=1e-12)
    np.testing.assert_allclose(rs, [[0.5, 1.75], [0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(ds, [[1.0, 1.0], [0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(d2s, 0.0, atol=1e-12)
    np.testing.assert_allclose(dist2s, [1.0, 0.0625], atol=1e-12)


def test_chunker_nearest_selects_point_and_chunk():
    chnkr = chunkerpoly(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        {"ifclosed": False},
        {"k": 12},
    )

    rn, dn, d2n, dist, tn, ichn = chnkr.nearest(np.array([1.25, 0.6]))

    np.testing.assert_allclose(rn, [1.25, 0.0], atol=1e-12)
    np.testing.assert_allclose(dn, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(d2n, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dist, 0.6, atol=1e-12)
    np.testing.assert_allclose(tn, 0.25, atol=1e-12)
    assert ichn == 0
