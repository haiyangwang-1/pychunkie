import numpy as np

from chunkie import chunkerfunc
from chunkie.chnk import flagnear, flagself


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
