import numpy as np

from chunkie import chunkerfunc
from chunkie.chnk import curves


def circle(t, radius=1.0, center=(0.0, 0.0)):
    t = np.asarray(t)
    r = np.vstack((center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))
    d = np.vstack((-radius * np.sin(t), radius * np.cos(t)))
    d2 = np.vstack((-radius * np.cos(t), -radius * np.sin(t)))
    return r, d, d2


def test_chunkerfunc_builds_closed_circle_with_area_and_adjacency():
    chnkr, ab = chunkerfunc(lambda t: circle(t, radius=2.5), {"nchmin": 4}, {"k": 16})

    assert chnkr.nch == 4
    np.testing.assert_allclose(ab[:, 0], [0.0, np.pi / 2.0])
    np.testing.assert_array_equal(chnkr.adj[:, 0], [4, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [3, 1])
    np.testing.assert_allclose(chnkr.area(), np.pi * 2.5**2, atol=1e-12)
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 2 * np.pi * 2.5, atol=1e-12)


def test_chunkerfunc_open_curve_marks_free_ends():
    chnkr, ab = chunkerfunc(
        lambda t: curves.linefunc(t, [0.0, 0.0], [2.0, 0.0]),
        {"ta": 0.0, "tb": 1.0, "ifclosed": False, "nchmin": 2},
        {"k": 8},
    )

    assert chnkr.nch == 2
    np.testing.assert_allclose(ab[:, 0], [0.0, 0.5])
    np.testing.assert_array_equal(chnkr.adj[:, 0], [-1, 2])
    np.testing.assert_array_equal(chnkr.adj[:, 1], [1, -1])
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 2.0, atol=1e-13)


def test_chunkerfunc_can_spectrally_differentiate_position_only_curve():
    def position_only(t):
        return np.vstack((np.cos(t), np.sin(t)))

    chnkr, _ = chunkerfunc(position_only, {"nchmin": 4}, {"k": 16})
    np.testing.assert_allclose(chnkr.area(), np.pi, atol=1e-12)


def test_basic_curve_helpers_match_expected_derivatives():
    t = np.array([0.0, 0.25, 0.5])
    r, d, d2 = curves.fsine(t, 2.0, 3.0, 0.1)

    np.testing.assert_allclose(r[0], t)
    np.testing.assert_allclose(d[0], 1.0)
    np.testing.assert_allclose(d2[0], 0.0)
    np.testing.assert_allclose(r[1], 2.0 * np.sin(3.0 * t + 0.1))
