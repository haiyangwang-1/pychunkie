import numpy as np
import pytest

from chunkie import chunkerfit


def test_chunkerfit_open_line_with_split_points():
    xy = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

    chnkr = chunkerfit(
        xy,
        {"ifclosed": False, "splitatpoints": True, "pref": {"k": 8}},
    )

    assert chnkr.nch == 3
    np.testing.assert_array_equal(chnkr.adj[:, 0], [-1, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [2, -1])
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 3.0, atol=1e-13)
    np.testing.assert_allclose(chnkr.r[1], 0.0, atol=1e-14)


def test_chunkerfit_closed_circle_spline_area():
    t = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    xy = np.vstack((np.cos(t), np.sin(t)))

    chnkr = chunkerfit(
        xy,
        {"ifclosed": True, "splitatpoints": True, "pref": {"k": 12}},
    )

    assert chnkr.nch == 16
    np.testing.assert_allclose(chnkr.area(), np.pi, rtol=2e-3)


def test_chunkerfit_rejects_unsupported_methods():
    xy = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        chunkerfit(xy, {"method": "linear", "ifclosed": False})
