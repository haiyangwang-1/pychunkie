import numpy as np
import pytest

from chunkie import chunkerfit


def test_chunkerfit_open_line_with_split_points():
    xy = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

    chnkr = chunkerfit(
        xy,
        closed=False,
        split_at_points=True,
        order=8,
    )

    assert chnkr.nch == 3
    np.testing.assert_array_equal(chnkr.adj[:, 0], [-1, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [2, -1])
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 3.0, atol=1e-13)
    np.testing.assert_allclose(chnkr.r[1], 0.0, atol=1e-14)
    u = (chnkr.tstor + 1.0) / 2.0
    for ich in range(chnkr.nch):
        expected_x = ich + u
        np.testing.assert_allclose(chnkr.r[:, :, ich], np.vstack((expected_x, np.zeros_like(u))), atol=1e-14)
        np.testing.assert_allclose(chnkr.d[:, :, ich], np.repeat([[0.5], [0.0]], chnkr.k, axis=1))
        np.testing.assert_allclose(chnkr.d2[:, :, ich], 0.0, atol=1e-13)
        np.testing.assert_allclose(chnkr.n[:, :, ich], np.repeat([[0.0], [-1.0]], chnkr.k, axis=1))
        np.testing.assert_allclose(chnkr.wts[:, ich], 0.5 * chnkr.wstor)


def test_chunkerfit_closed_circle_spline_area():
    t = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    xy = np.vstack((np.cos(t), np.sin(t)))

    chnkr = chunkerfit(
        xy,
        closed=True,
        split_at_points=True,
        order=12,
    )

    assert chnkr.nch == 16
    np.testing.assert_allclose(chnkr.area(), np.pi, rtol=2e-3)
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 2.0 * np.pi, rtol=2e-3)
    np.testing.assert_allclose(np.sqrt(np.sum(chnkr.r**2, axis=0)), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.sum(chnkr.r * chnkr.d, axis=0), 0.0, atol=2e-4)
    np.testing.assert_allclose(chnkr.d2, 0.0, atol=0.0)
    np.testing.assert_array_equal(chnkr.adj[:, 0], [16, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [15, 1])


def test_chunkerfit_rejects_unsupported_methods():
    xy = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        chunkerfit(xy, {"_chunkie_normalized_geometry_options": True, "method": "linear", "ifclosed": False})
