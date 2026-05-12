import numpy as np

from chunkie import chunkerfunc, chunkerfuncuni
from chunkie.geometry import curves


def circle(t, radius=1.0, center=(0.0, 0.0)):
    t = np.asarray(t)
    r = np.vstack((center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)))
    d = np.vstack((-radius * np.sin(t), radius * np.cos(t)))
    d2 = np.vstack((-radius * np.cos(t), -radius * np.sin(t)))
    return r, d, d2


def assert_circle_panels(chnkr, ab, radius, center=(0.0, 0.0), atol=1e-12):
    center_arr = np.asarray(center, dtype=float)
    for ich, (a, b) in enumerate(ab.T):
        theta = a + (b - a) * (chnkr.tstor + 1.0) / 2.0
        h = (b - a) / 2.0
        expected_r, global_d, global_d2 = circle(theta, radius, center)
        expected_n = (expected_r - center_arr[:, None]) / radius

        np.testing.assert_allclose(chnkr.r[:, :, ich], expected_r, atol=atol)
        np.testing.assert_allclose(chnkr.d[:, :, ich], h * global_d, atol=atol)
        np.testing.assert_allclose(chnkr.d2[:, :, ich], h * h * global_d2, atol=atol)
        np.testing.assert_allclose(chnkr.n[:, :, ich], expected_n, atol=atol)
        np.testing.assert_allclose(chnkr.wts[:, ich], radius * h * chnkr.wstor, atol=atol)


def test_chunkerfunc_builds_closed_circle_with_area_and_adjacency():
    chnkr, ab = chunkerfunc(lambda t: circle(t, radius=2.5), {"nchmin": 4}, {"k": 16})

    assert chnkr.nch == 4
    np.testing.assert_allclose(ab[:, 0], [0.0, np.pi / 2.0])
    np.testing.assert_array_equal(chnkr.adj[:, 0], [4, 2])
    np.testing.assert_array_equal(chnkr.adj[:, -1], [3, 1])
    np.testing.assert_allclose(chnkr.area(), np.pi * 2.5**2, atol=1e-12)
    np.testing.assert_allclose(np.sum(chnkr.chunklen()), 2 * np.pi * 2.5, atol=1e-12)
    assert_circle_panels(chnkr, ab, 2.5)


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
    for ich, (a, b) in enumerate(ab.T):
        t = a + (b - a) * (chnkr.tstor + 1.0) / 2.0
        h = (b - a) / 2.0
        np.testing.assert_allclose(chnkr.r[:, :, ich], np.vstack((2.0 * t, np.zeros_like(t))), atol=1e-15)
        np.testing.assert_allclose(chnkr.d[:, :, ich], np.repeat([[2.0 * h], [0.0]], chnkr.k, axis=1))
        np.testing.assert_allclose(chnkr.d2[:, :, ich], 0.0, atol=1e-15)
        np.testing.assert_allclose(chnkr.n[:, :, ich], np.repeat([[0.0], [-1.0]], chnkr.k, axis=1))
        np.testing.assert_allclose(chnkr.wts[:, ich], 2.0 * h * chnkr.wstor)


def test_chunkerfuncuni_builds_requested_uniform_panel_count():
    chnkr = chunkerfuncuni(lambda t: circle(t, radius=1.5), 6, pref={"k": 10})
    breaks = np.linspace(0.0, 2.0 * np.pi, 7)
    ab = np.vstack((breaks[:-1], breaks[1:]))

    assert chnkr.nch == 6
    assert chnkr.k == 10
    np.testing.assert_allclose(chnkr.area(), np.pi * 1.5**2, atol=1e-12)
    assert_circle_panels(chnkr, ab, 1.5, atol=1e-10)


def test_chunkerfunc_can_spectrally_differentiate_position_only_curve():
    def position_only(t):
        return np.vstack((np.cos(t), np.sin(t)))

    chnkr, ab = chunkerfunc(position_only, {"nchmin": 4}, {"k": 16})
    for ich, (a, b) in enumerate(ab.T):
        theta = a + (b - a) * (chnkr.tstor + 1.0) / 2.0
        h = (b - a) / 2.0
        expected_d = h * np.vstack((-np.sin(theta), np.cos(theta)))
        expected_d2 = h**2 * np.vstack((-np.cos(theta), -np.sin(theta)))
        np.testing.assert_allclose(chnkr.d[:, :, ich], expected_d, atol=1e-12)
        np.testing.assert_allclose(chnkr.d2[:, :, ich], expected_d2, atol=1e-11)
    np.testing.assert_allclose(np.linalg.norm(chnkr.n, axis=0), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(chnkr.r * chnkr.n, axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(chnkr.area(), np.pi, atol=1e-12)


def test_chunkerfunc_adaptively_refines_unresolved_curve():
    freq = 24.0 * np.pi

    def wavy(t):
        t = np.asarray(t)
        r = np.vstack((t, 0.05 * np.sin(freq * t)))
        d = np.vstack((np.ones_like(t), 0.05 * freq * np.cos(freq * t)))
        d2 = np.vstack((np.zeros_like(t), -0.05 * freq**2 * np.sin(freq * t)))
        return r, d, d2

    coarse, _ = chunkerfunc(
        wavy,
        {"ta": 0.0, "tb": 1.0, "ifclosed": False, "nchmin": 1, "ifrefine": False, "lvlr": "n"},
        {"k": 8, "nchmax": 256},
    )
    refined, ab = chunkerfunc(
        wavy,
        {"ta": 0.0, "tb": 1.0, "ifclosed": False, "nchmin": 1, "eps": 1e-6, "lvlr": "n"},
        {"k": 8, "nchmax": 256},
    )

    assert coarse.nch == 1
    assert refined.nch > coarse.nch
    np.testing.assert_allclose(ab[0, 0], 0.0)
    np.testing.assert_allclose(ab[1, -1], 1.0)
    np.testing.assert_allclose(ab[1, :-1], ab[0, 1:])
    np.testing.assert_allclose(np.sum(refined.chunklen()), np.sum(refined.wts), atol=1e-14)
    xg, wg = np.polynomial.legendre.leggauss(2000)
    tg = (xg + 1.0) / 2.0
    reference_length = 0.5 * np.sum(wg * np.sqrt(1.0 + (0.05 * freq * np.cos(freq * tg)) ** 2))
    np.testing.assert_allclose(np.sum(refined.chunklen()), reference_length, rtol=1e-10, atol=1e-11)


def test_basic_curve_helpers_match_expected_derivatives():
    t = np.array([0.0, 0.25, 0.5])
    r, d, d2 = curves.fsine(t, 2.0, 3.0, 0.1)

    np.testing.assert_allclose(r[0], t)
    np.testing.assert_allclose(d[0], 1.0)
    np.testing.assert_allclose(d2[0], 0.0)
    np.testing.assert_allclose(r[1], 2.0 * np.sin(3.0 * t + 0.1))
    np.testing.assert_allclose(d[1], 6.0 * np.cos(3.0 * t + 0.1))
    np.testing.assert_allclose(d2[1], -18.0 * np.sin(3.0 * t + 0.1))
