from pathlib import Path

import numpy as np
from scipy.io import loadmat

from chunkie import chunkerfunc, lege


GOLDEN = Path(__file__).parent / "golden"


def circle(t, radius):
    t = np.asarray(t)
    return (
        radius * np.vstack((np.cos(t), np.sin(t))),
        radius * np.vstack((-np.sin(t), np.cos(t))),
        radius * np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_legendre_basic_fixture_matches_matlab():
    fixture = loadmat(GOLDEN / "lege_basic.mat", squeeze_me=True)
    k = int(fixture["k"])
    x, w, u, v = lege.exps(k)
    dmat = lege.dermat(k, u, v)

    np.testing.assert_allclose(x, fixture["x"], atol=1e-14)
    np.testing.assert_allclose(w, fixture["w"], atol=1e-14)
    np.testing.assert_allclose(u, fixture["u"], atol=1e-13)
    np.testing.assert_allclose(v, fixture["v"], atol=1e-13)
    np.testing.assert_allclose(dmat, fixture["dmat"], atol=1e-12)


def test_circle_chunker_fixture_matches_matlab():
    fixture = loadmat(GOLDEN / "chunker_circle.mat", squeeze_me=True, struct_as_record=False)
    radius = float(fixture["radius"])
    chnkr, ab = chunkerfunc(lambda t: circle(t, radius), {"nchmin": 4}, {"k": 16})
    fields = fixture["chunker_fields"]

    np.testing.assert_allclose(ab, fixture["ab"], atol=1e-14)
    np.testing.assert_allclose(chnkr.r, fields.r, atol=1e-13)
    np.testing.assert_allclose(chnkr.d, fields.d, atol=1e-13)
    np.testing.assert_allclose(chnkr.d2, fields.d2, atol=1e-13)
    np.testing.assert_allclose(chnkr.n, fields.n, atol=1e-13)
    np.testing.assert_allclose(chnkr.wts, fields.wts, atol=1e-13)
    np.testing.assert_array_equal(chnkr.adj, fields.adj)
    np.testing.assert_allclose(chnkr.area(), fixture["area_val"], atol=1e-13)
    np.testing.assert_allclose(chnkr.chunklen(), fixture["chunklens"], atol=1e-13)
