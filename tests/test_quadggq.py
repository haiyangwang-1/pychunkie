import numpy as np

from chunkie import chunkerfunc, chunkerkerneval, chunkermat, kernel
from chunkie.chnk import quadggq


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_generated_self_quadrature_splits_at_each_legendre_node():
    aux = quadggq.setup(8)
    xleg = np.polynomial.legendre.leggauss(8)[0]

    xs1, wts1, xs0, wts0 = quadggq.getlogquad(8, 4)
    assert xs1.shape == wts1.shape
    assert len(xs0) == len(wts0) == 8
    assert 8 in quadggq.logavail()

    assert len(aux.xs0) == 8
    assert aux.ainterp1.shape[1] == 8
    for node, xs, wts, interp in zip(xleg, aux.xs0, aux.wts0, aux.ainterps0):
        assert xs.shape == wts.shape
        assert interp.shape == (xs.size, 8)
        assert np.all(xs >= -1.0)
        assert np.all(xs <= 1.0)
        assert not np.any(np.isclose(xs, node))
        np.testing.assert_allclose(np.sum(wts), 2.0, atol=1e-14)


def test_quadggq_buildmat_removes_laplace_single_layer_diagonal_infinities():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    lap_s = kernel("lap", "s")

    smooth = chunkermat(chnkr, lap_s, {"forcesmooth": True})
    special = quadggq.buildmat(chnkr, lap_s, lap_s.opdims)

    assert np.isinf(np.diag(smooth)).all()
    assert np.isfinite(special).all()
    np.testing.assert_allclose(special @ np.ones(chnkr.npt), 0.0, atol=5e-5)


def test_chunkermat_uses_special_quadrature_for_log_kernels_by_default():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 8}, {"k": 8})
    lap_s = kernel("lap", "s")
    dens = np.ones(chnkr.npt)

    mat_vals = chunkermat(chnkr, lap_s) @ dens
    eval_vals = chunkerkerneval(chnkr, lap_s, dens, chnkr).reshape(-1, order="F")

    assert np.isfinite(mat_vals).all()
    np.testing.assert_allclose(mat_vals, eval_vals)
    np.testing.assert_allclose(mat_vals, 0.0, atol=5e-5)


def test_quadggq_handles_complex_helmholtz_single_layer_blocks():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 6}, {"k": 8})
    helm_s = kernel("helm", "s", 1.3 + 0.2j)

    mat = quadggq.buildmat(chnkr, helm_s, helm_s.opdims)

    assert np.iscomplexobj(mat)
    assert np.isfinite(mat).all()
