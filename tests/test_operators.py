import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, chunkermat, chunkermatapply, kernel


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def smooth_kernel(src: PointInfo, targ: PointInfo):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    return 1.0 + dx**2 + 0.5 * dy**2


def test_chunkermat_matches_chunkerkerneval_on_boundary_for_smooth_kernel():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    dens = np.cos(chnkr.r[0].reshape(-1))

    mat_vals = chunkermatapply(chnkr, smooth_kernel, dens)
    eval_vals = chunkerkerneval(chnkr, smooth_kernel, dens, chnkr).reshape(-1)

    np.testing.assert_allclose(mat_vals, eval_vals)


def test_chunkermat_accepts_kernel_objects():
    chnkr, _ = chunkerfunc(circle, {"nchmin": 4}, {"k": 8})
    zero = kernel("zero")
    mat = chunkermat(chnkr, zero)

    assert mat.shape == (chnkr.npt, chnkr.npt)
    np.testing.assert_allclose(mat, 0.0)
