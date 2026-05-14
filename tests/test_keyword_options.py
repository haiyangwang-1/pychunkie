import numpy as np

from chunkie import chunkerfunc, chunkerinterior, chunkerkerneval, chunkerpoly, kernel


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_keyword_migration_forms_for_geometry_and_operator_helpers():
    chnkr, _ = chunkerfunc(circle, min_chunks=4, order=6)
    targets = np.array([[0.0, 1.2], [0.0, 0.0]])
    direct_inside = chunkerinterior(chnkr, targets)

    flam_inside = chunkerinterior(
        chnkr,
        targets,
        acceleration="flam",
        rank_or_tol=1.0e-8,
        proxy=False,
    )
    near_values = chunkerkerneval(
        chnkr,
        kernel("lap", "s"),
        np.ones(chnkr.npt),
        targets,
        force_adaptive=True,
        near_factor=0.5,
        use_panel_quadrature=False,
    )

    verts = np.array([[0.0, 1.0], [0.0, 0.0]])
    edgevals = np.array([[3.0]])
    line = chunkerpoly(verts, edgevals, closed=False, order=4)

    np.testing.assert_array_equal(flam_inside, direct_inside)
    assert near_values.shape == (1, targets.shape[1])
    assert line.datadim == 1
    np.testing.assert_allclose(line.data[:, :, 0], 3.0)
