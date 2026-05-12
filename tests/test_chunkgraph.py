import numpy as np

from chunkie import (
    chunkerkerneval,
    chunkermat,
    chunkgraph,
    chunkgraphinregion,
    chunkerpoly,
    tochunkgraph,
)
from chunkie.operators import PointInfo


def smooth_kernel(src: PointInfo, targ: PointInfo):
    dx = targ.r[0, :, None] - src.r[0, None, :]
    dy = targ.r[1, :, None] - src.r[1, None, :]
    return 1.0 + dx**2 + dy**2


def square_graph():
    verts = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    return chunkgraph(verts, edges, pref={"k": 8}, cparams={"nchmin": 1})


def test_chunkgraph_constructs_edges_and_vertex_incidence():
    cg = square_graph()

    assert len(cg.echnks) == 4
    assert cg.v2emat.shape == (4, 4)
    np.testing.assert_array_equal(
        cg.v2emat,
        np.array(
            [
                [-1, 1, 0, 0],
                [0, -1, 1, 0],
                [0, 0, -1, 1],
                [1, 0, 0, -1],
            ]
        ),
    )
    assert cg.npt == sum(edge.npt for edge in cg.echnks)
    assert cg.sourceinfo.r.shape == (2, cg.npt)
    assert len(cg.regions) >= 2


def test_chunkgraph_accepts_incidence_matrix_edges():
    incidence = np.array(
        [
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
            [1, 0, 0, -1],
        ]
    )
    verts = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )

    cg = chunkgraph(verts, incidence, pref={"k": 6}, cparams={"nchmin": 1})

    np.testing.assert_array_equal(cg.edgesendverts, np.array([[0, 1, 2, 3], [1, 2, 3, 0]]))


def test_chunkgraph_slice_and_edgeids_match_selected_edges():
    cg = square_graph()
    sub = cg.slicegraph([0, 1])

    assert len(sub.echnks) == 2
    np.testing.assert_array_equal(sub.edgesendverts, np.array([[0, 1], [1, 2]]))
    ids = cg.edgeids([0, 1])
    assert ids.size == cg.echnks[0].npt + cg.echnks[1].npt
    expected = np.hstack(
        (
            cg.echnks[0].r.reshape(2, -1, order="F"),
            cg.echnks[1].r.reshape(2, -1, order="F"),
        )
    )
    np.testing.assert_allclose(cg.r.reshape(2, -1, order="F")[:, ids], expected)


def test_chunkgraph_region_ids_survive_translation():
    cg = square_graph()
    pts = np.array([[0.5, 1.5], [0.5, 0.5]])

    np.testing.assert_array_equal(chunkgraphinregion(cg, pts), [2, 1])

    moved = cg + np.array([2.0, -1.0])
    np.testing.assert_array_equal(chunkgraphinregion(moved, pts + np.array([[2.0], [-1.0]])), [2, 1])


def test_chunkgraph_works_with_dense_operator_helpers():
    cg = square_graph()
    dens = np.sin(cg.r[0].reshape(-1, order="F"))
    weights = cg.wts.reshape(-1, order="F")
    targets = np.array([[0.25, 1.5], [0.25, 0.25]])

    mat = chunkermat(cg, smooth_kernel)
    vals = chunkerkerneval(cg, smooth_kernel, dens, targets).reshape(-1)
    expected_mat = smooth_kernel(cg.sourceinfo, cg.sourceinfo) * weights[None, :]
    expected_vals = smooth_kernel(cg.sourceinfo, PointInfo(r=targets)) @ (dens * weights)

    np.testing.assert_allclose(mat, expected_mat)
    np.testing.assert_allclose(vals, expected_vals)


def test_tochunkgraph_preserves_closed_and_open_components():
    closed = chunkerpoly(
        np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        {"ifclosed": True},
        {"k": 8},
    )
    open_line = chunkerpoly(
        np.array([[2.0, 3.0], [0.0, 0.0]]),
        {"ifclosed": False},
        {"k": 8},
    )

    closed_graph = tochunkgraph(closed)
    line_graph = tochunkgraph(open_line)

    assert closed_graph.edgesendverts.shape == (2, 1)
    assert closed_graph.edgesendverts[0, 0] == closed_graph.edgesendverts[1, 0]
    np.testing.assert_array_equal(line_graph.edgesendverts, np.array([[0], [1]]))
