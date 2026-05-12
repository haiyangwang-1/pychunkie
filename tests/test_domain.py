import numpy as np
import pytest

from chunkie import (
    checkcurveparam,
    chunkgraph,
    ellipse,
    hypoct_uni,
    mergeregions,
    nonflatinterface,
    pointinregion,
    redblue,
    regioninside,
    starfish,
)
from chunkie.chnk import curves


def test_ellipse_and_starfish_helpers_match_expected_formulas():
    t = np.array([0.0, np.pi / 2.0, np.pi])

    r, d, d2 = ellipse(t, 2.0, 0.5)
    np.testing.assert_allclose(r, [[2.0, 0.0, -2.0], [0.0, 0.5, 0.0]], atol=1e-15)
    np.testing.assert_allclose(d, [[0.0, -2.0, 0.0], [0.5, 0.0, -0.5]], atol=1e-15)
    np.testing.assert_allclose(d2, -r, atol=1e-15)

    top = starfish(t, 3, 0.2, [0.1, -0.2], np.pi / 7.0, 1.4)
    nested = curves.starfish(t, 3, 0.2, [0.1, -0.2], np.pi / 7.0, 1.4)
    for actual, expected in zip(top, nested):
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_checkcurveparam_validates_dimension_and_output_shapes():
    assert checkcurveparam(lambda t: ellipse(t, 2.0, 1.0), np.array([0.0, 0.5])) == 2

    def bad_shape(t):
        return np.ones((2, np.asarray(t).size + 1))

    with pytest.raises(ValueError, match="match input"):
        checkcurveparam(bad_shape, np.array([0.0, 0.5]), nout=1)

    def bad_dim(t):
        return np.ones((2, np.asarray(t).size)), np.ones((3, np.asarray(t).size))

    with pytest.raises(ValueError, match="consistent"):
        checkcurveparam(bad_dim, np.array([0.0, 0.5]), nout=2)


def test_nonflatinterface_derivatives_and_redblue_colormap():
    t = np.array([-0.3, 0.1, 0.4])
    r, d, d2 = nonflatinterface(t, 0.7, 2.0, -0.4, 1.3)
    expfac = np.exp(-0.7 * t**2 / 2.0)
    phase = 2.0 * t - 0.4
    expected_y = 1.3 * expfac * np.sin(phase)
    expected_dy = 1.3 * expfac * (2.0 * np.cos(phase) - 0.7 * t * np.sin(phase))
    expected_d2y = 1.3 * expfac * (
        -2.0 * 0.7 * 2.0 * t * np.cos(phase) + (0.7 * 0.7 * t**2 - 0.7 - 2.0 * 2.0) * np.sin(phase)
    )

    np.testing.assert_allclose(r[0], t)
    np.testing.assert_allclose(r[1], expected_y)
    np.testing.assert_allclose(d[0], 1.0)
    np.testing.assert_allclose(d[1], expected_dy)
    np.testing.assert_allclose(d2[0], 0.0)
    np.testing.assert_allclose(d2[1], expected_d2y)

    eps = 1e-6
    rp, _, _ = nonflatinterface(t + eps, 0.7, 2.0, -0.4, 1.3)
    rm, _, _ = nonflatinterface(t - eps, 0.7, 2.0, -0.4, 1.3)
    np.testing.assert_allclose((rp[1] - rm[1]) / (2 * eps), d[1], rtol=1e-9, atol=1e-10)

    cmap = redblue(5)
    np.testing.assert_allclose(cmap[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(cmap[2], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cmap[-1], [1.0, 0.0, 0.0])


def test_hypoct_uni_builds_zero_based_uniform_tree():
    pts = np.array(
        [
            [0.1, 0.9, 0.1, 0.9],
            [0.1, 0.1, 0.9, 0.9],
        ]
    )
    tree = hypoct_uni(pts, 0.4, ext=np.array([[0.0, 1.0], [0.0, 1.0]]))

    assert tree.nlvl == 2
    np.testing.assert_array_equal(tree.lvp, [0, 1, 5])
    assert tree.lrt == 1.0
    np.testing.assert_allclose(
        np.column_stack([node.ctr for node in tree.nodes]),
        [[0.5, 0.25, 0.75, 0.25, 0.75], [0.5, 0.25, 0.25, 0.75, 0.75]],
        atol=0.0,
    )
    assert tree.nodes[0].xi.size == 0
    assert tree.nodes[0].chld == [1, 2, 3, 4]
    assert [node.prnt for node in tree.nodes[1:]] == [0, 0, 0, 0]
    np.testing.assert_array_equal([int(node.xi[0]) for node in tree.nodes[1:]], [0, 1, 2, 3])
    assert [node.nbor for node in tree.nodes[1:]] == [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]


def test_chunkgraph_region_helpers_count_inside_and_merge_nested_regions():
    verts = np.array(
        [
            [0.0, 2.0, 2.0, 0.0, 0.75, 1.25, 1.25, 0.75],
            [0.0, 0.0, 2.0, 2.0, 0.75, 0.75, 1.25, 1.25],
        ]
    )
    edges = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 0, 5, 6, 7, 4],
        ]
    )
    cg = chunkgraph(verts, edges, pref={"k": 6}, cparams={"nchmin": 1})
    outer = [[], [[0, 1, 2, 3]]]
    inner = [[[4, 5, 6, 7]]]

    assert pointinregion(cg, outer[1], [1.0, 1.0]) == 1
    assert pointinregion(cg, outer[1], [3.0, 1.0]) == 0
    assert regioninside(cg, outer, inner)

    merged = mergeregions(cg, outer, inner)
    assert merged[0] == []
    assert merged[1] == [[0, 1, 2, 3], [4, 5, 6, 7]]
