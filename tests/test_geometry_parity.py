import numpy as np
import pytest

from chunkie import (
    Chunker,
    checkcurveparam,
    chunkerfunc,
    chunkerfuncuni,
    chunkerpoints,
    chunkgraph,
    chunkgraphinregion,
    ellipse,
    hypoct_uni,
    merge,
    mergeregions,
    nonflatinterface,
    pointinregion,
    redblue,
    regioninside,
    starfish,
    tochunkgraph,
)
from chunkie import lege
from chunkie.chnk import (
    chunk_nearparam,
    curvature2d,
    curves,
    flagnear,
    flagnear_rectangle,
    flagnear_rectangle_grid,
    flagself,
    normal2d,
    perp,
)
from chunkie.numerics import arcparam
from _fixture_generation import assert_chunker_matches_fields, chunker_from_fields, load_generated_mat_fixture


def load_geometry_core():
    return load_generated_mat_fixture("geometry_core.mat", squeeze_me=True, struct_as_record=False)["geometry_core"]


def attach_data(chnkr: Chunker, data: np.ndarray) -> Chunker:
    data_arr = np.asarray(data)
    if data_arr.ndim == 2:
        data_arr = data_arr.reshape(1, *data_arr.shape)
    out = chnkr.copy()
    out.makedatarows(data_arr.shape[0])
    out.data = data_arr
    return out


def assert_chunkgraph_matches_fields(cgrph, fields, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(cgrph.verts, np.asarray(fields.verts).reshape(cgrph.verts.shape, order="F"), atol=atol)
    np.testing.assert_array_equal(cgrph.edgesendverts + 1, np.asarray(fields.edgesendverts, dtype=int).reshape(2, -1, order="F"))
    np.testing.assert_array_equal(cgrph.v2emat, np.asarray(fields.v2emat, dtype=int).reshape(cgrph.v2emat.shape, order="F"))
    assert cgrph.npt == int(fields.npt)
    np.testing.assert_allclose(cgrph.r, fields.r, atol=atol)
    np.testing.assert_allclose(cgrph.d, fields.d, atol=atol)
    np.testing.assert_allclose(cgrph.d2, fields.d2, atol=atol)
    np.testing.assert_allclose(cgrph.n, fields.n, atol=atol)
    np.testing.assert_allclose(cgrph.wts, fields.wts, atol=atol)
    np.testing.assert_array_equal(cgrph.adj, np.asarray(fields.adj, dtype=int))
    np.testing.assert_allclose(cgrph.min(), np.asarray(fields.min).reshape(-1), atol=atol)
    np.testing.assert_allclose(cgrph.max(), np.asarray(fields.max).reshape(-1), atol=atol)
    np.testing.assert_array_equal([edge.npt for edge in cgrph.echnks], np.asarray(fields.edge_npt, dtype=int).reshape(-1))


def assert_arcparam_matches_fields(pdata: arcparam.ArcParamData, fields, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(pdata.plen, np.asarray(fields.plen).reshape(-1), atol=atol)
    np.testing.assert_allclose(pdata.pstrt, np.asarray(fields.pstrt).reshape(-1), atol=atol)
    np.testing.assert_allclose(pdata.cr, fields.cr, atol=atol)
    np.testing.assert_allclose(pdata.cd, fields.cd, atol=atol)
    np.testing.assert_allclose(pdata.cd2, fields.cd2, atol=atol)
    assert pdata.k == int(fields.k)
    assert pdata.dim == int(fields.dim)
    assert pdata.nch == int(fields.nch)
    np.testing.assert_allclose(pdata.eps, float(fields.eps), atol=atol)
    np.testing.assert_allclose(pdata.maxcond, float(fields.maxcond), atol=atol)


def padded_indices(rows: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=int)
    for i, row in enumerate(rows):
        row_arr = np.asarray(row, dtype=int).reshape(-1)
        if row_arr.size:
            out[i, : row_arr.size] = row_arr + 1
    return out


def test_top_level_domain_helpers_match_matlab_fixture():
    fixture = load_geometry_core().domain

    for actual, expected in zip(ellipse(fixture.t, 2.0, 0.35), (fixture.ellipse_r, fixture.ellipse_d, fixture.ellipse_d2)):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(
        starfish(fixture.t, 3, 0.2, [0.1, -0.25], np.pi / 7.0, 1.4),
        (fixture.starfish_r, fixture.starfish_d, fixture.starfish_d2),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(
        nonflatinterface(fixture.t, 0.7, 2.0, -0.4, 1.3),
        (fixture.nonflat_r, fixture.nonflat_d, fixture.nonflat_d2),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    np.testing.assert_allclose(redblue(6), fixture.redblue_even, atol=0.0)
    np.testing.assert_allclose(redblue(5), fixture.redblue_odd, atol=0.0)
    assert checkcurveparam(lambda t: ellipse(t, 2.0, 0.35), fixture.t, 3) == int(fixture.checkcurveparam_dim)
    assert bool(fixture.checkcurveparam_bad_shape)
    assert bool(fixture.checkcurveparam_bad_dim)
    with pytest.raises(ValueError, match="match input"):
        checkcurveparam(lambda t: np.ones((2, np.asarray(t).size + 1)), fixture.t, 1)
    with pytest.raises(ValueError, match="consistent"):
        checkcurveparam(
            lambda t: (np.ones((2, np.asarray(t).size)), np.ones((3, np.asarray(t).size))),
            fixture.t,
            2,
        )


def test_chnk_curve_helpers_match_matlab_fixture():
    fixture = load_geometry_core().curves

    for actual, expected in zip(
        curves.linefunc(fixture.t, [0.25, -0.5], [1.75, 0.75]),
        (fixture.line_r, fixture.line_d, fixture.line_d2),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(curves.fpara(fixture.t, 0.9, -0.15), (fixture.fpara_r, fixture.fpara_d, fixture.fpara_d2)):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(curves.fsine(fixture.t, 1.3, 2.4, -0.2), (fixture.fsine_r, fixture.fsine_d, fixture.fsine_d2)):
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    for actual, expected in zip(
        curves.bymode(load_geometry_core().domain.t, fixture.modes, fixture.center, fixture.scale),
        (fixture.bymode_r, fixture.bymode_d, fixture.bymode_d2),
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_domain_tree_and_region_helpers_match_matlab_fixture():
    fixture = load_geometry_core().regions
    tree = hypoct_uni(fixture.tree_points, 0.4, ext=fixture.tree_extent)

    assert tree.nlvl == int(fixture.tree_nlvl)
    np.testing.assert_array_equal(tree.lvp, np.asarray(fixture.tree_lvp, dtype=int))
    np.testing.assert_allclose(tree.lrt, fixture.tree_lrt, atol=0.0)
    np.testing.assert_allclose(np.column_stack([node.ctr for node in tree.nodes]), fixture.tree_ctr, atol=0.0)
    np.testing.assert_array_equal([0 if node.prnt is None else node.prnt + 1 for node in tree.nodes], fixture.tree_prnt)
    np.testing.assert_array_equal(padded_indices([node.xi for node in tree.nodes], fixture.tree_xi.shape[1]), fixture.tree_xi)
    np.testing.assert_array_equal(padded_indices([node.chld for node in tree.nodes], fixture.tree_chld.shape[1]), fixture.tree_chld)
    np.testing.assert_array_equal(padded_indices([node.nbor for node in tree.nodes], fixture.tree_nbor.shape[1]), fixture.tree_nbor)

    cg = chunkgraph(fixture.verts, np.asarray(fixture.edges, dtype=int) - 1, cparams={"nchmin": 4}, pref={"k": 6})
    outer = [[], [[0, 1, 2, 3]]]
    inner = [[[4, 5, 6, 7]]]

    assert pointinregion(cg, outer[1], [1.0, 1.0]) == int(fixture.point_inside)
    assert pointinregion(cg, outer[1], [3.0, 1.0]) == int(fixture.point_outside)
    assert regioninside(cg, outer, inner) == bool(fixture.region_inside)

    merged = mergeregions(cg, outer, inner)
    assert len(merged) == int(fixture.merged_region_count)
    assert [len(region) for region in merged] == list(np.asarray(fixture.merged_loop_count, dtype=int).reshape(-1))
    assert len(merged[1][0]) == int(fixture.merged_first_inner_edge_count)
    assert len(merged[1][1]) == int(fixture.merged_second_inner_edge_count)


def test_chunker_core_geometry_helpers_match_matlab_fixture():
    fixture = load_geometry_core().chunker
    base = chunker_from_fields(fixture.base)

    assert_chunker_matches_fields(base, fixture.base)
    np.testing.assert_allclose(base.weights(), fixture.weights, atol=1e-13)
    np.testing.assert_allclose(base.normals(), fixture.normals, atol=1e-13)
    np.testing.assert_allclose(base.tangents(), fixture.tangents, atol=1e-13)
    np.testing.assert_allclose(base.arclengthdens(), fixture.arclengthdens, atol=1e-13)
    np.testing.assert_allclose(base.signed_curvature(), fixture.signed_curvature, atol=1e-12)
    np.testing.assert_allclose(base.arclengthfun(), fixture.arclengthfun, atol=1e-12)
    np.testing.assert_allclose(base.arclengthder(fixture.uvals), fixture.arclengthder, atol=1e-11)
    np.testing.assert_allclose(base.chunkends([0, base.nch - 1])[0], fixture.chunkends_r, atol=1e-13)
    np.testing.assert_allclose(base.chunkends([0, base.nch - 1])[1], fixture.chunkends_tau, atol=1e-13)
    np.testing.assert_allclose(base.min(), fixture.min, atol=0.0)
    np.testing.assert_allclose(base.max(), fixture.max, atol=0.0)

    inds, adjs, info = base.sortinfo()
    np.testing.assert_array_equal(inds + 1, np.asarray(fixture.sort_inds, dtype=int))
    np.testing.assert_array_equal(adjs, np.asarray(fixture.sort_adjs, dtype=int))
    assert info["ier"] == int(fixture.sort_info.ier)
    assert info["ncomp"] == int(fixture.sort_info.ncomp)
    np.testing.assert_array_equal(info["nchs"], np.atleast_1d(np.asarray(fixture.sort_info.nchs, dtype=int)))
    np.testing.assert_array_equal(info["ifclosed"], np.atleast_1d(np.asarray(fixture.sort_info.ifclosed, dtype=bool)))
    assert_chunker_matches_fields(base.sort()[0], fixture.sorted)

    data_base = attach_data(base, fixture.data)
    actual_datares = data_base.datares({"idata": [0, 1], "ncoeff": 3, "tol": 1.0e-8})
    np.testing.assert_array_equal(actual_datares, np.asarray(fixture.datares, dtype=bool))


def test_chunker_flag_nearest_translate_and_uniform_helpers_match_matlab_fixture():
    fixture = load_geometry_core().chunker
    base = chunker_from_fields(fixture.base)

    flag_opts = {"fac": float(fixture.flag_fac)}
    expected_flagnear = np.asarray(fixture.flagnear, dtype=bool)
    np.testing.assert_array_equal(base.flagnear(fixture.flag_targets, flag_opts), expected_flagnear)
    np.testing.assert_array_equal(flagnear(base, fixture.flag_targets, flag_opts), expected_flagnear)

    rect_opts = {"rho": float(fixture.rect_rho)}
    expected_rect = np.asarray(fixture.flagnear_rectangle, dtype=bool)
    expected_grid = np.asarray(fixture.flagnear_rectangle_grid, dtype=bool)
    np.testing.assert_array_equal(base.flagnear_rectangle(fixture.rect_targets, rect_opts), expected_rect)
    np.testing.assert_array_equal(base.flagnear_rectangle_grid(fixture.rect_x, fixture.rect_y, rect_opts), expected_grid)
    np.testing.assert_array_equal(flagnear_rectangle(base, fixture.rect_targets, rect_opts), expected_rect)
    np.testing.assert_array_equal(flagnear_rectangle_grid(base, fixture.rect_x, fixture.rect_y, rect_opts), expected_grid)

    rn, dn, d2n, dist, tn, ichn = base.nearest(fixture.nearest_targets)
    np.testing.assert_allclose(rn, fixture.nearest_r, atol=2e-12)
    np.testing.assert_allclose(dn, fixture.nearest_d, atol=2e-12)
    np.testing.assert_allclose(d2n, fixture.nearest_d2, atol=2e-11)
    np.testing.assert_allclose(dist, fixture.nearest_dist, atol=2e-12)
    np.testing.assert_allclose(tn, fixture.nearest_t, atol=2e-12)
    np.testing.assert_array_equal(ichn + 1, np.asarray(fixture.nearest_ich, dtype=int))

    dirty = base.copy()
    dirty.n = np.zeros_like(dirty.n)
    dirty.wts = np.zeros_like(dirty.wts)
    assert_chunker_matches_fields(dirty.recompute_geometry(), fixture.recomputed, atol=2e-12)
    assert_chunker_matches_fields(fixture.translation_vector + base, fixture.translated_left, atol=2e-12)
    assert_chunker_matches_fields(base + fixture.translation_vector, fixture.translated_right, atol=2e-12)

    uniform = chunkerfuncuni(
        lambda t: starfish(t, 4, 0.15, [0.05, -0.1], 0.2, 0.9),
        int(fixture.chunkerfuncuni_nch),
        {
            "ta": float(fixture.chunkerfuncuni_cparams.ta),
            "tb": float(fixture.chunkerfuncuni_cparams.tb),
            "ifclosed": bool(fixture.chunkerfuncuni_cparams.ifclosed),
        },
        {"k": int(fixture.chunkerfuncuni_pref.k)},
    )
    assert_chunker_matches_fields(uniform, fixture.chunkerfuncuni, atol=2e-12)


def test_chunker_refinement_and_reconstruction_helpers_match_matlab_fixture():
    fixture = load_geometry_core().chunker
    base = chunker_from_fields(fixture.base)
    data_base = attach_data(base, fixture.data)

    assert_chunker_matches_fields(base.copy().split(1, stype="t"), fixture.split_param, atol=2e-12)
    assert_chunker_matches_fields(
        base.refine({"splitchunks": [1], "lvlr": "n", "stype": "t", "nover": 1}),
        fixture.refined,
        atol=2e-12,
    )

    upsampled, sigmaup = data_base.upsample(14, fixture.sigma)
    assert_chunker_matches_fields(upsampled, fixture.upsampled, atol=2e-12)
    np.testing.assert_allclose(upsampled.data, fixture.upsampled_data, atol=2e-12)
    np.testing.assert_allclose(sigmaup, fixture.sigmaup, atol=2e-12)

    arcresampled, arc_eps = base.arcresample()
    assert_chunker_matches_fields(arcresampled, fixture.arcresampled, atol=5e-12)
    np.testing.assert_allclose(arc_eps, fixture.arcresample_eps, rtol=0.1)

    assert_chunker_matches_fields(base.rotate(0.37, [0.2, -0.1], [-0.3, 0.4]), fixture.rotated, atol=2e-12)
    assert_chunker_matches_fields(base.reflect(-0.2, [0.1, 0.2], [0.25, -0.35]), fixture.reflected, atol=2e-12)
    assert_chunker_matches_fields(base.reverse(), fixture.reversed, atol=2e-12)
    assert_chunker_matches_fields(chunkerpoints(base.r, {"ifclosed": True}), fixture.points_from_r, atol=2e-12)
    assert_chunker_matches_fields(
        chunkerpoints({"r": base.r, "d": 2.0 * base.d, "d2": 3.0 * base.d2}, {"ifclosed": True}),
        fixture.points_explicit,
        atol=2e-12,
    )

    circ1 = attach_data(chunker_from_fields(fixture.circ1), fixture.circ1_data)
    circ2 = attach_data(chunker_from_fields(fixture.circ2), fixture.circ2_data)
    merged = merge([circ1, circ2])
    assert_chunker_matches_fields(merged, fixture.merged_circles, atol=2e-12)
    np.testing.assert_allclose(merged.data, fixture.merged_circles_data, atol=1e-14)


def test_chnk_geometry_helpers_match_matlab_fixture():
    root = load_geometry_core()
    fixture = root.geometry
    base = chunker_from_fields(root.chunker.base)
    ptinfo = {"r": fixture.ptinfo.r, "d": fixture.ptinfo.d, "d2": fixture.ptinfo.d2}

    np.testing.assert_allclose(perp(ptinfo["d"]), fixture.perp, atol=0.0)
    np.testing.assert_allclose(normal2d(ptinfo), fixture.normal2d, atol=1e-13)
    np.testing.assert_allclose(curvature2d(ptinfo), fixture.curvature2d, atol=1e-12)

    _, _, u, _ = lege.exps(base.k)
    ts, rs, ds, d2s, dist2s = chunk_nearparam(base.r[:, :, 1], fixture.targets, t=base.tstor, u=u)
    np.testing.assert_allclose(ts, fixture.near_t, atol=1e-12)
    np.testing.assert_allclose(rs, fixture.near_r, atol=1e-12)
    np.testing.assert_allclose(ds, fixture.near_d, atol=1e-12)
    np.testing.assert_allclose(d2s, fixture.near_d2, atol=1e-11)
    np.testing.assert_allclose(dist2s, fixture.near_dist2, atol=1e-12)

    actual_pairs = flagself(fixture.flagself_src, fixture.flagself_targ) + 1
    np.testing.assert_array_equal(actual_pairs, np.asarray(fixture.flagself_pairs, dtype=int))


def test_arcparam_helpers_match_matlab_fixture():
    root = load_geometry_core()
    fixture = root.arcparam
    base = chunker_from_fields(root.chunker.base)

    full = arcparam.init(base)
    assert_arcparam_matches_fields(full, fixture.full, atol=3e-12)

    r, d, d2 = arcparam.eval(fixture.eval_s, full)
    np.testing.assert_allclose(r, fixture.eval_r, atol=2e-12)
    np.testing.assert_allclose(d, fixture.eval_d, atol=2e-12)
    np.testing.assert_allclose(d2, fixture.eval_d2, atol=8e-12)

    rn, dn, d2n = arcparam.eval(fixture.node_s, full)
    np.testing.assert_allclose(rn, fixture.node_r, atol=2e-12)
    np.testing.assert_allclose(dn, fixture.node_d, atol=2e-12)
    np.testing.assert_allclose(d2n, fixture.node_d2, atol=8e-12)

    selected = arcparam.init(base, np.asarray(fixture.selected_ich, dtype=int).reshape(-1) - 1)
    assert_arcparam_matches_fields(selected, fixture.selected, atol=3e-12)
    rs, ds, d2s = arcparam.eval(fixture.selected_eval_s, selected)
    np.testing.assert_allclose(rs, fixture.selected_eval_r, atol=2e-12)
    np.testing.assert_allclose(ds, fixture.selected_eval_d, atol=2e-12)
    np.testing.assert_allclose(d2s, fixture.selected_eval_d2, atol=8e-12)


def test_chunkgraph_helpers_match_matlab_fixture():
    fixture = load_geometry_core().chunkgraph
    cg = chunkgraph(fixture.verts, np.asarray(fixture.edges, dtype=int) - 1, cparams={"nchmin": 4}, pref={"k": 6})

    np.testing.assert_array_equal(cg.edgesendverts + 1, np.asarray(fixture.edgesendverts, dtype=int))
    np.testing.assert_array_equal(cg.v2emat, np.asarray(fixture.v2emat, dtype=int))
    assert cg.npt == int(fixture.npt)
    np.testing.assert_allclose(cg.r, fixture.r, atol=1e-13)
    np.testing.assert_allclose(cg.d, fixture.d, atol=1e-13)
    np.testing.assert_allclose(cg.d2, fixture.d2, atol=1e-13)
    np.testing.assert_allclose(cg.n, fixture.n, atol=1e-13)
    np.testing.assert_allclose(cg.wts, fixture.wts, atol=1e-13)
    np.testing.assert_array_equal(cg.adj, np.asarray(fixture.adj, dtype=int))
    np.testing.assert_allclose(cg.min(), fixture.min, atol=0.0)
    np.testing.assert_allclose(cg.max(), fixture.max, atol=0.0)
    np.testing.assert_allclose(cg.onesmat(), fixture.onesmat, atol=1e-13)
    np.testing.assert_allclose(cg.normonesmat(), fixture.normonesmat, atol=1e-13)

    src = cg.sourceinfo
    np.testing.assert_allclose(src.r, fixture.sourceinfo_r, atol=1e-13)
    np.testing.assert_allclose(src.n, fixture.sourceinfo_n, atol=1e-13)
    np.testing.assert_allclose(src.d, fixture.sourceinfo_d, atol=1e-13)
    np.testing.assert_allclose(src.d2, fixture.sourceinfo_d2, atol=1e-13)
    np.testing.assert_allclose(src.w, np.asarray(fixture.sourceinfo_w).reshape(-1), atol=1e-13)
    np.testing.assert_array_equal(cg.edgeids([0, 2]) + 1, np.asarray(fixture.edgeids, dtype=int))

    sliced = cg.slicegraph([0, 2])
    np.testing.assert_array_equal(sliced.edgesendverts + 1, np.asarray(fixture.sliced_edgesendverts, dtype=int))
    np.testing.assert_array_equal(sliced.v2emat, np.asarray(fixture.sliced_v2emat, dtype=int))
    assert sliced.npt == int(fixture.sliced_npt)

    np.testing.assert_allclose((cg + np.array([0.4, -0.25])).verts, fixture.translated_verts, atol=1e-14)
    np.testing.assert_allclose(cg.transform(np.array([[1.1, 0.2], [-0.3, 0.9]])).verts, fixture.transformed_verts, atol=1e-14)
    np.testing.assert_allclose(cg.rotate(0.31, [0.5, 0.5], [0.1, -0.2]).verts, fixture.rotated_verts, atol=1e-14)
    np.testing.assert_allclose(cg.reflect(-0.15, [0.5, 0.5], [0.25, -0.1]).verts, fixture.reflected_verts, atol=1e-14)

    copied = cg.copy()
    copied.verts[0, 0] += float(fixture.copy_delta)
    copied.echnks[0].rstor[0, 0, 0] -= float(fixture.copy_delta)
    np.testing.assert_allclose(cg.verts[0, 0], fixture.copy_source_vert, atol=0.0)
    np.testing.assert_allclose(copied.verts[0, 0], fixture.copy_mutated_vert, atol=0.0)
    np.testing.assert_allclose(cg.echnks[0].rstor[0, 0, 0], fixture.copy_source_edge_node, atol=0.0)
    np.testing.assert_allclose(copied.echnks[0].rstor[0, 0, 0], fixture.copy_mutated_edge_node, atol=0.0)


def test_chunkgraph_region_flag_operator_and_conversion_helpers_match_matlab_fixture():
    fixture = load_geometry_core().chunkgraph
    cg = chunkgraph(fixture.verts, np.asarray(fixture.edges, dtype=int) - 1, cparams={"nchmin": 4}, pref={"k": 6})

    for ivert, (edges, signs) in enumerate(cg.procverts()):
        count = int(np.asarray(fixture.procverts_counts, dtype=int).reshape(-1)[ivert])
        np.testing.assert_array_equal(edges + 1, np.asarray(fixture.procverts_edges, dtype=int)[:count, ivert])
        np.testing.assert_array_equal(signs, np.asarray(fixture.procverts_signs, dtype=int)[:count, ivert])

    regions = cg.findregions()
    assert len(regions) == int(fixture.region_count)
    np.testing.assert_array_equal(np.asarray(regions[0][0], dtype=int) + 1, np.asarray(fixture.region_first_loop, dtype=int))
    np.testing.assert_array_equal(np.asarray(regions[1][0], dtype=int), np.asarray(fixture.region_second_loop, dtype=int))

    refined = cg.refine({"nover": int(fixture.refine_opts.nover), "lvlr": str(fixture.refine_opts.lvlr)})
    assert_chunkgraph_matches_fields(refined, fixture.refined, atol=2e-12)

    flag_opts = {"fac": float(fixture.flag_fac)}
    np.testing.assert_array_equal(cg.flagnear(fixture.flag_targets, flag_opts), np.asarray(fixture.flagnear, dtype=bool))

    rect_opts = {"rho": float(fixture.rect_rho)}
    np.testing.assert_array_equal(
        cg.flagnear_rectangle(fixture.rect_targets, rect_opts),
        np.asarray(fixture.flagnear_rectangle, dtype=bool),
    )
    np.testing.assert_array_equal(
        cg.flagnear_rectangle_grid(fixture.rect_x, fixture.rect_y, rect_opts),
        np.asarray(fixture.flagnear_rectangle_grid, dtype=bool),
    )

    assert_chunkgraph_matches_fields(fixture.translation_vector + cg, fixture.translated, atol=2e-12)
    assert_chunkgraph_matches_fields(cg + fixture.translation_vector, fixture.translated_right, atol=2e-12)
    assert_chunkgraph_matches_fields(fixture.transform_matrix @ cg, fixture.transformed, atol=2e-12)
    assert_chunkgraph_matches_fields(fixture.scale * cg, fixture.scaled_left, atol=2e-12)
    assert_chunkgraph_matches_fields(cg * fixture.scale, fixture.scaled_right, atol=2e-12)

    closed = chunker_from_fields(load_geometry_core().chunker.circ1)
    open_line, _ = chunkerfunc(
        lambda t: curves.linefunc(t, [2.0, -0.5], [3.0, 0.2]),
        {"ifclosed": False, "ifrefine": False, "nchmin": 2, "eps": 1.0e-3, "ta": 0.0, "tb": 1.0},
        {"k": 8},
    )
    assert int(fixture.tochunkgraph_closed_edge_count) == 1
    assert int(fixture.tochunkgraph_open_edge_count) == 1
    assert_chunkgraph_matches_fields(tochunkgraph(closed), fixture.tochunkgraph_closed, atol=2e-12)
    assert_chunkgraph_matches_fields(tochunkgraph(open_line), fixture.tochunkgraph_open, atol=2e-12)

    np.testing.assert_array_equal(chunkgraphinregion(cg, fixture.region_points), np.asarray(fixture.inregion_points, dtype=int))
    grid_ids = chunkgraphinregion(cg, (fixture.region_x, fixture.region_y)).reshape(-1, order="F")
    np.testing.assert_array_equal(grid_ids, np.asarray(fixture.inregion_grid, dtype=int))
