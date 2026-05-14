import numpy as np

import chunkie.operators as operators_mod
from chunkie import chunkerfunc, chunkerinterior, chunkerkerneval, chunkermat, chunkerpoly, kernel
from chunkie.operators import core as operators_core


def circle(t):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def test_keyword_migration_forms_for_geometry_and_operator_helpers():
    boundary, _ = chunkerfunc(circle, min_chunks=4, order=6)
    targets = np.array([[0.0, 1.2], [0.0, 0.0]])
    direct_inside = chunkerinterior(boundary, targets)

    flam_inside = chunkerinterior(
        boundary,
        targets,
        acceleration="flam",
        rank_or_tol=1.0e-8,
        proxy=False,
    )
    near_values = chunkerkerneval(
        boundary,
        kernel("lap", "s"),
        np.ones(boundary.npt),
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


def test_chunkerinterior_forwards_accelerated_keyword_options(monkeypatch):
    boundary, _ = chunkerfunc(circle, min_chunks=3, order=4)
    targets = np.array([[0.0, 1.2], [0.0, 0.0]])
    captured = {}

    def fake_eval(boundary0, kernel0, density, target, options=None, **kwargs):
        captured.update({} if options is None else options)
        captured.update(kwargs)
        return -np.ones(target.r.shape[1])

    monkeypatch.setattr(operators_core, "chunkerkerneval", fake_eval)

    actual = operators_mod.chunkerinterior(
        boundary,
        targets,
        acceleration="flam",
        rank_or_tol=1.0e-7,
        proxy=False,
        near_factor=0.25,
        close_correction=False,
    )

    np.testing.assert_array_equal(actual, [True, True])
    assert captured["acceleration"] == "flam"
    assert captured["rank_or_tol"] == 1.0e-7
    assert captured["useproxy"] is False
    assert captured["fac"] == 0.25


def test_operator_python_first_keywords_map_to_backend_options(monkeypatch):
    boundary, _ = chunkerfunc(circle, min_chunks=3, order=4)
    captured = {}

    def fake_chunkerflam(boundary0, kernel0, dval=0.0, options=None, **kwargs):
        captured["dval"] = dval
        captured["options"] = dict(options)
        captured["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(operators_core, "chunkerflam", fake_chunkerflam)

    chunkermat(
        boundary,
        kernel("lap", "s"),
        acceleration="flam",
        dval=1.0,
        flam_occupancy=7,
        rank_or_tol=1.0e-6,
        proxy=False,
    )

    assert captured["dval"] == 1.0
    assert captured["options"]["acceleration"] == "flam"
    assert captured["options"]["occ"] == 7
    assert captured["options"]["rank_or_tol"] == 1.0e-6
    assert captured["options"]["useproxy"] is False


def test_rcip_keyword_options_normalize_to_internal_names():
    options = operators_mod._normalize_public_options(
        None,
        rcip_subdivisions=6,
        rcip_save_depth=4,
        rcip_eval_depth=3,
    )

    assert options["nsub"] == 6
    assert options["rcip_savedepth"] == 4
    assert options["rcip_eval_depth"] == 3
