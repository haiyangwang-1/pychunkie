import numpy as np

from chunkie.geometry import chunker_from_polygon
from chunkie.kernels import kernel
from chunkie.quadrature import dense_panel_operator_matrix
from chunkie.system import build_corrections, build_panel_correction


def test_panel_correction_replaces_component_major_dense_block():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=6)
    stokes_s = kernel("stokes", selector="s", viscosity=1.2)
    matrix = dense_panel_operator_matrix(boundary.pointinfo, boundary.pointinfo, stokes_s)
    original = matrix.copy()
    target_panel_id = 1
    target_point_ids = target_panel_id * boundary.quadrature_order + np.arange(
        boundary.quadrature_order
    )

    correction = build_panel_correction(
        boundary,
        boundary,
        stokes_s,
        source_panel_id=0,
        target_point_ids=target_point_ids,
        method="adaptive",
    )
    correction.apply_to(matrix)

    assert correction.values.shape == (
        stokes_s.output_dim * boundary.quadrature_order,
        stokes_s.input_dim * boundary.quadrature_order,
    )
    np.testing.assert_allclose(matrix[np.ix_(correction.rows, correction.columns)], correction.values)

    unchanged = np.ones(matrix.shape, dtype=bool)
    unchanged[np.ix_(correction.rows, correction.columns)] = False
    np.testing.assert_allclose(matrix[unchanged], original[unchanged])


def test_panel_correction_uses_ggq_for_laplace_single_layer_self_block():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=8)
    laplace_s = kernel("laplace", selector="s")
    matrix = dense_panel_operator_matrix(boundary.pointinfo, boundary.pointinfo, laplace_s)
    target_point_ids = np.arange(boundary.quadrature_order)

    correction = build_panel_correction(
        boundary,
        boundary,
        laplace_s,
        source_panel_id=0,
        target_point_ids=target_point_ids,
        method="ggq",
    )
    correction.apply_to(matrix)

    assert np.isfinite(correction.values).all()
    np.testing.assert_allclose(matrix[np.ix_(correction.rows, correction.columns)], correction.values)


def test_build_corrections_selects_laplace_self_blocks_with_ggq():
    boundary = chunker_from_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], quadrature_order=8)
    laplace_s = kernel("laplace", selector="s")
    matrix = dense_panel_operator_matrix(boundary.pointinfo, boundary.pointinfo, laplace_s)

    corrections = build_corrections(
        boundary,
        boundary,
        laplace_s,
        method="auto",
        include_near=False,
    )
    for correction in corrections:
        correction.apply_to(matrix)

    assert len(corrections) == boundary.panel_count
    assert {correction.diagnostics["method"] for correction in corrections} == {"ggq"}
    assert all(np.isfinite(correction.values).all() for correction in corrections)
    for panel_id, correction in enumerate(corrections):
        expected_points = panel_id * boundary.quadrature_order + np.arange(
            boundary.quadrature_order
        )
        np.testing.assert_array_equal(correction.columns, expected_points)
        np.testing.assert_allclose(matrix[np.ix_(correction.rows, correction.columns)], correction.values)
