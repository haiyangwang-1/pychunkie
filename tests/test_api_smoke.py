import chunkie
from chunkie import geometry, rcip, system


def test_top_level_rewrite_api_exports_design_objects():
    expected = {
        "BoundaryEquation",
        "BoundaryTrace",
        "ChunkGraph",
        "Chunker",
        "Constraint",
        "ConstraintTerm",
        "Density",
        "IntegralSystem",
        "Kernel",
        "LaplaceExteriorDirichletSystem",
        "LayerPotential",
        "SystemMatrix",
        "SystemSolution",
        "geometry",
        "kernel",
        "kernels",
        "quadrature",
        "rcip",
        "system",
    }
    assert set(chunkie.__all__) == expected
    for name in expected:
        assert hasattr(chunkie, name)


def test_old_matlab_shaped_operator_names_are_not_active_api():
    assert not hasattr(chunkie, "chunkermat")
    assert not hasattr(chunkie, "chunkerkerneval")


def test_geometry_rcip_and_system_exports_include_new_rewrite_helpers():
    for name in (
        "NearestPoint",
        "affine",
        "bernstein_ellipse",
        "bernstein_panel_image",
        "bernstein_radius",
        "flagnear_rectangle",
        "flagnear_rectangle_grid",
        "nearest_point",
        "reflect",
        "refine",
        "rotate",
        "scale",
        "translate",
    ):
        assert name in geometry.__all__
        assert hasattr(geometry, name)

    for name in (
        "build_block_prolongation",
        "build_split_panel_prolongation",
        "recursive_schur_compress",
        "schur_compress_block",
    ):
        assert name in rcip.__all__
        assert hasattr(rcip, name)

    assert "build_corrections" in system.__all__
    assert hasattr(system, "build_corrections")
    assert "build_rcip_state" in system.__all__
    assert hasattr(system, "build_rcip_state")
    assert "fmm_matvec" in system.__all__
    assert hasattr(system, "fmm_matvec")
    assert "matrix_free_matvec" in system.__all__
    assert hasattr(system, "matrix_free_matvec")
