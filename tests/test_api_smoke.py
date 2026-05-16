import chunkie


def test_top_level_rewrite_api_exports_design_objects():
    expected = {
        "BoundaryEquation",
        "BoundaryTrace",
        "ChunkGraph",
        "Chunker",
        "Density",
        "IntegralSystem",
        "Kernel",
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
