import importlib
import sys
from contextlib import contextmanager


_MISSING = object()


@contextmanager
def temporarily_unloaded(*names):
    saved = {name: sys.modules.get(name, _MISSING) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_top_level_public_exports_are_stable():
    chunkie = importlib.import_module("chunkie")

    expected = {
        "Chunker",
        "ChunkerPref",
        "ChunkGraph",
        "Kernel",
        "ChunkerFLAMMatrix",
        "ChunkerFMMMatrix",
        "PointInfo",
        "HypOctNode",
        "HypOctTree",
        "checkcurveparam",
        "chunker",
        "chunkerfit",
        "chunkerflam",
        "chunkerfunc",
        "chunkerfuncuni",
        "chunkerpoints",
        "chunkerpoly",
        "chunkgraph",
        "chunkgraphinregion",
        "find_edge_regions",
        "chunkerinterior",
        "chunkerintegral",
        "chunkerkerneval",
        "chunkerkernevalmat",
        "chunkermat",
        "chunkermatapply",
        "chunkerpref",
        "ellipse",
        "hypoct_uni",
        "kernel",
        "mergeregions",
        "merge",
        "nonflatinterface",
        "lege",
        "pointinregion",
        "pointinfo",
        "redblue",
        "regioninside",
        "starfish",
        "tochunkgraph",
    }
    assert set(chunkie.__all__) == expected
    for name in expected:
        assert hasattr(chunkie, name)


def test_acceleration_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.acceleration.flam",
        "chunkie.acceleration",
    ):
        acceleration = importlib.import_module("chunkie.acceleration")
        expected = {"flam"}
        assert set(acceleration.__all__) == expected
        assert "chunkie.acceleration.flam" not in sys.modules

        from chunkie.acceleration import flam

        assert flam.__name__ == "chunkie.acceleration.flam"


def test_geometry_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.geometry.curves",
        "chunkie.geometry.predicates",
        "chunkie.geometry",
    ):
        geometry = importlib.import_module("chunkie.geometry")
        expected = {
            "chunk_nearparam",
            "curvature2d",
            "curves",
            "flagnear",
            "flagnear_rectangle",
            "flagnear_rectangle_grid",
            "flagself",
            "normal2d",
            "perp",
            "predicates",
        }
        assert set(geometry.__all__) == expected
        assert "chunkie.geometry.curves" not in sys.modules
        assert "chunkie.geometry.predicates" not in sys.modules

        from chunkie.geometry import curves, flagnear

        assert curves.__name__ == "chunkie.geometry.curves"
        assert callable(flagnear)


def test_kernels_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.kernels.biharmonic",
        "chunkie.kernels.elasticity",
        "chunkie.kernels.helmholtz",
        "chunkie.kernels.helmholtz_1d",
        "chunkie.kernels.laplace",
        "chunkie.kernels.stokes",
        "chunkie.kernels",
    ):
        kernels = importlib.import_module("chunkie.kernels")
        expected = {"biharmonic", "elasticity", "helmholtz", "helmholtz_1d", "laplace", "stokes"}
        assert set(kernels.__all__) == expected
        assert "chunkie.kernels.laplace" not in sys.modules
        assert "chunkie.kernels.helmholtz" not in sys.modules

        from chunkie.kernels import laplace, stokes

        assert laplace.__name__ == "chunkie.kernels.laplace"
        assert callable(stokes.kern)


def test_quadrature_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.quadrature.adaptive",
        "chunkie.quadrature.ggq",
        "chunkie.quadrature.panel",
        "chunkie.quadrature.rcip",
        "chunkie.quadrature",
    ):
        quadrature = importlib.import_module("chunkie.quadrature")
        expected = {"adaptive", "ggq", "native", "panel", "rcip"}
        assert set(quadrature.__all__) == expected
        assert "chunkie.quadrature.ggq" not in sys.modules
        assert "chunkie.quadrature.rcip" not in sys.modules

        from chunkie.quadrature import ggq, rcip

        assert ggq.__name__ == "chunkie.quadrature.ggq"
        assert rcip.__name__ == "chunkie.quadrature.rcip"


def test_numerics_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.numerics.arcparam",
        "chunkie.numerics.smoother",
        "chunkie.numerics.special",
        "chunkie.numerics",
    ):
        numerics = importlib.import_module("chunkie.numerics")
        expected = {"arcparam", "smoother", "special"}
        assert set(numerics.__all__) == expected
        assert "chunkie.numerics.arcparam" not in sys.modules
        assert "chunkie.numerics.smoother" not in sys.modules

        from chunkie.numerics import arcparam, special

        assert arcparam.__name__ == "chunkie.numerics.arcparam"
        assert callable(special.absconvgauss)
