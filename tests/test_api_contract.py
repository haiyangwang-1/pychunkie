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
        "ChunkerRCIPMatrix",
        "PointInfo",
        "RCIPContext",
        "HypOctNode",
        "HypOctTree",
        "checkcurveparam",
        "chunkerfit",
        "chunkerflam",
        "chunkerfunc",
        "chunkerfuncuni",
        "chunkerpoints",
        "chunkerpoly",
        "chunkgraphinregion",
        "find_edge_regions",
        "chunkerinterior",
        "chunkerintegral",
        "chunkerkerneval",
        "chunkerkernevalmat",
        "chunkermat",
        "chunkermatapply",
        "ellipse",
        "hypoct_uni",
        "kernel",
        "mergeregions",
        "merge",
        "nonflatinterface",
        "lege",
        "pointinregion",
        "redblue",
        "regioninside",
        "rcip",
        "starfish",
        "tochunkgraph",
    }
    assert set(chunkie.__all__) == expected
    for name in expected:
        assert hasattr(chunkie, name)
    assert not hasattr(chunkie, "chunker")
    assert not hasattr(chunkie, "chunkerpref")
    assert not hasattr(chunkie, "chunkgraph")


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
        "chunkie.geometry.chunker",
        "chunkie.geometry.chunkgraph",
        "chunkie.geometry.curves",
        "chunkie.geometry.domain",
        "chunkie.geometry.pointinfo",
        "chunkie.geometry",
    ):
        geometry = importlib.import_module("chunkie.geometry")
        expected = {
            "Chunker",
            "ChunkerPref",
            "ChunkGraph",
            "HypOctNode",
            "HypOctTree",
            "checkcurveparam",
            "chunker",
            "chunkerfit",
            "chunkerfunc",
            "chunkerfuncuni",
            "chunkerpoints",
            "chunkerpoly",
            "chunkgraph",
            "chunkgraphinregion",
            "curves",
            "domain",
            "ellipse",
            "find_edge_regions",
            "hypoct_uni",
            "mergeregions",
            "merge",
            "nonflatinterface",
            "PointInfo",
            "pointinregion",
            "redblue",
            "regioninside",
            "starfish",
            "tochunkgraph",
        }
        assert set(geometry.__all__) == expected
        from chunkie.geometry import (
            PointInfo,
            chunkerfunc,
            curves,
            domain,
        )
        from chunkie.geometry import (
            chunker as chunker_module,
        )
        from chunkie.geometry import (
            chunkgraph as chunkgraph_module,
        )

        assert chunker_module.__name__ == "chunkie.geometry.chunker"
        assert chunkgraph_module.__name__ == "chunkie.geometry.chunkgraph"
        assert curves.__name__ == "chunkie.geometry.curves"
        assert domain.__name__ == "chunkie.geometry.domain"
        assert PointInfo.__name__ == "PointInfo"
        assert callable(chunkerfunc)


def test_kernels_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.kernels.biharmonic",
        "chunkie.kernels.elasticity",
        "chunkie.kernels.factory",
        "chunkie.kernels.helmholtz",
        "chunkie.kernels.helmholtz_1d",
        "chunkie.kernels.laplace",
        "chunkie.kernels.stokes",
        "chunkie.kernels",
    ):
        kernels = importlib.import_module("chunkie.kernels")
        expected = {
            "biharmonic",
            "elasticity",
            "factory",
            "helmholtz",
            "helmholtz_1d",
            "Kernel",
            "kernel",
            "laplace",
            "stokes",
        }
        assert set(kernels.__all__) == expected
        assert "chunkie.kernels.laplace" not in sys.modules
        assert "chunkie.kernels.helmholtz" not in sys.modules

        from chunkie.kernels import Kernel, kernel, laplace, stokes

        assert laplace.__name__ == "chunkie.kernels.laplace"
        assert Kernel.__name__ == "Kernel"
        assert callable(kernel)
        assert callable(stokes.kernel)


def test_quadrature_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.quadrature.adaptive",
        "chunkie.quadrature.ggq",
        "chunkie.quadrature.native",
        "chunkie.quadrature.panel",
        "chunkie.quadrature",
    ):
        quadrature = importlib.import_module("chunkie.quadrature")
        expected = {"adaptive", "ggq", "native", "panel"}
        assert set(quadrature.__all__) == expected
        assert "chunkie.quadrature.ggq" not in sys.modules

        from chunkie.quadrature import ggq, panel

        assert ggq.__name__ == "chunkie.quadrature.ggq"
        assert panel.__name__ == "chunkie.quadrature.panel"


def test_misc_public_exports_are_stable_and_lazy():
    with temporarily_unloaded(
        "chunkie.misc.absconvgauss",
        "chunkie.misc.arcparam",
        "chunkie.misc.smoother",
        "chunkie.misc",
    ):
        misc = importlib.import_module("chunkie.misc")
        expected = {"absconvgauss", "arcparam", "smoother"}
        assert set(misc.__all__) == expected
        assert "chunkie.misc.arcparam" not in sys.modules
        assert "chunkie.misc.smoother" not in sys.modules

        from chunkie.misc import absconvgauss, arcparam

        assert arcparam.__name__ == "chunkie.misc.arcparam"
        assert callable(absconvgauss.absconvgauss)
