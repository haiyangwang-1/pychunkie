import importlib
import sys


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


def test_chnk_public_exports_are_stable_and_lazy():
    for name in [
        "chunkie.chnk.flam",
        "chunkie.chnk.smoother",
    ]:
        sys.modules.pop(name, None)
    sys.modules.pop("chunkie.chnk", None)

    chnk = importlib.import_module("chunkie.chnk")
    expected = {
        "chunk_nearparam",
        "arcparam",
        "biharm2d",
        "curvature2d",
        "curves",
        "elast2d",
        "flam",
        "flagnear",
        "flagnear_rectangle",
        "flagnear_rectangle_grid",
        "flagself",
        "geometry",
        "helm1d",
        "helm2d",
        "lap2d",
        "normal2d",
        "perp",
        "smoother",
        "spcl",
        "stok2d",
    }
    assert set(chnk.__all__) == expected
    assert "chunkie.chnk.flam" not in sys.modules

    from chunkie.chnk import flagnear, lap2d

    assert lap2d.__name__ == "chunkie.chnk.lap2d"
    assert callable(flagnear)


def test_quadrature_public_exports_are_stable_and_lazy():
    for name in [
        "chunkie.quadrature.adaptive",
        "chunkie.quadrature.ggq",
        "chunkie.quadrature.panel",
        "chunkie.quadrature.rcip",
    ]:
        sys.modules.pop(name, None)
    sys.modules.pop("chunkie.quadrature", None)

    quadrature = importlib.import_module("chunkie.quadrature")
    expected = {"adaptive", "ggq", "native", "panel", "rcip"}
    assert set(quadrature.__all__) == expected
    assert "chunkie.quadrature.ggq" not in sys.modules
    assert "chunkie.quadrature.rcip" not in sys.modules

    from chunkie.quadrature import ggq, rcip

    assert ggq.__name__ == "chunkie.quadrature.ggq"
    assert rcip.__name__ == "chunkie.quadrature.rcip"
