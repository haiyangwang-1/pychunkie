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
        "ChunkerRCIPMatrix",
        "PointInfo",
        "RCIPContext",
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
    removed_modules = {}
    for name in [
        "chunkie.chnk.quadggq",
        "chunkie.chnk.quadadap",
        "chunkie.chnk.flam",
        "chunkie.chnk.pquad",
        "chunkie.chnk.rcip",
        "chunkie.chnk.smoother",
    ]:
        removed_modules[name] = sys.modules.pop(name, None)
    removed_modules["chunkie.chnk"] = sys.modules.pop("chunkie.chnk", None)

    try:
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
            "pquad",
            "quadadap",
            "quadggq",
            "quadnative",
            "rcip",
            "smoother",
            "spcl",
            "stok2d",
        }
        assert set(chnk.__all__) == expected
        assert "chunkie.chnk.quadggq" not in sys.modules
        assert "chunkie.chnk.flam" not in sys.modules
        assert "chunkie.chnk.rcip" not in sys.modules

        from chunkie.chnk import flagnear, lap2d, quadggq

        assert lap2d.__name__ == "chunkie.chnk.lap2d"
        assert quadggq.__name__ == "chunkie.chnk.quadggq"
        assert callable(flagnear)
    finally:
        for name, module in removed_modules.items():
            if module is not None:
                sys.modules[name] = module
