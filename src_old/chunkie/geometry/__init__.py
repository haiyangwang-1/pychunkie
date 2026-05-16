"""Geometry data structures, curve helpers, and point data."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .chunker import (
    Chunker,
    ChunkerPref,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerpoints,
    chunkerpoly,
    merge,
)
from .chunkgraph import ChunkGraph, chunkgraphinregion, find_edge_regions, tochunkgraph
from .domain import (
    HypOctNode,
    HypOctTree,
    checkcurveparam,
    ellipse,
    hypoct_uni,
    mergeregions,
    nonflatinterface,
    pointinregion,
    redblue,
    regioninside,
    starfish,
)
from .pointinfo import PointInfo

_SUBMODULES = {"chunker", "chunkgraph", "curves", "domain", "pointinfo"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Chunker",
    "ChunkerPref",
    "ChunkGraph",
    "HypOctNode",
    "HypOctTree",
    "checkcurveparam",
    "chunkerfit",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "chunker",
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
]
