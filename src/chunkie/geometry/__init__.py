"""Geometry data structures, curve helpers, and point data."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .chunker import (
    Chunker,
    ChunkerPref,
    chunker,
    chunkerfit,
    chunkerfunc,
    chunkerfuncuni,
    chunkerpoints,
    chunkerpoly,
    chunkerpref,
    merge,
)
from .chunkgraph import ChunkGraph, chunkgraph, chunkgraphinregion, find_edge_regions, tochunkgraph
from .pointinfo import PointInfo


_SUBMODULES = {"chunker", "chunkgraph", "curves", "pointinfo"}


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
    "chunker",
    "chunkerfit",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "chunkerpref",
    "chunkgraph",
    "chunkgraphinregion",
    "curves",
    "find_edge_regions",
    "merge",
    "PointInfo",
    "tochunkgraph",
]
