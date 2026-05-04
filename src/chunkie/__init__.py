"""Python port of the MATLAB chunkIE package."""

from . import lege
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
from .chunkgraph import ChunkGraph, chunkgraph, chunkgraphinregion, tochunkgraph
from .kernel import Kernel, kernel
from .operators import (
    PointInfo,
    chunkerinterior,
    chunkerintegral,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
    pointinfo,
)

__all__ = [
    "Chunker",
    "ChunkerPref",
    "ChunkGraph",
    "Kernel",
    "PointInfo",
    "chunker",
    "chunkerfit",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
    "chunkgraph",
    "chunkgraphinregion",
    "chunkerinterior",
    "chunkerintegral",
    "chunkerkerneval",
    "chunkerkernevalmat",
    "chunkermat",
    "chunkermatapply",
    "chunkerpref",
    "kernel",
    "merge",
    "lege",
    "pointinfo",
    "tochunkgraph",
]
