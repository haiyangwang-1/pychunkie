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
    "Kernel",
    "PointInfo",
    "chunker",
    "chunkerfit",
    "chunkerfunc",
    "chunkerfuncuni",
    "chunkerpoints",
    "chunkerpoly",
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
]
