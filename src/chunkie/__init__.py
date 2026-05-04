"""Python port of the MATLAB chunkIE package."""

from . import lege
from .chunker import (
    Chunker,
    ChunkerPref,
    chunker,
    chunkerfunc,
    chunkerpoints,
    chunkerpoly,
    chunkerpref,
    merge,
)
from .kernel import Kernel, kernel
from .operators import (
    PointInfo,
    chunkerintegral,
    chunkerkerneval,
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
    "chunkerfunc",
    "chunkerpoints",
    "chunkerpoly",
    "chunkerintegral",
    "chunkerkerneval",
    "chunkermat",
    "chunkermatapply",
    "chunkerpref",
    "kernel",
    "merge",
    "lege",
    "pointinfo",
]
