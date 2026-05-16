"""Operator assembly, matrix-free application, and evaluation."""

from .core import (
    ChunkerFLAMMatrix,
    ChunkerFMMMatrix,
    ChunkerRCIPMatrix,
    PointInfo,
    RCIPContext,
    chunkerflam,
    chunkerintegral,
    chunkerinterior,
    chunkerkerneval,
    chunkerkernevalmat,
    chunkermat,
    chunkermatapply,
)
from .options import _NORMALIZED_OPTIONS_MARKER, _normalize_public_options

__all__ = [
    "_NORMALIZED_OPTIONS_MARKER",
    "_normalize_public_options",
    "ChunkerFLAMMatrix",
    "ChunkerFMMMatrix",
    "ChunkerRCIPMatrix",
    "PointInfo",
    "RCIPContext",
    "chunkerflam",
    "chunkerintegral",
    "chunkerinterior",
    "chunkerkerneval",
    "chunkerkernevalmat",
    "chunkermat",
    "chunkermatapply",
]
