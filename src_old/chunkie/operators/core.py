"""Public operator facade."""

from __future__ import annotations

from ..geometry.pointinfo import PointInfo
from ._assembly import chunkerintegral, chunkermat, chunkermatapply
from ._evaluation import chunkerinterior, chunkerkerneval, chunkerkernevalmat
from ._flam import chunkerflam
from ._matrices import ChunkerFLAMMatrix, ChunkerFMMMatrix
from .types import ChunkerRCIPMatrix, RCIPContext

__all__ = [
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
