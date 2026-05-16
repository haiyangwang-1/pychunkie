"""Small operator data containers and ndarray wrappers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..geometry.chunker import Chunker


@dataclass
class _BlockKernelLayout:
    chunkers: list[Chunker]
    kernels: np.ndarray
    opdims_mat: np.ndarray
    rowdims: np.ndarray
    coldims: np.ndarray
    row_offsets: np.ndarray
    col_offsets: np.ndarray


@dataclass
class RCIPContext:
    """Corner compression metadata produced by chunkgraph RCIP assembly."""

    source: Any
    system_kernel: Callable[[Any, Any], np.ndarray]
    saved: list[Any]
    nsub: int
    savedepth: int
    ndim: int = 1


class ChunkerRCIPMatrix(np.ndarray):
    """Dense matrix with attached RCIP interpolation metadata."""

    rcip: RCIPContext | None

    def __new__(cls, input_array: ArrayLike, rcip_context: RCIPContext | None = None):
        obj = np.asarray(input_array).view(cls)
        obj.rcip = rcip_context
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        self.rcip = None if obj is None else getattr(obj, "rcip", None)
