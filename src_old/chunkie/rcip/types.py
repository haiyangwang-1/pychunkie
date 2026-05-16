"""Small RCIP data containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.chunker import Chunker


@dataclass
class RCIPSaved:
    k: int
    ndim: int
    nedge: int
    Pbc: np.ndarray
    PWbc: np.ndarray
    starL: np.ndarray
    circL: np.ndarray
    starS: np.ndarray
    circS: np.ndarray
    ilist: np.ndarray
    starL1: np.ndarray
    circL1: np.ndarray
    nsub: int = 0
    savedepth: float = np.inf
    R: list[np.ndarray] | None = None
    MAT: list[np.ndarray] | None = None
    chnkrlocals: list[Chunker] | None = None
    starind: np.ndarray | None = None
    ctr: np.ndarray | None = None
    rcs: np.ndarray | None = None
    dcs: np.ndarray | None = None
    d2cs: np.ndarray | None = None
    dscal: np.ndarray | None = None
    d2scal: np.ndarray | None = None
    ileftright: np.ndarray | None = None
    glxs: np.ndarray | None = None
    glws: np.ndarray | None = None


@dataclass
class RCIPChunkGraphResult:
    vertices: np.ndarray
    edge_indices: list[np.ndarray]
    R: list[np.ndarray]
    saved: list[RCIPSaved]
    kernels: list[Any]
