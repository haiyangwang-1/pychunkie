"""Small geometry helpers from MATLAB ``+chnk``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.chunker import Chunker


def flagnear(chnkr: Chunker, pts: ArrayLike, opts: dict | None = None) -> np.ndarray:
    return chnkr.flagnear(pts, opts)


def flagself(srcs: ArrayLike, targs: ArrayLike, tol: float = 1e-14) -> np.ndarray:
    src = np.asarray(srcs, dtype=float).reshape(np.asarray(srcs).shape[0], -1)
    targ = np.asarray(targs, dtype=float).reshape(np.asarray(targs).shape[0], -1)
    pairs: list[tuple[int, int]] = []
    for i in range(src.shape[1]):
        dists = np.sqrt(np.sum((targ - src[:, i : i + 1]) ** 2, axis=0))
        close = np.flatnonzero(dists < tol)
        pairs.extend((i, int(j)) for j in close)
    if not pairs:
        return np.zeros((2, 0), dtype=int)
    return np.array(pairs, dtype=int).T
