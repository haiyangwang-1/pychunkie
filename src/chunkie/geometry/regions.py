"""Region classification helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def winding_number(points: ArrayLike, polygon: ArrayLike) -> NDArray[np.floating]:
    """Compute polygon winding numbers for target points."""

    pts = np.asarray(points, dtype=float).reshape(2, -1)
    poly = np.asarray(polygon, dtype=float)
    if poly.shape[0] != 2 and poly.shape[1] == 2:
        poly = poly.T
    out = np.zeros(pts.shape[1], dtype=float)
    for idx in range(poly.shape[1]):
        a = poly[:, idx]
        b = poly[:, (idx + 1) % poly.shape[1]]
        va = a[:, None] - pts
        vb = b[:, None] - pts
        out += np.arctan2(va[0] * vb[1] - va[1] * vb[0], np.sum(va * vb, axis=0))
    return out / (2.0 * np.pi)
