"""Bernstein ellipse helpers for close-panel detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bernstein_radius(points: ArrayLike) -> float:
    """Return a conservative radius for a point cloud.

    A richer Bernstein-ellipse implementation will replace this helper when
    near-panel quadrature starts consuming analytic-continuation geometry.
    """

    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return 0.0
    center = np.mean(arr, axis=-1, keepdims=True)
    return float(np.max(np.linalg.norm(arr - center, axis=0)))
