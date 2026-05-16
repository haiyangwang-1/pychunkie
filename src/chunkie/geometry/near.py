"""Near-panel geometric queries."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chunker import Chunker


def flagnear(chunker: Chunker, points: ArrayLike, *, near_factor: float = 1.0) -> NDArray[np.bool_]:
    """Flag panels whose nodes are close to target points.

    This first implementation is intentionally direct: it compares target
    points with panel nodes and uses the panel arclength as the length scale.
    """

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    flags = np.zeros((targets.shape[1], chunker.panel_count), dtype=bool)
    panel_lengths = np.sum(chunker.weights, axis=0)
    for panel_id in range(chunker.panel_count):
        source = chunker.positions[:, :, panel_id]
        distances = np.linalg.norm(targets[:, :, None] - source[:, None, :], axis=0)
        flags[:, panel_id] = np.min(distances, axis=1) <= near_factor * panel_lengths[panel_id]
    return flags
