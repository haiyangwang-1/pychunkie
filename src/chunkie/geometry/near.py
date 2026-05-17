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


def flagnear_rectangle(chunker: Chunker, points: ArrayLike, *, rho: float = 1.0) -> NDArray[np.bool_]:
    """Flag targets inside per-panel padded bounding boxes."""

    targets = np.asarray(points, dtype=float).reshape(chunker.coordinate_dim, -1)
    flags = np.zeros((targets.shape[1], chunker.panel_count), dtype=bool)
    panel_lengths = chunker.panel_lengths
    for panel_id in range(chunker.panel_count):
        panel_positions = chunker.positions[:, :, panel_id]
        padding = float(rho) * panel_lengths[panel_id]
        lower = np.min(panel_positions, axis=1) - padding
        upper = np.max(panel_positions, axis=1) + padding
        # The rectangle test is deliberately axis-aligned. It is a conservative
        # candidate filter before more expensive distance or special-quadrature
        # logic decides whether a panel truly needs local treatment.
        flags[:, panel_id] = np.all((targets.T >= lower[None, :]) & (targets.T <= upper[None, :]), axis=1)
    return flags


def flagnear_rectangle_grid(
    chunker: Chunker,
    x: ArrayLike,
    y: ArrayLike,
    *,
    rho: float = 1.0,
) -> NDArray[np.bool_]:
    """Evaluate :func:`flagnear_rectangle` on a meshgrid in row-major grid order."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    xx, yy = np.meshgrid(x_array, y_array)
    points = np.vstack((xx.ravel(), yy.ravel()))
    return flagnear_rectangle(chunker, points, rho=rho).reshape(y_array.size, x_array.size, chunker.panel_count)
