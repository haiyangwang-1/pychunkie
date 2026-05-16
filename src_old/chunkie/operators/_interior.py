"""Direct polygon-based interior tests for chunker boundaries."""

from __future__ import annotations

import numpy as np

from .. import lege
from ..geometry.chunker import Chunker


def _chunkerinterior_direct(
    chunker: Chunker, points: np.ndarray, axissym: bool = False
) -> np.ndarray:
    inside = np.zeros(points.shape[1], dtype=bool)
    for boundary in _chunker_component_polygons(chunker, axissym):
        inside ^= _points_in_polygon(points, boundary)
    return inside


def _chunker_polygon_points(chunker: Chunker) -> np.ndarray:
    return _chunker_component_polygons(chunker)[0]


def _chunker_component_polygons(chunker: Chunker, axissym: bool = False) -> list[np.ndarray]:
    sorted_chunker, info = chunker.sort()
    polygons: list[np.ndarray] = []
    start = 0
    for nch, closed in zip(
        np.asarray(info["nchs"], dtype=int),
        np.asarray(info["ifclosed"], dtype=bool),
        strict=True,
    ):
        polygons.append(
            _chunker_component_polygon(sorted_chunker, start, int(nch), bool(closed), axissym)
        )
        start += int(nch)
    if not polygons:
        raise ValueError("chunker has no boundary points")
    return polygons


def _chunker_component_polygon(
    chunker: Chunker, start: int, nch: int, closed: bool, axissym: bool
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    ts = np.linspace(-1.0, 1.0, max(4 * chunker.k, 64))
    interp = lege.matrin(chunker.k, ts)[0]
    for ich in range(start, start + nch):
        panel = (interp @ chunker.r[:, :, ich].T).T
        if pieces:
            panel = panel[:, 1:]
        pieces.append(panel.T)
    points = np.vstack(pieces)
    if axissym and not closed:
        axis_end = np.array([[0.0, points[-1, 1]], [0.0, points[0, 1]]])
        points = np.vstack((points, axis_end))
    if np.linalg.norm(points[0] - points[-1]) > 1e-12:
        points = np.vstack((points, points[0]))
    return points


def _points_in_polygon(pts: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    x = pts[0]
    y = pts[1]
    inside = np.zeros(pts.shape[1], dtype=bool)
    x0 = boundary[:, 0]
    y0 = boundary[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1, strict=True):
        crosses = (ya > y) != (yb > y)
        hits = np.zeros_like(crosses)
        if np.any(crosses):
            xhit = (xb - xa) * (y[crosses] - ya) / (yb - ya) + xa
            hits[crosses] = x[crosses] < xhit
        inside ^= hits
    return inside
