"""Shared setup for accelerated-kernel examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import Chunker, circle

TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def make_circle() -> Chunker:
    return circle(quadrature_order=12, panel_count=20)


def scalar_density(boundary: Chunker) -> NDArray[np.floating]:
    return np.cos(boundary.positions[0])[None, :, :]


def stokes_density(boundary: Chunker) -> NDArray[np.floating]:
    return np.stack((np.cos(boundary.positions[0]), np.sin(boundary.positions[1])), axis=0)


def component_vector(values: NDArray[np.generic]) -> NDArray[np.generic]:
    return np.asarray(values).reshape(-1)


def relerr(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
