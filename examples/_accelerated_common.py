"""Shared setup for accelerated-kernel examples."""

from __future__ import annotations

import numpy as np

from chunkie import PointInfo, chunkerfunc

TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def circle(t: np.ndarray):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def make_circle():
    return chunkerfunc(circle, min_chunks=6, tol=1e-8, order=8)[0]


def boundary_nodes(boundary) -> np.ndarray:
    return PointInfo.from_any(boundary).r


def component_vector(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).T.reshape(-1)


def relerr(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
