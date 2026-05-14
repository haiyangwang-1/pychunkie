"""Shared utilities for the smooth Laplace examples."""

from __future__ import annotations

import numpy as np

from chunkie import PointInfo, chunkerfunc, chunkerkerneval, kernel

INTERIOR_TARGETS = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
EXTERIOR_TARGETS = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])


def circle(t: np.ndarray):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def make_circle():
    return chunkerfunc(circle, min_chunks=10, tol=1e-10, order=16)[0]


def boundary_nodes(boundary) -> np.ndarray:
    return PointInfo.from_any(boundary).r


def boundary_normals(boundary) -> np.ndarray:
    return PointInfo.from_any(boundary).n


def boundary_weights(boundary) -> np.ndarray:
    return boundary.quadrature_weights.T.reshape(-1)


def interior_solution(targets: np.ndarray) -> np.ndarray:
    return np.asarray(targets)[0]


def exterior_solution(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    r2 = np.sum(targets**2, axis=0)
    return targets[0] / r2


def single_layer_values(
    boundary, sigma: np.ndarray, const: float, targets: np.ndarray
) -> np.ndarray:
    vals = chunkerkerneval(boundary, kernel("lap", "s"), sigma, targets, force_adaptive=True)
    return vals.reshape(-1) + const
