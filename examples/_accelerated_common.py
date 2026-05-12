"""Shared setup for accelerated-kernel examples."""

from __future__ import annotations

import numpy as np

from chunkie import chunkerfunc


TARGETS = np.array([[0.1, 1.5, -0.7], [0.2, 0.3, 1.4]])


def circle(t: np.ndarray):
    t = np.asarray(t)
    return (
        np.vstack((np.cos(t), np.sin(t))),
        np.vstack((-np.sin(t), np.cos(t))),
        np.vstack((-np.cos(t), -np.sin(t))),
    )


def make_circle():
    return chunkerfunc(circle, {"nchmin": 6, "eps": 1e-8}, {"k": 8})[0]


def boundary_nodes(chnkr) -> np.ndarray:
    return chnkr.r.reshape(2, chnkr.npt, order="F")


def relerr(actual, expected) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0))
