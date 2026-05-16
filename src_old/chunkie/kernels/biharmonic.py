"""Two-dimensional biharmonic Green kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.geometry import PointInfo


def green(
    source: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``r^2 log(r) / (8*pi)`` and its target derivatives."""

    source_points = np.asarray(source, dtype=float).reshape(2, -1)
    target_points = np.asarray(target, dtype=float).reshape(2, -1)
    x = target_points[0, :, None] - source_points[0, None, :]
    y = target_points[1, :, None] - source_points[1, None, :]
    r2 = x**2 + y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        logr2 = np.log(r2)
        val = r2 * logr2 / (16.0 * np.pi)
        grad = np.empty((target_points.shape[1], source_points.shape[1], 2))
        grad[:, :, 0] = x * (logr2 + 1.0) / (8.0 * np.pi)
        grad[:, :, 1] = y * (logr2 + 1.0) / (8.0 * np.pi)
        hess = np.empty((target_points.shape[1], source_points.shape[1], 3))
        hess[:, :, 0] = (logr2 + 1.0 + 2.0 * x**2 / r2) / (8.0 * np.pi)
        hess[:, :, 1] = x * y / (4.0 * np.pi * r2)
        hess[:, :, 2] = (logr2 + 1.0 + 2.0 * y**2 / r2) / (8.0 * np.pi)
        lap = (logr2 + 2.0) / (4.0 * np.pi)
    coincident = r2 == 0.0
    val[coincident] = 0.0
    grad[coincident, :] = 0.0
    return val, grad, hess, lap


def kernel(
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str,
) -> np.ndarray:
    """Evaluate biharmonic single, double, derivative, gradient, or Hessian kernels."""

    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
    val, grad, hess, lap = green(source_info.r, target_info.r)

    if typ in {"s", "single"}:
        return val
    if typ in {"lap", "slap", "laplacian"}:
        return lap
    if typ in {"d", "double"}:
        _require(source_info.n, "source normals")
        return -(
            grad[:, :, 0] * source_info.n[0, None, :] + grad[:, :, 1] * source_info.n[1, None, :]
        )
    if typ in {"sp", "sprime"}:
        _require(target_info.n, "target normals")
        return grad[:, :, 0] * target_info.n[0, :, None] + grad[:, :, 1] * target_info.n[1, :, None]
    if typ in {"sgrad", "sg"}:
        return grad.transpose(0, 2, 1).reshape(2 * target_info.r.shape[1], source_info.r.shape[1])
    if typ in {"shess", "hess"}:
        return hess.transpose(0, 2, 1).reshape(3 * target_info.r.shape[1], source_info.r.shape[1])
    raise ValueError(f"Unknown biharmonic kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
