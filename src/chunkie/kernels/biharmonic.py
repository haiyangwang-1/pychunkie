"""Two-dimensional biharmonic Green kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.geometry import PointInfo


def green(src: ArrayLike, targ: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``r^2 log(r) / (8*pi)`` and its target derivatives."""

    src_arr = np.asarray(src, dtype=float).reshape(2, -1)
    targ_arr = np.asarray(targ, dtype=float).reshape(2, -1)
    x = targ_arr[0, :, None] - src_arr[0, None, :]
    y = targ_arr[1, :, None] - src_arr[1, None, :]
    r2 = x**2 + y**2
    with np.errstate(divide="ignore", invalid="ignore"):
        logr2 = np.log(r2)
        val = r2 * logr2 / (16.0 * np.pi)
        grad = np.empty((targ_arr.shape[1], src_arr.shape[1], 2))
        grad[:, :, 0] = x * (logr2 + 1.0) / (8.0 * np.pi)
        grad[:, :, 1] = y * (logr2 + 1.0) / (8.0 * np.pi)
        hess = np.empty((targ_arr.shape[1], src_arr.shape[1], 3))
        hess[:, :, 0] = (logr2 + 1.0 + 2.0 * x**2 / r2) / (8.0 * np.pi)
        hess[:, :, 1] = x * y / (4.0 * np.pi * r2)
        hess[:, :, 2] = (logr2 + 1.0 + 2.0 * y**2 / r2) / (8.0 * np.pi)
        lap = (logr2 + 2.0) / (4.0 * np.pi)
    val = np.nan_to_num(val, nan=0.0, neginf=0.0, posinf=0.0)
    grad = np.nan_to_num(grad, nan=0.0, neginf=0.0, posinf=0.0)
    hess = np.nan_to_num(hess, nan=0.0, neginf=0.0, posinf=0.0)
    lap = np.nan_to_num(lap, nan=0.0, neginf=0.0, posinf=0.0)
    return val, grad, hess, lap


def kern(
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
    kind: str,
) -> np.ndarray:
    """Evaluate biharmonic single, double, derivative, gradient, or Hessian kernels."""

    src = PointInfo.from_any(srcinfo)
    targ = PointInfo.from_any(targinfo)
    typ = kind.lower()
    val, grad, hess, lap = green(src.r, targ.r)

    if typ in {"s", "single"}:
        return val
    if typ in {"lap", "slap", "laplacian"}:
        return lap
    if typ in {"d", "double"}:
        _require(src.n, "source normals")
        return -(grad[:, :, 0] * src.n[0, None, :] + grad[:, :, 1] * src.n[1, None, :])
    if typ in {"sp", "sprime"}:
        _require(targ.n, "target normals")
        return grad[:, :, 0] * targ.n[0, :, None] + grad[:, :, 1] * targ.n[1, :, None]
    if typ in {"sgrad", "sg"}:
        return grad.transpose(0, 2, 1).reshape(2 * targ.r.shape[1], src.r.shape[1])
    if typ in {"shess", "hess"}:
        return hess.transpose(0, 2, 1).reshape(3 * targ.r.shape[1], src.r.shape[1])
    raise ValueError(f"Unknown biharmonic kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
