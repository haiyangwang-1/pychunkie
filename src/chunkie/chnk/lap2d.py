"""Two-dimensional Laplace kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.operators import PointInfo, pointinfo


def green(
    src: ArrayLike,
    targ: ArrayLike,
    nolog: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the 2D Laplace Green's function, gradient, and Hessian."""

    src_arr = np.asarray(src, dtype=float).reshape(2, -1)
    targ_arr = np.asarray(targ, dtype=float).reshape(2, -1)
    rx = targ_arr[0, :, None] - src_arr[0, None, :]
    ry = targ_arr[1, :, None] - src_arr[1, None, :]
    r2 = rx**2 + ry**2

    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.empty((targ_arr.shape[1], src_arr.shape[1])) if nolog else -np.log(r2) / (4.0 * np.pi)
        grad = np.empty((targ_arr.shape[1], src_arr.shape[1], 2))
        grad[:, :, 0] = -rx / (2.0 * np.pi * r2)
        grad[:, :, 1] = -ry / (2.0 * np.pi * r2)
        r4 = r2**2
        hess = np.empty((targ_arr.shape[1], src_arr.shape[1], 3))
        hess[:, :, 0] = rx**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2)
        hess[:, :, 1] = rx * ry / (np.pi * r4)
        hess[:, :, 2] = ry**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2)
    if nolog:
        val = np.empty((0, 0))
    return val, grad, hess


def kern(srcinfo: PointInfo | dict | ArrayLike, targinfo: PointInfo | dict | ArrayLike, kind: str) -> np.ndarray:
    """Evaluate standard Laplace layer kernels."""

    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    val, grad, hess = green(src.r, targ.r, nolog=typ not in {"s", "single", "c", "combined"})

    if typ in {"s", "single"}:
        return val
    if typ in {"d", "double"}:
        _require(src.n, "source normals")
        return -(grad[:, :, 0] * src.n[0, None, :] + grad[:, :, 1] * src.n[1, None, :])
    if typ in {"sp", "sprime"}:
        _require(targ.n, "target normals")
        return grad[:, :, 0] * targ.n[0, :, None] + grad[:, :, 1] * targ.n[1, :, None]
    if typ == "stau":
        _require(targ.n, "target normals")
        return -grad[:, :, 0] * targ.n[1, :, None] + grad[:, :, 1] * targ.n[0, :, None]
    if typ in {"hilb"}:
        _require(src.n, "source normals")
        return 2.0 * (grad[:, :, 0] * src.n[1, None, :] - grad[:, :, 1] * src.n[0, None, :])
    if typ in {"sgrad", "sg"}:
        return np.moveaxis(grad, 2, 0).reshape(2 * targ.r.shape[1], src.r.shape[1])
    if typ in {"dgrad", "dg"}:
        _require(src.n, "source normals")
        sub = -(hess[:, :, 0:2] * src.n[0, None, :, None] + hess[:, :, 1:3] * src.n[1, None, :, None])
        return np.moveaxis(sub, 2, 0).reshape(2 * targ.r.shape[1], src.r.shape[1])
    raise ValueError(f"Unknown Laplace kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
