"""Two-dimensional Helmholtz kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import hankel1

from chunkie.operators import PointInfo, pointinfo


def green(zk: complex, src: ArrayLike, targ: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Helmholtz Green's function, gradient, and Hessian."""

    src_arr = np.asarray(src, dtype=float).reshape(2, -1)
    targ_arr = np.asarray(targ, dtype=float).reshape(2, -1)
    rx = targ_arr[0, :, None] - src_arr[0, None, :]
    ry = targ_arr[1, :, None] - src_arr[1, None, :]
    r2 = rx**2 + ry**2
    r = np.sqrt(r2)

    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = hankel1(0, zk * r)
        h1 = hankel1(1, zk * r)
        val = 0.25j * h0
        grad = np.empty((targ_arr.shape[1], src_arr.shape[1], 2), dtype=complex)
        grad[:, :, 0] = -0.25j * zk * h1 * rx / r
        grad[:, :, 1] = -0.25j * zk * h1 * ry / r
        h2 = 2.0 * h1 / (zk * r) - h0
        hess = np.empty((targ_arr.shape[1], src_arr.shape[1], 3), dtype=complex)
        hess[:, :, 0] = 0.25j * zk * (((rx - ry) * (rx + ry) * h1 / r**3) - zk * rx**2 * h0 / r2)
        hess[:, :, 1] = 0.25j * zk * zk * rx * ry * h2 / r2
        hess[:, :, 2] = 0.25j * zk * (((ry - rx) * (rx + ry) * h1 / r**3) - zk * ry**2 * h0 / r2)
    return val, grad, hess


def kern(
    zk: complex,
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
    kind: str,
) -> np.ndarray:
    """Evaluate standard Helmholtz layer kernels."""

    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    val, grad, hess = green(zk, src.r, targ.r)

    if typ in {"s", "single"}:
        return val
    if typ in {"d", "double"}:
        _require(src.n, "source normals")
        return -(grad[:, :, 0] * src.n[0, None, :] + grad[:, :, 1] * src.n[1, None, :])
    if typ in {"sp", "sprime"}:
        _require(targ.n, "target normals")
        return grad[:, :, 0] * targ.n[0, :, None] + grad[:, :, 1] * targ.n[1, :, None]
    if typ in {"stau", "st"}:
        _require(targ.d, "target tangents")
        speed = np.sqrt(targ.d[0] ** 2 + targ.d[1] ** 2)
        return (grad[:, :, 0] * targ.d[0, :, None] + grad[:, :, 1] * targ.d[1, :, None]) / speed[:, None]
    if typ in {"sgrad", "sg"}:
        return np.moveaxis(grad, 2, 0).reshape(2 * targ.r.shape[1], src.r.shape[1])
    if typ in {"dgrad", "dg"}:
        _require(src.n, "source normals")
        sub = -(hess[:, :, 0:2] * src.n[0, None, :, None] + hess[:, :, 1:3] * src.n[1, None, :, None])
        return np.moveaxis(sub, 2, 0).reshape(2 * targ.r.shape[1], src.r.shape[1])
    raise ValueError(f"Unknown Helmholtz kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
