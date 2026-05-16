"""Two-dimensional Laplace kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.geometry import PointInfo


def green(
    source: ArrayLike,
    target: ArrayLike,
    nolog: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the 2D Laplace Green's function, gradient, and Hessian."""

    source_points = np.asarray(source, dtype=float).reshape(2, -1)
    target_points = np.asarray(target, dtype=float).reshape(2, -1)
    rx = target_points[0, :, None] - source_points[0, None, :]
    ry = target_points[1, :, None] - source_points[1, None, :]
    r2 = rx**2 + ry**2

    with np.errstate(divide="ignore", invalid="ignore"):
        val = (
            np.empty((target_points.shape[1], source_points.shape[1]))
            if nolog
            else -np.log(r2) / (4.0 * np.pi)
        )
        grad = np.empty((target_points.shape[1], source_points.shape[1], 2))
        grad[:, :, 0] = -rx / (2.0 * np.pi * r2)
        grad[:, :, 1] = -ry / (2.0 * np.pi * r2)
        r4 = r2**2
        hess = np.empty((target_points.shape[1], source_points.shape[1], 3))
        hess[:, :, 0] = rx**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2)
        hess[:, :, 1] = rx * ry / (np.pi * r4)
        hess[:, :, 2] = ry**2 / (np.pi * r4) - 1.0 / (2.0 * np.pi * r2)
    if nolog:
        val = np.empty((0, 0))
    return val, grad, hess


def kernel(
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str,
    coefs: ArrayLike | None = None,
) -> np.ndarray:
    """Evaluate standard Laplace layer kernels.

    Selectors include single layer ``"s"``, double layer ``"d"``,
    target-normal derivative ``"sp"``, tangential derivative ``"stau"``,
    source-gradient rows ``"sgrad"``, double-layer gradient ``"dgrad"``,
    hypersingular normal-normal derivative ``"dp"``, and combined forms
    ``"c"``, ``"cp"``, and ``"cgrad"``.
    """

    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
    val, grad, hess = green(
        source_info.r,
        target_info.r,
        nolog=typ not in {"s", "single", "c", "combined"},
    )

    if typ in {"s", "single"}:
        return val
    if typ in {"d", "double"}:
        _require(source_info.n, "source normals")
        return -(
            grad[:, :, 0] * source_info.n[0, None, :] + grad[:, :, 1] * source_info.n[1, None, :]
        )
    if typ in {"sp", "sprime"}:
        _require(target_info.n, "target normals")
        return grad[:, :, 0] * target_info.n[0, :, None] + grad[:, :, 1] * target_info.n[1, :, None]
    if typ == "stau":
        _require(target_info.n, "target normals")
        return (
            -grad[:, :, 0] * target_info.n[1, :, None] + grad[:, :, 1] * target_info.n[0, :, None]
        )
    if typ in {"hilb"}:
        _require(source_info.n, "source normals")
        return 2.0 * (
            grad[:, :, 0] * source_info.n[1, None, :] - grad[:, :, 1] * source_info.n[0, None, :]
        )
    if typ in {"sgrad", "sg"}:
        return grad.transpose(0, 2, 1).reshape(2 * target_info.r.shape[1], source_info.r.shape[1])
    if typ in {"dgrad", "dg"}:
        _require(source_info.n, "source normals")
        sub = -(
            hess[:, :, 0:2] * source_info.n[0, None, :, None]
            + hess[:, :, 1:3] * source_info.n[1, None, :, None]
        )
        return sub.transpose(0, 2, 1).reshape(2 * target_info.r.shape[1], source_info.r.shape[1])
    if typ in {"dp", "dprime"}:
        _require(source_info.n, "source normals")
        _require(target_info.n, "target normals")
        return -(
            hess[:, :, 0] * source_info.n[0, None, :] * target_info.n[0, :, None]
            + hess[:, :, 1]
            * (
                source_info.n[1, None, :] * target_info.n[0, :, None]
                + source_info.n[0, None, :] * target_info.n[1, :, None]
            )
            + hess[:, :, 2] * source_info.n[1, None, :] * target_info.n[1, :, None]
        )
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return c[0] * kernel(source_info, target_info, "d") + c[1] * kernel(
            source_info, target_info, "s"
        )
    if typ in {"cp", "cprime"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return c[0] * kernel(source_info, target_info, "dp") + c[1] * kernel(
            source_info, target_info, "sp"
        )
    if typ in {"cg", "cgrad"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return c[0] * kernel(source_info, target_info, "dg") + c[1] * kernel(
            source_info, target_info, "sg"
        )
    raise ValueError(f"Unknown Laplace kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
