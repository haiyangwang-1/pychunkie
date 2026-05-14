"""Two-dimensional Helmholtz kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import hankel1

from chunkie.geometry import PointInfo

from . import laplace as lap2d

_EULER_GAMMA = 0.57721566490153286060651209008240243


def green(
    zk: complex, source: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Helmholtz Green's function, gradient, and Hessian."""

    source_points = np.asarray(source, dtype=float).reshape(2, -1)
    target_points = np.asarray(target, dtype=float).reshape(2, -1)
    rx = target_points[0, :, None] - source_points[0, None, :]
    ry = target_points[1, :, None] - source_points[1, None, :]
    r2 = rx**2 + ry**2
    r = np.sqrt(r2)

    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = hankel1(0, zk * r)
        h1 = hankel1(1, zk * r)
        val = 0.25j * h0
        grad = np.empty((target_points.shape[1], source_points.shape[1], 2), dtype=complex)
        grad[:, :, 0] = -0.25j * zk * h1 * rx / r
        grad[:, :, 1] = -0.25j * zk * h1 * ry / r
        h2 = 2.0 * h1 / (zk * r) - h0
        hess = np.empty((target_points.shape[1], source_points.shape[1], 3), dtype=complex)
        hess[:, :, 0] = 0.25j * zk * (((rx - ry) * (rx + ry) * h1 / r**3) - zk * rx**2 * h0 / r2)
        hess[:, :, 1] = 0.25j * zk * zk * rx * ry * h2 / r2
        hess[:, :, 2] = 0.25j * zk * (((ry - rx) * (rx + ry) * h1 / r**3) - zk * ry**2 * h0 / r2)
    return val, grad, hess


def helmdiffgreen(
    zk: complex, source: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Helmholtz Green function with the Laplace log singularity removed."""

    source_points = np.asarray(source, dtype=float).reshape(2, -1)
    target_points = np.asarray(target, dtype=float).reshape(2, -1)
    val, grad, hess = green(zk, source_points, target_points)
    lap_val, lap_grad, lap_hess = lap2d.green(source_points, target_points)
    val = val - lap_val
    grad = grad - lap_grad
    hess = hess - lap_hess

    r2 = (target_points[0, :, None] - source_points[0, None, :]) ** 2 + (
        target_points[1, :, None] - source_points[1, None, :]
    ) ** 2
    coincident = r2 == 0.0
    if np.any(coincident):
        val[coincident] = 0.25j - (np.log(zk / 2.0) + _EULER_GAMMA) / (2.0 * np.pi)
        grad[coincident, :] = 0.0
    return val, grad, hess


def kernel(
    zk: complex,
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str,
    coefs: ArrayLike | None = None,
) -> np.ndarray:
    """Evaluate standard Helmholtz layer and transmission kernels.

    The scalar selector surface mirrors Laplace where applicable:
    ``"s"``, ``"d"``, ``"sp"``, ``"stau"``, ``"sgrad"``, ``"dgrad"``,
    ``"dp"``, and combined forms. Transmission-style selectors such as
    ``"c2trans"``, ``"all"``, and ``"trans_rep"`` return interleaved block
    rows/columns for coupled representation systems.
    """

    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
    is_diff = typ.endswith("_diff")
    suffix = "_diff" if is_diff else ""
    if is_diff:
        typ = typ[: -len("_diff")]
        val, grad, hess = helmdiffgreen(zk, source_info.r, target_info.r)
    else:
        val, grad, hess = green(zk, source_info.r, target_info.r)

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
    if typ in {"stau", "st"}:
        _require(target_info.d, "target tangents")
        speed = np.sqrt(target_info.d[0] ** 2 + target_info.d[1] ** 2)
        return (
            grad[:, :, 0] * target_info.d[0, :, None] + grad[:, :, 1] * target_info.d[1, :, None]
        ) / speed[:, None]
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
        c = _coef_vector(coefs)
        return c[0] * kernel(zk, source_info, target_info, f"d{suffix}") + c[1] * kernel(
            zk, source_info, target_info, f"s{suffix}"
        )
    if typ in {"cp", "cprime"}:
        c = _coef_vector(coefs)
        return c[0] * kernel(zk, source_info, target_info, f"dp{suffix}") + c[1] * kernel(
            zk, source_info, target_info, f"sp{suffix}"
        )
    if typ in {"cg", "cgrad"}:
        c = _coef_vector(coefs)
        return c[0] * kernel(zk, source_info, target_info, f"dgrad{suffix}") + c[1] * kernel(
            zk, source_info, target_info, f"sgrad{suffix}"
        )
    if typ in {"c2tr", "c2trans"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        if c.size == 2:
            c = np.tile(c.reshape(-1)[:2].reshape(1, 2), (2, 1))
        out = np.zeros((2 * target_info.r.shape[1], source_info.r.shape[1]), dtype=complex)
        out[0::2] = c[0, 0] * kernel(zk, source_info, target_info, f"d{suffix}") + c[0, 1] * val
        out[1::2] = c[1, 0] * kernel(zk, source_info, target_info, f"dp{suffix}") + c[
            1, 1
        ] * kernel(zk, source_info, target_info, f"sp{suffix}")
        return out
    if typ in {"all", "trans_sys", "ts"}:
        cc = np.ones((2, 2), dtype=complex) if coefs is None else np.asarray(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        out[0::2, 0::2] = cc[0, 0] * kernel(zk, source_info, target_info, f"d{suffix}")
        out[0::2, 1::2] = cc[0, 1] * val
        out[1::2, 0::2] = cc[1, 0] * kernel(zk, source_info, target_info, f"dp{suffix}")
        out[1::2, 1::2] = cc[1, 1] * kernel(zk, source_info, target_info, f"sp{suffix}")
        return out
    if typ in {"trans_rep", "trep"}:
        c = _coef_vector(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kernel(zk, source_info, target_info, f"d{suffix}")
        out[:, 1::2] = c[1] * val
        return out
    if typ in {"trans_rep_prime", "trep_p", "trans_rep_p"}:
        c = _coef_vector(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kernel(zk, source_info, target_info, f"dp{suffix}")
        out[:, 1::2] = c[1] * kernel(zk, source_info, target_info, f"sp{suffix}")
        return out
    if typ in {"trans_rep_grad", "trep_g", "trans_rep_g"}:
        c = _coef_vector(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        dgrad = kernel(zk, source_info, target_info, f"dgrad{suffix}")
        sgrad = kernel(zk, source_info, target_info, f"sgrad{suffix}")
        out[0::2, 0::2] = c[0] * dgrad[0::2]
        out[0::2, 1::2] = c[1] * sgrad[0::2]
        out[1::2, 0::2] = c[0] * dgrad[1::2]
        out[1::2, 1::2] = c[1] * sgrad[1::2]
        return out
    raise ValueError(f"Unknown Helmholtz kernel type {kind!r}.")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")


def _coef_vector(coefs: ArrayLike | None) -> np.ndarray:
    if coefs is None:
        return np.array([1.0, 1.0j])
    return np.asarray(coefs).reshape(-1)
