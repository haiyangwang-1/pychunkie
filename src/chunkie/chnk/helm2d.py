"""Two-dimensional Helmholtz kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import hankel1

from . import lap2d
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


def helmdiffgreen(zk: complex, src: ArrayLike, targ: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Helmholtz Green function with the Laplace log singularity removed."""

    val, grad, hess = green(zk, src, targ)
    lap_val, lap_grad, lap_hess = lap2d.green(src, targ)
    return val - lap_val, grad - lap_grad, hess - lap_hess


def kern(
    zk: complex,
    srcinfo: PointInfo | dict | ArrayLike,
    targinfo: PointInfo | dict | ArrayLike,
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

    src = pointinfo(srcinfo)
    targ = pointinfo(targinfo)
    typ = kind.lower()
    is_diff = typ.endswith("_diff")
    suffix = "_diff" if is_diff else ""
    if is_diff:
        typ = typ[: -len("_diff")]
        val, grad, hess = helmdiffgreen(zk, src.r, targ.r)
    else:
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
        return grad.transpose(0, 2, 1).reshape(2 * targ.r.shape[1], src.r.shape[1])
    if typ in {"dgrad", "dg"}:
        _require(src.n, "source normals")
        sub = -(hess[:, :, 0:2] * src.n[0, None, :, None] + hess[:, :, 1:3] * src.n[1, None, :, None])
        return sub.transpose(0, 2, 1).reshape(2 * targ.r.shape[1], src.r.shape[1])
    if typ in {"dp", "dprime"}:
        _require(src.n, "source normals")
        _require(targ.n, "target normals")
        return -(
            hess[:, :, 0] * src.n[0, None, :] * targ.n[0, :, None]
            + hess[:, :, 1]
            * (src.n[1, None, :] * targ.n[0, :, None] + src.n[0, None, :] * targ.n[1, :, None])
            + hess[:, :, 2] * src.n[1, None, :] * targ.n[1, :, None]
        )
    if typ in {"c", "combined"}:
        c = _coef_vector(coefs)
        return c[0] * kern(zk, src, targ, f"d{suffix}") + c[1] * kern(zk, src, targ, f"s{suffix}")
    if typ in {"cp", "cprime"}:
        c = _coef_vector(coefs)
        return c[0] * kern(zk, src, targ, f"dp{suffix}") + c[1] * kern(zk, src, targ, f"sp{suffix}")
    if typ in {"cg", "cgrad"}:
        c = _coef_vector(coefs)
        return c[0] * kern(zk, src, targ, f"dgrad{suffix}") + c[1] * kern(zk, src, targ, f"sgrad{suffix}")
    if typ in {"c2tr", "c2trans"}:
        c = np.array([1.0, 1.0j]) if coefs is None else np.asarray(coefs)
        if c.size == 2 or is_diff:
            c = np.tile(c.reshape(-1, order="F")[:2].reshape(1, 2), (2, 1))
        out = np.zeros((2 * targ.r.shape[1], src.r.shape[1]), dtype=complex)
        out[0::2] = c[0, 0] * kern(zk, src, targ, f"d{suffix}") + c[0, 1] * val
        out[1::2] = c[1, 0] * kern(zk, src, targ, f"dp{suffix}") + c[1, 1] * kern(zk, src, targ, f"sp{suffix}")
        return out
    if typ in {"all", "trans_sys", "ts"}:
        cc = np.ones((2, 2), dtype=complex) if coefs is None else np.asarray(coefs)
        nt = targ.r.shape[1]
        ns = src.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        out[0::2, 0::2] = cc[0, 0] * kern(zk, src, targ, f"d{suffix}")
        out[0::2, 1::2] = cc[0, 1] * val
        out[1::2, 0::2] = cc[1, 0] * kern(zk, src, targ, f"dp{suffix}")
        out[1::2, 1::2] = cc[1, 1] * kern(zk, src, targ, f"sp{suffix}")
        return out
    if typ in {"trans_rep", "trep"}:
        c = _coef_vector(coefs)
        nt = targ.r.shape[1]
        ns = src.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kern(zk, src, targ, f"d{suffix}")
        out[:, 1::2] = c[1] * val
        return out
    if typ in {"trans_rep_prime", "trep_p", "trans_rep_p"}:
        c = _coef_vector(coefs)
        nt = targ.r.shape[1]
        ns = src.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kern(zk, src, targ, f"dp{suffix}")
        out[:, 1::2] = c[1] * kern(zk, src, targ, f"sp{suffix}")
        return out
    if typ in {"trans_rep_grad", "trep_g", "trans_rep_g"}:
        c = _coef_vector(coefs)
        nt = targ.r.shape[1]
        ns = src.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        dgrad = kern(zk, src, targ, f"dgrad{suffix}")
        sgrad = kern(zk, src, targ, f"sgrad{suffix}")
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
    return np.asarray(coefs).reshape(-1, order="F")
