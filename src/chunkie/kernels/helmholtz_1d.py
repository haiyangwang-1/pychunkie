"""One-dimensional Helmholtz kernels used by flat-interface tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie.geometry import PointInfo


def green(
    zk: complex, source: ArrayLike, target: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``exp(1j*zk*|x-y|)`` and its 2D gradient/Hessian."""

    source_points = np.asarray(source, dtype=float).reshape(2, -1)
    target_points = np.asarray(target, dtype=float).reshape(2, -1)
    rx = target_points[0, :, None] - source_points[0, None, :]
    ry = target_points[1, :, None] - source_points[1, None, :]
    r2 = rx**2 + ry**2
    r = np.sqrt(r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.exp(1j * zk * r)
        grad = np.zeros((target_points.shape[1], source_points.shape[1], 2), dtype=complex)
        grad[:, :, 0] = 1j * zk * (rx / r) * val
        grad[:, :, 1] = 1j * zk * (ry / r) * val
        hess = np.zeros((target_points.shape[1], source_points.shape[1], 3), dtype=complex)
        r3 = r**3
        hess[:, :, 0] = ((1j * zk * r * rx**2 - rx**2 + r2) / r3) * val
        hess[:, :, 1] = ((1j * zk * r * rx * ry - rx * ry) / r3) * val
        hess[:, :, 2] = ((1j * zk * r * ry**2 - ry**2 + r2) / r3) * val
        hess *= 1j * zk
    return val, grad, hess


def kernel(
    zk: complex,
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str,
    coefs: ArrayLike | None = None,
) -> np.ndarray:
    """Evaluate 1D Helmholtz layer kernels using MATLAB selector names."""

    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
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
        speed = np.sqrt(np.sum(target_info.d**2, axis=0))
        return (
            grad[:, :, 0] * target_info.d[0, :, None] + grad[:, :, 1] * target_info.d[1, :, None]
        ) / speed[:, None]
    if typ in {"dp", "dprime"}:
        _require(source_info.n, "source normals")
        _require(target_info.n, "target normals")
        return -_normal_hessian(hess, source_info.n, target_info.n)
    if typ in {"c", "combined"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return c[0] * kernel(zk, source_info, target_info, "d") + c[1] * val
    if typ in {"cp", "cprime"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        return c[0] * kernel(zk, source_info, target_info, "dp") + c[1] * kernel(
            zk, source_info, target_info, "sp"
        )
    if typ in {"c2trans"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        out = np.zeros((2 * target_info.r.shape[1], source_info.r.shape[1]), dtype=complex)
        out[0::2] = c[0] * kernel(zk, source_info, target_info, "d") + c[1] * val
        out[1::2] = c[0] * kernel(zk, source_info, target_info, "dp") + c[1] * kernel(
            zk, source_info, target_info, "sp"
        )
        return out
    if typ in {"all", "trans_sys", "ts"}:
        cc = np.ones((2, 2)) if coefs is None else np.asarray(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        out[0::2, 0::2] = cc[0, 0] * kernel(zk, source_info, target_info, "d")
        out[0::2, 1::2] = cc[0, 1] * val
        out[1::2, 0::2] = cc[1, 0] * kernel(zk, source_info, target_info, "dp")
        out[1::2, 1::2] = cc[1, 1] * kernel(zk, source_info, target_info, "sp")
        return out
    if typ in {"trans_rep", "trep"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kernel(zk, source_info, target_info, "d")
        out[:, 1::2] = c[1] * val
        return out
    if typ in {"trans_rep_prime", "trep_p", "trans_rep_p"}:
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((nt, 2 * ns), dtype=complex)
        out[:, 0::2] = c[0] * kernel(zk, source_info, target_info, "dp")
        out[:, 1::2] = c[1] * kernel(zk, source_info, target_info, "sp")
        return out
    if typ in {"trans_rep_grad", "trep_g", "trans_rep_g"}:
        _require(source_info.n, "source normals")
        c = np.ones(2) if coefs is None else np.asarray(coefs)
        nt = target_info.r.shape[1]
        ns = source_info.r.shape[1]
        out = np.zeros((2 * nt, 2 * ns), dtype=complex)
        out[0::2, 0::2] = -c[0] * (
            hess[:, :, 0] * source_info.n[0, None, :] + hess[:, :, 1] * source_info.n[1, None, :]
        )
        out[0::2, 1::2] = c[1] * grad[:, :, 0]
        out[1::2, 0::2] = -c[0] * (
            hess[:, :, 1] * source_info.n[0, None, :] + hess[:, :, 2] * source_info.n[1, None, :]
        )
        out[1::2, 1::2] = c[1] * grad[:, :, 1]
        return out
    raise ValueError(f"Unknown 1D Helmholtz kernel type {kind!r}.")


def sweep(
    uin: ArrayLike, inds: ArrayLike, ts: ArrayLike, wts: ArrayLike, zk: complex
) -> np.ndarray:
    """Sweeping convolution helper from MATLAB ``chnk.helm1d.sweep``."""

    t = np.asarray(ts).reshape(-1)
    weights = np.asarray(wts).reshape(-1)
    indices = np.asarray(inds, dtype=int).reshape(-1)
    u = np.zeros(t.size, dtype=np.result_type(uin, complex))
    u[indices] = np.asarray(uin).reshape(-1)
    charges = u * weights
    exps = np.exp(1j * np.diff(t) * zk)

    vplus = np.zeros_like(u, dtype=complex)
    vplus[0] = charges[0]
    for idx in range(1, t.size):
        vplus[idx] = exps[idx - 1] * vplus[idx - 1] + charges[idx]

    vminus = np.zeros_like(u, dtype=complex)
    rev_charges = charges[::-1]
    rev_exps = exps[::-1]
    for idx in range(1, t.size):
        vminus[idx] = rev_exps[idx - 1] * (vminus[idx - 1] + rev_charges[idx - 1])
    return vplus + vminus[::-1]


def _normal_hessian(
    hess: np.ndarray, source_normal: np.ndarray, target_normal: np.ndarray
) -> np.ndarray:
    return (
        hess[:, :, 0] * source_normal[0, None, :] * target_normal[0, :, None]
        + hess[:, :, 1]
        * (
            source_normal[1, None, :] * target_normal[0, :, None]
            + source_normal[0, None, :] * target_normal[1, :, None]
        )
        + hess[:, :, 2] * source_normal[1, None, :] * target_normal[1, :, None]
    )


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
