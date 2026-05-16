"""Two-dimensional Stokes kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import boundary_matrix_from_kernel_tensor
from chunkie.geometry import PointInfo


def kernel(
    mu: float,
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str,
    coefs: ArrayLike | None = None,
) -> np.ndarray:
    """Evaluate 2D Stokes velocity, pressure, traction, or gradient kernels.

    Vector densities and vector values are interleaved by node. For example,
    a velocity single-layer matrix has shape ``(2*ntarget, 2*nsource)`` with
    ``x`` and ``y`` components alternating in both rows and columns.
    """

    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
    rx = target_info.r[0, :, None] - source_info.r[0, None, :]
    ry = target_info.r[1, :, None] - source_info.r[1, None, :]
    r2 = rx**2 + ry**2
    nt, ns = rx.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        if typ in {"svel", "svelocity", "s", "single"}:
            r = np.sqrt(r2)
            log1r = np.log(1.0 / r)
            kxx = (rx**2 / r2 + log1r) / (4.0 * np.pi * mu)
            kyy = (ry**2 / r2 + log1r) / (4.0 * np.pi * mu)
            kxy = (rx * ry / r2) / (4.0 * np.pi * mu)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ in {"spres", "spressure"}:
            kx = rx / (2.0 * np.pi * r2)
            ky = ry / (2.0 * np.pi * r2)
            return _interleave_1x2(kx, ky, nt, ns)
        if typ in {"strac", "straction"}:
            _require(target_info.n, "target normals")
            r4 = r2**2
            rn = rx * target_info.n[0, :, None] + ry * target_info.n[1, :, None]
            kxx = -(rx**2) * rn / (np.pi * r4)
            kyy = -(ry**2) * rn / (np.pi * r4)
            kxy = -rx * ry * rn / (np.pi * r4)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ in {"dvel", "dvelocity", "d", "double"}:
            _require(source_info.n, "source normals")
            r4 = r2**2
            rn = rx * source_info.n[0, None, :] + ry * source_info.n[1, None, :]
            kxx = rx**2 * rn / (np.pi * r4)
            kyy = ry**2 * rn / (np.pi * r4)
            kxy = rx * ry * rn / (np.pi * r4)
            return _interleave_2x2(kxx, kxy, kxy, kyy, nt, ns)
        if typ in {"dpres", "dpressure"}:
            _require(source_info.n, "source normals")
            r4 = r2**2
            rn = rx * source_info.n[0, None, :] + ry * source_info.n[1, None, :]
            kx = mu * (-source_info.n[0, None, :] / r2 + 2.0 * rx * rn / r4) / np.pi
            ky = mu * (-source_info.n[1, None, :] / r2 + 2.0 * ry * rn / r4) / np.pi
            return _interleave_1x2(kx, ky, nt, ns)
        if typ in {"dtrac", "dtraction"}:
            _require(source_info.n, "source normals")
            _require(target_info.n, "target normals")
            r4 = r2**2
            r6 = r2**3
            nsx = source_info.n[0, None, :]
            nsy = source_info.n[1, None, :]
            ntx = target_info.n[0, :, None]
            nty = target_info.n[1, :, None]
            rns = rx * nsx + ry * nsy
            rnt = rx * ntx + ry * nty
            nn = ntx * nsx + nty * nsy
            kxx = (
                mu
                / np.pi
                * (
                    -8 * rx**2 * rnt * rns / r6
                    + (rx * ntx * rns + rx**2 * nn + rnt * rns + nsx * rx * rnt) / r4
                    + ntx * nsx / r2
                )
            )
            kyy = (
                mu
                / np.pi
                * (
                    -8 * ry**2 * rnt * rns / r6
                    + (ry * nty * rns + ry**2 * nn + rnt * rns + nsy * ry * rnt) / r4
                    + nty * nsy / r2
                )
            )
            kxy = (
                mu
                / np.pi
                * (
                    -8 * rx * ry * rnt * rns / r6
                    + (rx * nty * rns + rx * ry * nn + nsx * ry * rnt) / r4
                    + ntx * nsy / r2
                )
            )
            kyx = (
                mu
                / np.pi
                * (
                    -8 * ry * rx * rnt * rns / r6
                    + (ry * ntx * rns + ry * rx * nn + nsy * rx * rnt) / r4
                    + nty * nsx / r2
                )
            )
            return _interleave_2x2(kxx, kxy, kyx, kyy, nt, ns)
        if typ in {"sgrad", "sg"}:
            r4inv = 1.0 / r2**2 / (4.0 * np.pi * mu)
            return _interleave_grad(
                rx * (ry**2 - rx**2) * r4inv,
                ry * (ry**2 - rx**2) * r4inv,
                (-3 * rx**2 * ry - ry**3) * r4inv,
                rx * (rx**2 - ry**2) * r4inv,
                ry * (ry**2 - rx**2) * r4inv,
                (-3 * rx * ry**2 - rx**3) * r4inv,
                rx * (rx**2 - ry**2) * r4inv,
                ry * (rx**2 - ry**2) * r4inv,
                nt,
                ns,
            )
        if typ in {"dgrad", "dg"}:
            _require(source_info.n, "source normals")
            r4 = r2**2
            r6 = r2**3
            nsx = source_info.n[0, None, :]
            nsy = source_info.n[1, None, :]
            rn = rx * nsx + ry * nsy
            op = 1.0 / np.pi
            return _interleave_grad(
                (rn * rx / r4 + rx * rn / r4 + rx**2 * nsx / r4 - 4 * rx**3 * rn / r6) * op,
                (rn * ry / r4 + rx * ry * nsx / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (rx**2 * nsy / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (rx * rn / r4 + rx * ry * nsy / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (ry * rn / r4 + rx * ry * nsx / r4 - 4 * rx**2 * ry * rn / r6) * op,
                (ry**2 * nsx / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (rn * rx / r4 + rx * ry * nsy / r4 - 4 * rx * ry**2 * rn / r6) * op,
                (rn * ry / r4 + ry * rn / r4 + ry**2 * nsy / r4 - 4 * ry**3 * rn / r6) * op,
                nt,
                ns,
            )
        if typ in {"cvel", "cvelocity", "c", "combined"}:
            c = np.ones(2) if coefs is None else np.asarray(coefs)
            return c[0] * kernel(mu, source_info, target_info, "d") + c[1] * kernel(
                mu, source_info, target_info, "s"
            )
        if typ in {"cpres", "cpressure"}:
            c = np.ones(2) if coefs is None else np.asarray(coefs)
            return c[0] * kernel(mu, source_info, target_info, "dpres") + c[1] * kernel(
                mu, source_info, target_info, "spres"
            )
        if typ in {"ctrac", "ctraction"}:
            c = np.ones(2) if coefs is None else np.asarray(coefs)
            return c[0] * kernel(mu, source_info, target_info, "dtrac") + c[1] * kernel(
                mu, source_info, target_info, "strac"
            )
        if typ in {"cgrad", "cg"}:
            c = np.ones(2) if coefs is None else np.asarray(coefs)
            return c[0] * kernel(mu, source_info, target_info, "dgrad") + c[1] * kernel(
                mu, source_info, target_info, "sgrad"
            )
    raise ValueError(f"Unknown Stokes kernel type {kind!r}.")


def _interleave_2x2(kxx, kxy, kyx, kyy, nt, ns):
    kernel_values = np.empty((2, nt, 2, ns), dtype=np.result_type(kxx, kxy, kyx, kyy))
    kernel_values[0, :, 0, :] = kxx
    kernel_values[0, :, 1, :] = kxy
    kernel_values[1, :, 0, :] = kyx
    kernel_values[1, :, 1, :] = kyy
    return boundary_matrix_from_kernel_tensor(kernel_values, name="Stokes kernel values")


def _interleave_1x2(kx, ky, nt, ns):
    kernel_values = np.empty((1, nt, 2, ns), dtype=np.result_type(kx, ky))
    kernel_values[0, :, 0, :] = kx
    kernel_values[0, :, 1, :] = ky
    return boundary_matrix_from_kernel_tensor(kernel_values, name="Stokes pressure kernel values")


def _interleave_grad(a, b, c, d, e, f, g, h, nt, ns):
    kernel_values = np.empty((4, nt, 2, ns), dtype=np.result_type(a, b, c, d, e, f, g, h))
    kernel_values[0, :, 0, :] = a
    kernel_values[0, :, 1, :] = b
    kernel_values[1, :, 0, :] = c
    kernel_values[1, :, 1, :] = d
    kernel_values[2, :, 0, :] = e
    kernel_values[2, :, 1, :] = f
    kernel_values[3, :, 0, :] = g
    kernel_values[3, :, 1, :] = h
    return boundary_matrix_from_kernel_tensor(kernel_values, name="Stokes gradient kernel values")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
