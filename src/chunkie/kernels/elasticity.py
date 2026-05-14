"""Two-dimensional linear elasticity kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import boundary_matrix_from_kernel_tensor
from chunkie.geometry import PointInfo


def kernel(
    lam: float,
    mu: float,
    source: PointInfo | dict | ArrayLike,
    target: PointInfo | dict | ArrayLike,
    kind: str = "s",
) -> np.ndarray:
    source_info = PointInfo.from_any(source)
    target_info = PointInfo.from_any(target)
    typ = kind.lower()
    beta = (lam + 3.0 * mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    gamma = -(lam + mu) / (4.0 * np.pi * mu * (lam + 2.0 * mu))
    eta = mu / (2.0 * np.pi * (lam + 2.0 * mu))
    zeta = (lam + mu) / (np.pi * (lam + 2.0 * mu))

    x = target_info.r[0, :, None] - source_info.r[0, None, :]
    y = target_info.r[1, :, None] - source_info.r[1, None, :]
    r2 = x**2 + y**2
    r4 = r2**2
    nt, ns = x.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        if typ in {"s", "single"}:
            logr = beta * np.log(r2) / 2.0
            kxx = logr + gamma / 2.0 + gamma * x**2 / r2
            kxy = gamma * x * y / r2
            kyy = logr + gamma / 2.0 + gamma * y**2 / r2
            return _interleave(kxx, kxy, kxy, kyy, nt, ns)
        if typ == "strac":
            _require(target_info.n, "target normals")
            nx = target_info.n[0, :, None]
            ny = target_info.n[1, :, None]
            rn = x * nx + y * ny
            term = zeta * rn / r4
            kxx = eta * rn / r2 + term * x**2
            kxy = eta * (x * ny - y * nx) / r2 + term * x * y
            kyx = eta * (y * nx - x * ny) / r2 + term * x * y
            kyy = eta * rn / r2 + term * y**2
            return _interleave(kxx, kxy, kyx, kyy, nt, ns)
        if typ in {"sgrad", "sg"}:
            out = np.zeros((4 * nt, 2 * ns), dtype=np.result_type(x, y, beta, gamma))
            out[0::4, 0::2] = beta * x / r2
            out[2::4, 1::2] = beta * x / r2
            out[1::4, 0::2] = beta * y / r2
            out[3::4, 1::2] = beta * y / r2

            out[0::4, 0::2] += gamma * (2.0 * r2 * x - 2.0 * x**3) / r4
            out[1::4, 0::2] += gamma * (-2.0 * x**2 * y) / r4
            tmp = gamma * (r2 * y - 2.0 * x**2 * y) / r4
            out[2::4, 0::2] += tmp
            out[0::4, 1::2] += tmp
            tmp = gamma * (r2 * x - 2.0 * x * y**2) / r4
            out[3::4, 0::2] += tmp
            out[1::4, 1::2] += tmp
            out[2::4, 1::2] += gamma * (-2.0 * y**2 * x) / r4
            out[3::4, 1::2] += gamma * (2.0 * r2 * y - 2.0 * y**3) / r4
            return out
        if typ in {"d", "double"}:
            _require(source_info.n, "source normals")
            nx = source_info.n[0, None, :]
            ny = source_info.n[1, None, :]
            rn = x * nx + y * ny
            term = zeta * rn / r4
            kxx = -(eta * rn / r2 + term * x**2)
            kxy = -(eta * (-x * ny + y * nx) / r2 + term * x * y)
            kyx = -(eta * (-y * nx + x * ny) / r2 + term * x * y)
            kyy = -(eta * rn / r2 + term * y**2)
            return _interleave(kxx, kxy, kyx, kyy, nt, ns)
        if typ == "dalt":
            _require(source_info.n, "source normals")
            rn = x * source_info.n[0, None, :] + y * source_info.n[1, None, :]
            term = -zeta * rn / r4
            diag = -2.0 * eta * rn / r2
            kxx = diag + term * x**2
            kxy = term * x * y
            kyy = diag + term * y**2
            return _interleave(kxx, kxy, kxy, kyy, nt, ns)
        if typ in {"daltgrad", "daltg", "dalttrac"}:
            _require(source_info.n, "source normals")
            nx = source_info.n[0, None, :]
            ny = source_info.n[1, None, :]
            rn = x * nx + y * ny
            r6 = r4 * r2
            grad = np.zeros((4 * nt, 2 * ns), dtype=np.result_type(x, y, lam, mu))

            grad[0::4, 0::2] = -zeta * (
                -4.0 * x**3 * rn / r6 + (2.0 * x * rn + x**2 * nx) / r4
            ) - 2.0 * eta * (nx / r2 - 2.0 * rn * x / r4)
            grad[1::4, 0::2] = -zeta * (-4.0 * x**2 * y * rn / r6 + x**2 * ny / r4) - 2.0 * eta * (
                ny / r2 - 2.0 * rn * y / r4
            )
            grad[2::4, 0::2] = -zeta * (-4.0 * x**2 * y * rn / r6 + (y * rn + x * y * nx) / r4)
            grad[3::4, 0::2] = -zeta * (-4.0 * x * y**2 * rn / r6 + (x * rn + x * y * ny) / r4)
            grad[0::4, 1::2] = -zeta * (-4.0 * x**2 * y * rn / r6 + (y * rn + x * y * nx) / r4)
            grad[1::4, 1::2] = -zeta * (-4.0 * x * y**2 * rn / r6 + (x * rn + x * y * ny) / r4)
            grad[2::4, 1::2] = -zeta * (-4.0 * y**2 * x * rn / r6 + y**2 * nx / r4) - 2.0 * eta * (
                nx / r2 - 2.0 * rn * x / r4
            )
            grad[3::4, 1::2] = -zeta * (
                -4.0 * y**3 * rn / r6 + (2.0 * y * rn + y**2 * ny) / r4
            ) - 2.0 * eta * (ny / r2 - 2.0 * rn * y / r4)

            if typ in {"daltgrad", "daltg"}:
                return grad

            _require(target_info.n, "target normals")
            n1 = target_info.n[0, :, None]
            n2 = target_info.n[1, :, None]
            out = np.zeros((2 * nt, 2 * ns), dtype=grad.dtype)
            div = grad[0::4, :] + grad[3::4, :]
            shear = mu * (grad[1::4, :] + grad[2::4, :])
            out[0::2, :] = lam * n1 * div + shear * n2 + 2.0 * mu * grad[0::4, :] * n1
            out[1::2, :] = lam * n2 * div + shear * n1 + 2.0 * mu * grad[3::4, :] * n2
            return out
    raise ValueError(f"Unknown elasticity kernel type {kind!r}.")


def _interleave(kxx, kxy, kyx, kyy, nt, ns):
    kernel_values = np.empty((2, nt, 2, ns), dtype=np.result_type(kxx, kxy, kyx, kyy))
    kernel_values[0, :, 0, :] = kxx
    kernel_values[0, :, 1, :] = kxy
    kernel_values[1, :, 0, :] = kyx
    kernel_values[1, :, 1, :] = kyy
    return boundary_matrix_from_kernel_tensor(kernel_values, name="elasticity kernel values")


def _require(value: object, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} are required")
