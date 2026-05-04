"""Core Legendre routines ported from MATLAB ``+lege``.

The public functions intentionally keep MATLAB chunkIE names. Arrays use the
same conceptual layout: Legendre coefficients live on the first axis, and
point values live on the first axis for interpolation matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pol(xs: ArrayLike, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the degree-``n`` Legendre polynomial and derivative."""

    if n < 0:
        raise ValueError("n must be non-negative")

    xs_arr = np.asarray(xs, dtype=float)
    if n == 0:
        return np.ones_like(xs_arr), np.zeros_like(xs_arr)
    if n == 1:
        return xs_arr.copy(), np.ones_like(xs_arr)

    pkm1 = np.ones_like(xs_arr)
    pk = xs_arr.copy()
    for k in range(1, n):
        pkp1 = ((2 * k + 1) * xs_arr * pk - k * pkm1) / (k + 1)
        pkm1, pk = pk, pkp1

    der = np.empty_like(xs_arr, dtype=float)
    mask = np.isclose(np.abs(xs_arr), 1.0)
    der[~mask] = n * (xs_arr[~mask] * pk[~mask] - pkm1[~mask]) / (
        xs_arr[~mask] ** 2 - 1.0
    )
    if np.any(mask):
        signs = np.where(xs_arr[mask] >= 0.0, 1.0, (-1.0) ** (n + 1))
        der[mask] = signs * n * (n + 1) / 2.0
    return pk, der


def pols(xs: ArrayLike, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Legendre polynomials up to degree ``n`` and derivatives."""

    if n < 0:
        raise ValueError("n must be non-negative")

    xs_arr = np.asarray(xs, dtype=float)
    flat = xs_arr.reshape(-1)
    vals = np.zeros((n + 1, flat.size), dtype=float)
    ders = np.zeros_like(vals)
    vals[0] = 1.0
    if n >= 1:
        vals[1] = flat
        ders[1] = 1.0

    for k in range(1, n):
        vals[k + 1] = ((2 * k + 1) * flat * vals[k] - k * vals[k - 1]) / (k + 1)

    if n >= 2:
        denom = flat**2 - 1.0
        endpoint = np.isclose(np.abs(flat), 1.0)
        for degree in range(2, n + 1):
            ders[degree, ~endpoint] = degree * (
                flat[~endpoint] * vals[degree, ~endpoint] - vals[degree - 1, ~endpoint]
            ) / denom[~endpoint]
            if np.any(endpoint):
                signs = np.where(
                    flat[endpoint] >= 0.0, 1.0, (-1.0) ** (degree + 1)
                )
                ders[degree, endpoint] = signs * degree * (degree + 1) / 2.0

    out_shape = (n + 1,) + xs_arr.shape
    return vals.reshape(out_shape), ders.reshape(out_shape)


def exps(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Gaussian nodes, weights, and value/coefficient transforms."""

    if k <= 0:
        raise ValueError("k must be positive")

    x, w = np.polynomial.legendre.leggauss(k)
    pvals, _ = pols(x, k - 1)
    v = pvals.T
    scale = (2.0 * np.arange(1, k + 1) - 1.0) / 2.0
    u = (v * (w[:, None] * scale[None, :])).T
    return x, w, u, v


def rts(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss nodes and weights."""

    if n <= 0:
        raise ValueError("n must be positive")
    return np.polynomial.legendre.leggauss(n)


def rts_stab(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable Legendre-Gauss nodes and weights.

    NumPy's Golub-Welsch implementation is the stable path for this port.
    """

    return rts(n)


def exev(xs: ArrayLike, coeff: ArrayLike) -> np.ndarray:
    """Evaluate a Legendre expansion at points ``xs``."""

    xs_arr = np.asarray(xs, dtype=float)
    coeff_arr = np.asarray(coeff)
    if coeff_arr.ndim == 1:
        vals, _ = pols(xs_arr, coeff_arr.size - 1)
        return np.moveaxis(vals, 0, -1) @ coeff_arr
    vals, _ = pols(xs_arr, coeff_arr.shape[0] - 1)
    return np.moveaxis(vals, 0, -1) @ coeff_arr


def derpol(coeffs: ArrayLike) -> np.ndarray:
    """Compute Legendre coefficients of the derivative."""

    coeffs_arr = np.asarray(coeffs)
    if coeffs_arr.shape[0] == 0:
        return np.zeros_like(coeffs_arr)

    out = np.zeros((max(coeffs_arr.shape[0] - 1, 0),) + coeffs_arr.shape[1:], dtype=coeffs_arr.dtype)
    for degree in range(1, coeffs_arr.shape[0]):
        for target in range(degree - 1, -1, -2):
            out[target] += (2 * target + 1) * coeffs_arr[degree]
    return out


def dermat(k: int, u: ArrayLike | None = None, v: ArrayLike | None = None) -> np.ndarray:
    """Return the spectral differentiation matrix on Legendre nodes."""

    if u is None or v is None:
        _, _, u_arr, v_arr = exps(k)
    else:
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)
    return v_arr[:, :-1] @ derpol(u_arr)


def intpol(coeffs: ArrayLike, const_option: str = "true") -> np.ndarray:
    """Compute Legendre coefficients of an indefinite integral."""

    coeffs_arr = np.asarray(coeffs)
    out = np.zeros((coeffs_arr.shape[0] + 1,) + coeffs_arr.shape[1:], dtype=coeffs_arr.dtype)

    option = const_option.lower()
    if option == "true":
        ncc = out.shape[0]
    elif option == "original":
        ncc = out.shape[0] - 1
    else:
        raise ValueError("unknown option for constant")

    for i in range(1, coeffs_arr.shape[0]):
        j = i
        out[i + 1] = coeffs_arr[i] / (2 * j + 1)
        out[i - 1] += -coeffs_arr[i] / (2 * j + 1)
    out[1] += coeffs_arr[0]

    sign = -1.0
    accum = np.zeros_like(out[-1])
    for idx in range(1, ncc):
        accum += out[idx] * sign
        sign = -sign
    out[0] = -accum
    return out


def intmat(
    n: int, u: ArrayLike | None = None, v: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the spectral integration matrix on Gaussian nodes."""

    if u is None or v is None:
        _, _, u_arr, v_arr = exps(n)
    else:
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)
    tmp = intpol(u_arr, "original")
    return v_arr @ tmp[:-1], u_arr, v_arr


def matrin(
    n: int, xs: ArrayLike, u: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Form the matrix interpolating from ``n`` Legendre nodes to ``xs``."""

    if u is None:
        x, w, u_arr, v = exps(n)
    else:
        x, w, _, v = exps(n)
        u_arr = np.asarray(u)
    xs_vals, _ = pols(xs, n - 1)
    mat = np.moveaxis(xs_vals, 0, -1) @ u_arr
    return mat, x, w, u_arr, v


def barywts(k: int, x: ArrayLike | None = None) -> np.ndarray:
    """Return barycentric Lagrange interpolation weights."""

    if x is None:
        x_arr, *_ = exps(k)
    else:
        x_arr = np.asarray(x, dtype=float)
    diffs = x_arr[:, None] - x_arr[None, :]
    np.fill_diagonal(diffs, 1.0)
    w = np.prod(diffs, axis=0)
    return w[0] / w
