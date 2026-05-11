"""Core Legendre routines ported from MATLAB ``+lege``.

The public functions intentionally keep MATLAB chunkIE names. Arrays use the
same conceptual layout: Legendre coefficients live on the first axis, and
point values live on the first axis for interpolation matrices.
"""

from __future__ import annotations

from collections.abc import Callable

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


def bernstein_ellipse(ntheta: int, rho: float) -> np.ndarray:
    """Return ``ntheta`` points on the Bernstein ellipse with parameter ``rho``."""

    if ntheta <= 0:
        raise ValueError("ntheta must be positive")
    if rho <= 0:
        raise ValueError("rho must be positive")
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta + 1)[:-1]
    z = rho * np.exp(1j * theta)
    return 0.5 * (z + 1.0 / z)


def polsum(xs: ArrayLike, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``P_n``, its derivative, and the recurrence normalization sum."""

    if n < 0:
        raise ValueError("n must be non-negative")

    xs_arr = np.asarray(xs, dtype=float)
    val = np.ones_like(xs_arr)
    der = np.zeros_like(xs_arr)
    total = val**2 / 2.0
    if n == 0:
        return val, der, total

    prev = val
    val = xs_arr.copy()
    der = np.ones_like(xs_arr)
    total = total + val**2 * 1.5
    if n == 1:
        return val, der, total

    for degree in range(1, n):
        prev, val = val, ((2 * degree + 1) * xs_arr * val - degree * prev) / (degree + 1)
        total = total + val**2 * (degree + 1.5)

    endpoint = np.isclose(np.abs(xs_arr), 1.0)
    der = np.empty_like(xs_arr)
    der[~endpoint] = n * (xs_arr[~endpoint] * val[~endpoint] - prev[~endpoint]) / (
        xs_arr[~endpoint] ** 2 - 1.0
    )
    if np.any(endpoint):
        signs = np.where(xs_arr[endpoint] >= 0.0, 1.0, (-1.0) ** (n + 1))
        der[endpoint] = signs * n * (n + 1) / 2.0
    return val, der, total


def tayl(
    pol_val: ArrayLike,
    der_val: ArrayLike,
    x: ArrayLike,
    h: ArrayLike,
    n: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a Legendre polynomial Taylor step from ``x`` to ``x+h``."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        raise ValueError("k must be non-negative")

    pol_arr, der_arr, x_arr, h_arr = np.broadcast_arrays(
        np.asarray(pol_val, dtype=float),
        np.asarray(der_val, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(h, dtype=float),
    )
    out_pol = np.empty_like(pol_arr, dtype=float)
    out_der = np.empty_like(pol_arr, dtype=float)
    zero_step = h_arr == 0.0
    if np.any(zero_step):
        out_pol[zero_step] = pol_arr[zero_step]
        out_der[zero_step] = der_arr[zero_step]
    if not np.all(zero_step):
        nz = ~zero_step
        p_new, d_new = _tayl_nonzero_h(pol_arr[nz], der_arr[nz], x_arr[nz], h_arr[nz], n, k)
        out_pol[nz] = p_new
        out_der[nz] = d_new
    return out_pol, out_der


def adapgauss(
    fun: Callable[[np.ndarray], ArrayLike],
    a: float,
    b: float,
    t: ArrayLike | None = None,
    w: ArrayLike | None = None,
) -> tuple[np.ndarray, int, int, int]:
    """Adaptive Gauss-Legendre integration matching MATLAB ``lege.adapgauss``."""

    if t is None or w is None:
        t_arr, w_arr = exps(16)[:2]
    else:
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        w_arr = np.asarray(w, dtype=float).reshape(-1)
    if t_arr.shape != w_arr.shape:
        raise ValueError("t and w must have the same shape")

    eps = 1e-12
    nnmax = 100000
    maxdepth = 200

    stack = np.zeros((maxdepth, 2), dtype=float)
    vals: list[np.ndarray | None] = [None] * maxdepth
    stack[0] = (float(a), float(b))
    vals[0] = _oneintp(fun, float(a), float(b), t_arr, w_arr)

    depth = 0
    total = np.zeros_like(vals[0])
    ier = 0
    maxrec = 0
    numint = 0
    for idx in range(nnmax):
        numint = idx + 1
        maxrec = max(maxrec, depth + 1)
        left, right = stack[depth]
        mid = 0.5 * (left + right)
        left_val = _oneintp(fun, left, mid, t_arr, w_arr)
        right_val = _oneintp(fun, mid, right, t_arr, w_arr)
        current = vals[depth]
        assert current is not None

        if np.all(np.abs(left_val + right_val - current) <= eps):
            total = total + left_val + right_val
            vals[depth] = None
            depth -= 1
            if depth < 0:
                return total, maxrec, numint, ier
            continue

        if depth + 1 >= maxdepth:
            ier = 8
            return total, maxrec, numint, ier
        stack[depth + 1] = (left, mid)
        vals[depth + 1] = left_val
        stack[depth] = (mid, right)
        vals[depth] = right_val
        depth += 1

    ier = 16
    return total, maxrec, numint, ier


def _tayl_nonzero_h(
    pol_arr: np.ndarray,
    der_arr: np.ndarray,
    x_arr: np.ndarray,
    h_arr: np.ndarray,
    n: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    q0 = pol_arr
    q1 = der_arr * h_arr
    q2 = (2 * x_arr * der_arr - n * (n + 1) * pol_arr) / (1 - x_arr**2)
    q2 = q2 * h_arr**2 / 2.0

    pol_new = q0 + q1 + q2
    der_new = q1 / h_arr + (q2 * 2.0) / h_arr
    if k <= 2:
        return pol_new, der_new

    qi = q1
    qip1 = q2
    for order in range(1, k - 1):
        d = 2 * (x_arr * (order + 1) ** 2) / h_arr * qip1
        d = d - (n * (n + 1) - order * (order + 1)) * qi
        d = d / (order + 1) / (order + 2) * h_arr**2 / (1 - x_arr**2)
        pol_new = pol_new + d
        der_new = der_new + d * (order + 2) / h_arr
        qi = qip1
        qip1 = d
    return pol_new, der_new


def _oneintp(
    fun: Callable[[np.ndarray], ArrayLike], a: float, b: float, t: np.ndarray, w: np.ndarray
) -> np.ndarray:
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    nodes = scale * t + shift
    vals = np.asarray(fun(nodes))
    if vals.shape == ():
        return vals * np.sum(w) * scale
    if vals.shape[0] == t.size:
        return np.tensordot(w, vals, axes=(0, 0)) * scale
    if vals.shape[-1] == t.size:
        return np.tensordot(vals, w, axes=(-1, 0)) * scale
    raise ValueError("integrand output must have one axis matching t")
