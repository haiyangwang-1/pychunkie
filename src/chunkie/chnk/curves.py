"""Basic parameterized curves from MATLAB ``+chnk/+curves``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def linefunc(t: ArrayLike, v1: ArrayLike, v2: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    start = np.asarray(v1, dtype=float).reshape(2, 1)
    end = np.asarray(v2, dtype=float).reshape(2, 1)
    delta = end - start
    r = start + delta * flat[None, :]
    d = delta * np.ones((1, flat.size))
    d2 = np.zeros_like(d)
    return r.reshape((2,) + t_arr.shape), d.reshape((2,) + t_arr.shape), d2.reshape((2,) + t_arr.shape)


def fpara(t: ArrayLike, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    rx = flat
    ry = a * (flat - b) ** 2
    dx = np.ones_like(flat)
    dy = 2 * a * (flat - b)
    d2x = np.zeros_like(flat)
    d2y = np.full_like(flat, 2 * a)
    return _pack(t_arr, rx, ry, dx, dy, d2x, d2y)


def fsine(t: ArrayLike, a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    rx = flat
    ry = a * np.sin(b * flat + c)
    dx = np.ones_like(flat)
    dy = a * b * np.cos(b * flat + c)
    d2x = np.zeros_like(flat)
    d2y = -a * b * b * np.sin(b * flat + c)
    return _pack(t_arr, rx, ry, dx, dy, d2x, d2y)


def bymode(
    t: ArrayLike,
    modes: ArrayLike,
    ctr: ArrayLike | None = None,
    sc: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    modes_arr = np.asarray(modes, dtype=float).reshape(-1)
    center = np.zeros(2) if ctr is None else np.asarray(ctr, dtype=float).reshape(2)
    scale = np.ones(2) if sc is None else np.asarray(sc, dtype=float).reshape(2)

    eitfac = np.ones_like(flat, dtype=complex)
    radius = modes_arr[0] * np.ones_like(flat)
    rp = np.zeros_like(flat)
    rpp = np.zeros_like(flat)
    eit = np.exp(1j * flat)
    for idx in range(1, len(modes_arr), 2):
        eitfac *= eit
        ih = (idx + 1) // 2
        radius += np.real(eitfac) * modes_arr[idx]
        rp += -ih * np.imag(eitfac) * modes_arr[idx]
        rpp += -(ih**2) * np.real(eitfac) * modes_arr[idx]
        if idx + 1 < len(modes_arr):
            radius += np.imag(eitfac) * modes_arr[idx + 1]
            rp += ih * np.real(eitfac) * modes_arr[idx + 1]
            rpp += -(ih**2) * np.imag(eitfac) * modes_arr[idx + 1]

    cost = np.cos(flat)
    sint = np.sin(flat)
    xs = center[0] + radius * cost * scale[0]
    ys = center[1] + radius * sint * scale[1]
    dxs = (cost * rp - radius * sint) * scale[0]
    dys = (sint * rp + radius * cost) * scale[1]
    d2xs = (cost * rpp - 2.0 * sint * rp - radius * cost) * scale[0]
    d2ys = (sint * rpp + 2.0 * cost * rp - radius * sint) * scale[1]
    return _pack(t_arr, xs, ys, dxs, dys, d2xs, d2ys)


def _pack(
    t: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    d2x: np.ndarray,
    d2y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.vstack((rx, ry)).reshape((2,) + t.shape)
    d = np.vstack((dx, dy)).reshape((2,) + t.shape)
    d2 = np.vstack((d2x, d2y)).reshape((2,) + t.shape)
    return r, d, d2
