"""Top-level geometry and domain helpers from MATLAB ``chunkIE``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._chunkgraph_regions import _mergeregions, _pointinregion, _regioninside
from ._hypoctree import HypOctNode, HypOctTree, hypoct_uni
from .chunkgraph import ChunkGraph

__all__ = [
    "HypOctNode",
    "HypOctTree",
    "checkcurveparam",
    "ellipse",
    "hypoct_uni",
    "mergeregions",
    "nonflatinterface",
    "pointinregion",
    "redblue",
    "regioninside",
    "starfish",
]


def checkcurveparam(fcurve: Callable[[np.ndarray], Any], ta: ArrayLike, nout: int = 3) -> int:
    """Validate curve callback output shapes and return the ambient dimension."""

    t = np.asarray(ta, dtype=float)
    raw = fcurve(t)
    outs = raw if isinstance(raw, tuple) else (raw,)
    if nout > len(outs):
        raise ValueError("curve callback returned fewer outputs than requested")

    dims: list[int] = []
    for out in outs[:nout]:
        arr = np.asarray(out)
        if arr.ndim == 0:
            raise ValueError("curve outputs must have at least one dimension")
        if int(np.prod(arr.shape[1:], dtype=int)) != t.size:
            raise ValueError("size of each curve output should match input")
        dims.append(int(arr.shape[0]))

    dim = dims[0]
    if any(item != dim for item in dims):
        raise ValueError("dimension of curve output should be consistent")
    return dim


def ellipse(
    t: ArrayLike, a: float = 1.0, b: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position and derivatives for ``(a cos(t), b sin(t))``."""

    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    r = np.vstack((a * np.cos(flat), b * np.sin(flat)))
    d = np.vstack((-a * np.sin(flat), b * np.cos(flat)))
    d2 = np.vstack((-a * np.cos(flat), -b * np.sin(flat)))
    return _reshape_curve_outputs(t_arr, r, d, d2)


def starfish(
    t: ArrayLike,
    narms: int = 5,
    amp: float = 0.3,
    ctr: ArrayLike | None = None,
    phi: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position and derivatives for the standard starfish curve."""

    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    center = np.zeros(2) if ctr is None else np.asarray(ctr, dtype=float).reshape(2)

    ct = np.cos(flat)
    st = np.sin(flat)
    cnt = np.cos(narms * (flat + phi))
    snt = np.sin(narms * (flat + phi))
    radius = 1.0 + amp * cnt

    xs = center[0] + radius * ct * scale
    ys = center[1] + radius * st * scale
    dx = -(radius) * st - narms * amp * snt * ct
    dy = radius * ct - narms * amp * snt * st
    dxs = dx * scale
    dys = dy * scale
    d2xs = (-dy - narms * amp * (narms * cnt * ct - snt * st)) * scale
    d2ys = (dx - narms * amp * (narms * cnt * st + snt * ct)) * scale
    r = np.vstack((xs, ys))
    d = np.vstack((dxs, dys))
    d2 = np.vstack((d2xs, d2ys))
    return _reshape_curve_outputs(t_arr, r, d, d2)


def nonflatinterface(
    t: ArrayLike,
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the perturbed interface ``(t, d exp(-a t^2 / 2) sin(b t + c))``."""

    t_arr = np.asarray(t, dtype=float)
    flat = t_arr.reshape(-1)
    expfac = np.exp(-a * flat**2 / 2.0)
    phase = b * flat + c

    xs = flat
    xp = np.ones_like(flat)
    xpp = np.zeros_like(flat)
    ys = d * expfac * np.sin(phase)
    yp = d * expfac * (b * np.cos(phase) - a * flat * np.sin(phase))
    ypp = (
        d
        * expfac
        * (-2.0 * a * b * flat * np.cos(phase) + (a * a * flat**2 - a - b * b) * np.sin(phase))
    )
    r = np.vstack((xs, ys))
    dr = np.vstack((xp, yp))
    d2r = np.vstack((xpp, ypp))
    return _reshape_curve_outputs(t_arr, r, dr, d2r)


def redblue(m: int = 64) -> np.ndarray:
    """Return the MATLAB ``redblue`` colormap as an ``m x 3`` array."""

    m = int(m)
    if m < 0:
        raise ValueError("m must be nonnegative")
    if m == 0:
        return np.zeros((0, 3))

    if m % 2 == 0:
        m1 = m // 2
        ramp = np.arange(m1, dtype=float) / max(m1 - 1, 1)
        r = np.concatenate((ramp, np.ones(m1)))
        g = np.concatenate((ramp, ramp[::-1]))
        b = r[::-1]
    else:
        m1 = m // 2
        ramp = np.arange(m1, dtype=float) / max(m1, 1)
        r = np.concatenate((ramp, np.ones(m1 + 1)))
        g = np.concatenate((ramp, np.array([1.0]), ramp[::-1]))
        b = r[::-1]
    return np.column_stack((r, g, b))


def pointinregion(graph: ChunkGraph, region: list[list[int]], point: ArrayLike) -> int:
    """Count loops in ``region`` that contain ``point``."""

    return _pointinregion(graph, region, np.asarray(point, dtype=float).reshape(2))


def regioninside(
    graph: ChunkGraph,
    containing_region: list[list[list[int]]],
    candidate_region: list[list[list[int]]],
) -> bool:
    """Return whether a representative point of ``candidate_region`` lies inside ``containing_region``."""

    return _regioninside(graph, containing_region, candidate_region)


def mergeregions(
    graph: ChunkGraph,
    first_region: list[list[list[int]]],
    second_region: list[list[list[int]]],
) -> list[list[list[int]]]:
    """Merge two MATLAB-style chunkgraph region lists."""

    return _mergeregions(graph, first_region, second_region)


def _reshape_curve_outputs(
    t: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (2,) + t.shape
    return r.reshape(shape), d.reshape(shape), d2.reshape(shape)
