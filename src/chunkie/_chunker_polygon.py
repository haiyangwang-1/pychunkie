"""Private polygon-construction helpers for :mod:`chunkie.chunker`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .chunker import Chunker, ChunkerPref


def _dyadic_chunkerpoly(
    vertices: np.ndarray,
    cparams: dict[str, Any],
    pref: "ChunkerPref",
    edgevals: ArrayLike | None,
) -> "Chunker":
    from .chunker import Chunker, ChunkerPref

    ifclosed = bool(cparams.get("ifclosed", True))
    depth = int(cparams.get("depth", 30))
    if depth < 0:
        raise ValueError("depth must be a nonnegative integer")

    if ifclosed:
        starts = vertices
        ends = np.column_stack((vertices[:, 1:], vertices[:, 0]))
    else:
        starts = vertices[:, :-1]
        ends = vertices[:, 1:]
    nedge = starts.shape[1]
    lengths = np.sqrt(np.sum((ends - starts) ** 2, axis=0))
    if np.any(lengths <= 0.0):
        raise ValueError("polygon edges must have positive length")

    widths = _polygon_widths(vertices, lengths, cparams, ifclosed)
    eps = np.asarray(cparams.get("eps", 1.0e-6), dtype=float).reshape(-1)
    eps0 = float(eps[0]) if eps.size else 1.0e-6
    ncorner = nedge if ifclosed else max(nedge - 1, 0)
    nch = nedge + 2 * ncorner * (depth + 1)
    if nch > pref.nchmax:
        raise ValueError("too many polygon chunks for nchmax")

    edge_data = None
    if edgevals is not None:
        edge_data = np.asarray(edgevals, dtype=float)
        if edge_data.size % nedge != 0:
            raise ValueError("number of edge values should be multiple of number of edges")
        edge_data = edge_data.reshape(edge_data.size // nedge, nedge)

    p = ChunkerPref(pref.nchmax, pref.k, pref.dim, max(pref.nchstor, nch), pref.verttol)
    chnkr = Chunker(p).addchunk(nch)
    if edge_data is not None:
        chnkr.makedatarows(edge_data.shape[0])

    breaks = [0.0] + [2.0 ** (ilevel - depth) for ilevel in range(depth + 1)]

    ich = 0
    for iedge in range(nedge):
        start = starts[:, iedge]
        end = ends[:, iedge]
        length = float(lengths[iedge])
        tangent = (end - start) / length
        w0 = float(widths[iedge])
        w1 = float(widths[(iedge + 1) % widths.size] if ifclosed else widths[iedge + 1])
        if length <= w0 + w1 + 2.0 * eps0 * length:
            raise ValueError("widths too large for side")

        _fill_line_chunk(chnkr, ich, start + tangent * w0, end - tangent * w1)
        if edge_data is not None:
            chnkr.datastor[:, :, ich] = edge_data[:, iedge][:, None]
        ich += 1

        if not ifclosed and iedge == nedge - 1:
            continue

        next_edge = (iedge + 1) % nedge
        next_start = starts[:, next_edge]
        next_end = ends[:, next_edge]
        next_tangent = (next_end - next_start) / float(lengths[next_edge])
        corner = end
        width = w1

        for hi, lo in zip(reversed(breaks[1:]), reversed(breaks[:-1])):
            _fill_line_chunk(chnkr, ich, corner - tangent * (hi * width), corner - tangent * (lo * width))
            if edge_data is not None:
                chnkr.datastor[:, :, ich] = edge_data[:, iedge][:, None]
            ich += 1

        for lo, hi in zip(breaks[:-1], breaks[1:]):
            _fill_line_chunk(chnkr, ich, corner + next_tangent * (lo * width), corner + next_tangent * (hi * width))
            if edge_data is not None:
                chnkr.datastor[:, :, ich] = edge_data[:, next_edge][:, None]
            ich += 1

    if ich != nch:
        raise RuntimeError("dyadic polygon chunk count mismatch")
    adjs = np.zeros((2, nch), dtype=int)
    adjs[0] = np.arange(0, nch)
    adjs[1] = np.arange(2, nch + 2)
    if ifclosed:
        adjs[0, 0] = nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    refined = chnkr.refine()
    return refined.sort()[0]


def _rounded_chunkerpoly(
    vertices: np.ndarray,
    cparams: dict[str, Any],
    pref: "ChunkerPref",
    edgevals: ArrayLike | None,
) -> "Chunker":
    from .chunker import Chunker, ChunkerPref

    if vertices.shape[0] != 2:
        raise NotImplementedError("rounded chunkerpoly currently supports two-dimensional vertices")

    ifclosed = bool(cparams.get("ifclosed", True))
    nv = vertices.shape[1]
    nedge = nv if ifclosed else nv - 1
    edges = [
        (vertices[:, i], vertices[:, (i + 1) % nv])
        for i in range(nedge)
    ]
    lengths = np.array([np.linalg.norm(end - start) for start, end in edges])
    if np.any(lengths <= 0.0):
        raise ValueError("polygon edges must have positive length")

    widths = _polygon_widths(vertices, lengths, cparams, ifclosed)
    for iedge, length in enumerate(lengths):
        w0 = widths[iedge]
        w1 = widths[(iedge + 1) % nv] if ifclosed else widths[iedge + 1]
        if w0 + w1 >= length:
            scale = 0.45 * length / (w0 + w1)
            widths[iedge] *= scale
            widths[(iedge + 1) % nv if ifclosed else iedge + 1] *= scale

    edge_data = None
    if edgevals is not None:
        edge_data = np.asarray(edgevals, dtype=float)
        if edge_data.size % nedge != 0:
            raise ValueError("number of edge values should be multiple of number of edges")
        edge_data = edge_data.reshape(edge_data.size // nedge, nedge)

    ncorner = nv if ifclosed else max(nv - 2, 0)
    nch = nedge + ncorner
    p = ChunkerPref(pref.nchmax, pref.k, 2, max(pref.nchstor, nch), pref.verttol)
    chnkr = Chunker(p).addchunk(nch)
    if edge_data is not None:
        chnkr.makedatarows(edge_data.shape[0])

    t = chnkr.tstor
    u = (t + 1.0) / 2.0
    ich = 0
    for iedge, (start, end) in enumerate(edges):
        tangent = (end - start) / lengths[iedge]
        w0 = widths[iedge]
        w1 = widths[(iedge + 1) % nv] if ifclosed else widths[iedge + 1]
        p0 = start + tangent * w0
        p1 = end - tangent * w1
        _fill_line_chunk(chnkr, ich, p0, p1)
        if edge_data is not None:
            chnkr.datastor[:, :, ich] = edge_data[:, iedge][:, None]
        ich += 1

        corner_vertex = (iedge + 1) % nv
        if ifclosed or iedge < nedge - 1:
            if not ifclosed and (corner_vertex == 0 or corner_vertex == nv - 1):
                continue
            next_edge = (iedge + 1) % nedge
            next_start, next_end = edges[next_edge]
            next_tangent = (next_end - next_start) / lengths[next_edge]
            c0 = p1
            c1 = vertices[:, corner_vertex]
            c2 = c1 + next_tangent * widths[corner_vertex]
            _fill_quadratic_chunk(chnkr, ich, c0, c1, c2, u)
            if edge_data is not None:
                val0 = edge_data[:, iedge]
                val1 = edge_data[:, next_edge]
                chnkr.datastor[:, :, ich] = val0[:, None] * (1.0 - u)[None, :] + val1[:, None] * u[None, :]
            ich += 1

    if ich != nch:
        chnkr.nch = ich
    adjs = np.zeros((2, chnkr.nch), dtype=int)
    adjs[0] = np.arange(0, chnkr.nch)
    adjs[1] = np.arange(2, chnkr.nch + 2)
    if ifclosed:
        adjs[0, 0] = chnkr.nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    return chnkr


def _polygon_widths(vertices: np.ndarray, lengths: np.ndarray, cparams: dict[str, Any], ifclosed: bool) -> np.ndarray:
    nv = vertices.shape[1]
    if "widths" in cparams:
        widths = np.asarray(cparams["widths"], dtype=float).reshape(-1)
        if widths.size == 1:
            widths = np.full(nv, float(widths[0]))
        if widths.size != nv:
            raise ValueError("widths must be scalar or one value per vertex")
    else:
        fac = float(cparams.get("autowidthsfac", 0.1))
        widths = np.zeros(nv)
        for iv in range(nv):
            if not ifclosed and (iv == 0 or iv == nv - 1):
                continue
            left = lengths[iv - 1 if ifclosed else iv - 1]
            right = lengths[iv % lengths.size]
            widths[iv] = fac * min(left, right)
    if not ifclosed:
        widths[0] = 0.0
        widths[-1] = 0.0
    if np.any(widths < 0.0):
        raise ValueError("widths must be nonnegative")
    return widths


def _fill_line_chunk(chnkr: "Chunker", ich: int, start: np.ndarray, end: np.ndarray) -> None:
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("rounded polygon produced a zero-length straight panel")
    u = (chnkr.tstor + 1.0) / 2.0
    tangent = delta / length
    h = length / 2.0
    chnkr.rstor[:, :, ich] = start[:, None] + delta[:, None] * u[None, :]
    chnkr.dstor[:, :, ich] = tangent[:, None] * h
    chnkr.d2stor[:, :, ich] = 0.0


def _fill_quadratic_chunk(chnkr: "Chunker", ich: int, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, u: np.ndarray) -> None:
    omt = 1.0 - u
    chnkr.rstor[:, :, ich] = omt[None, :] ** 2 * p0[:, None] + 2.0 * omt[None, :] * u[None, :] * p1[:, None] + u[None, :] ** 2 * p2[:, None]
    drdu = 2.0 * omt[None, :] * (p1 - p0)[:, None] + 2.0 * u[None, :] * (p2 - p1)[:, None]
    chnkr.dstor[:, :, ich] = drdu / 2.0
    chnkr.d2stor[:, :, ich] = (p2 - 2.0 * p1 + p0)[:, None] / 2.0
