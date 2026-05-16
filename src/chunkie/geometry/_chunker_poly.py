"""Polygon/polyline chunker constructor."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._chunker_class import Chunker
from ._chunker_options import _legacy_options, _pref_with_order, _set_option
from ._chunker_polygon import _dyadic_chunkerpoly, _rounded_chunkerpoly
from ._chunker_pref import ChunkerPref


def chunkerpoly(
    verts: ArrayLike,
    cparams: dict[str, Any] | None = None,
    pref: ChunkerPref | dict[str, Any] | None = None,
    edgevals: ArrayLike | None = None,
    *,
    order: int | None = None,
    closed: bool | None = None,
    dyadic: bool | None = None,
    depth: int | None = None,
    rounded: bool | None = None,
    widths: ArrayLike | None = None,
) -> Chunker:
    """Create a chunker for a true polygon or open polyline.

    By default each polygon edge is one straight chunk. With
    ``cparams={"dyadic": True, "depth": ...}``, edges are refined
    geometrically near corners for non-smooth boundary-integral workflows. With
    ``cparams={"rounded": True}``, corners are trimmed and replaced by
    lightweight quadratic panels. The full MATLAB Gaussian smoother is richer,
    but the Python paths preserve the same high-level workflow and edge-data
    propagation.
    """

    if cparams is not None and not isinstance(cparams, dict):
        if edgevals is not None:
            raise TypeError("edgevals may be supplied positionally or by keyword, not both")
        edgevals = cparams
        cparams = None
    cparams = _legacy_options(cparams, "chunkerpoly cparams")
    _set_option(cparams, "ifclosed", closed)
    _set_option(cparams, "dyadic", dyadic)
    _set_option(cparams, "depth", depth)
    _set_option(cparams, "rounded", rounded)
    _set_option(cparams, "widths", widths)
    rounded = bool(cparams.get("rounded", False))

    vertices = np.asarray(verts, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2 or vertices.shape[1] < 2:
        raise ValueError("verts must have shape (dim, nverts) with dim > 1")
    dim, nv = vertices.shape
    ifclosed = bool(cparams.get("ifclosed", True))
    p = _pref_with_order(pref, order)
    p = ChunkerPref(p.nchmax, p.k, dim, max(p.nchstor, nv), p.verttol)

    if rounded:
        return _rounded_chunkerpoly(vertices, cparams, p, edgevals)
    if bool(cparams.get("dyadic", "depth" in cparams)):
        return _dyadic_chunkerpoly(vertices, cparams, p, edgevals)

    if ifclosed:
        starts = vertices
        ends = np.column_stack((vertices[:, 1:], vertices[:, 0]))
    else:
        starts = vertices[:, :-1]
        ends = vertices[:, 1:]
    nedge = starts.shape[1]

    if nedge > p.nchmax:
        raise ValueError("too many polygon edges for nchmax")
    chnkr = Chunker(p).addchunk(nedge)

    edge_data = None
    if edgevals is not None:
        edge_data = np.asarray(edgevals, dtype=float)
        if edge_data.size % nedge != 0:
            raise ValueError("number of edge values should be multiple of number of edges")
        edge_data = edge_data.reshape(edge_data.size // nedge, nedge)
        chnkr.makedatarows(edge_data.shape[0])

    t01 = (chnkr.tstor + 1.0) / 2.0
    for idx in range(nedge):
        start = starts[:, idx]
        end = ends[:, idx]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise ValueError("polygon edges must have positive length")
        tangent = delta / length
        h = length / 2.0
        chnkr.rstor[:, :, idx] = start[:, None] + delta[:, None] * t01[None, :]
        chnkr.dstor[:, :, idx] = tangent[:, None] * h
        chnkr.d2stor[:, :, idx] = 0.0
        if edge_data is not None:
            chnkr.datastor[:, :, idx] = edge_data[:, idx][:, None]

    adjs = np.zeros((2, nedge), dtype=int)
    adjs[0] = np.arange(0, nedge)
    adjs[1] = np.arange(2, nedge + 2)
    if ifclosed:
        adjs[0, 0] = nedge
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    return chnkr
