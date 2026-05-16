"""Chunkgraph-level RCIP helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._legacy import warn_legacy_options
from ._matrix import Rcompchunk
from .types import RCIPChunkGraphResult, RCIPSaved


def corner_refine(
    cg: Any, vertices: ArrayLike | None = None, depth: int = 1, stype: str = "a"
) -> Any:
    """Dyadically refine chunks adjacent to selected chunkgraph vertices."""

    out = cg.copy()
    if vertices is None:
        vinds = range(len(out.vstruc))
    else:
        vinds = np.asarray(vertices, dtype=int).reshape(-1)
    for _ in range(int(depth)):
        for ivert in vinds:
            edges, signs = out.vstruc[int(ivert)]
            for edge, sign in zip(edges, signs, strict=True):
                ch = out.echnks[int(edge)]
                ch.split(0 if sign < 0 else ch.nch - 1, stype=stype)
    return out


def chunkgraph_rcip(
    graph: Any,
    kernel: Any,
    dimension: int,
    vertices: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
    ignore_vertices: ArrayLike | None = None,
) -> RCIPChunkGraphResult:
    """Run RCIP compression at selected chunkgraph vertices.

    ``kernel`` may be a scalar kernel/callable used at every local corner or a
    global edge-by-edge block matrix. Global block matrices are restricted to
    the incident edges of each vertex before calling ``Rcompchunk``.
    """

    if not hasattr(graph, "echnks") or not hasattr(graph, "vstruc") or not hasattr(graph, "verts"):
        raise TypeError("chunkgraph_rcip expects a chunkgraph-like object")
    nvert = int(graph.verts.shape[1])
    vinds = _normalize_vertex_list(np.arange(nvert) if vertices is None else vertices, nvert)
    ignored = set(
        _normalize_vertex_list([] if ignore_vertices is None else ignore_vertices, nvert).tolist()
    )
    options = warn_legacy_options(options, "rcip.chunkgraph_rcip")

    used_vertices: list[int] = []
    edge_indices: list[np.ndarray] = []
    rmats: list[np.ndarray] = []
    saved_list: list[RCIPSaved] = []
    kernels: list[Any] = []

    for ivert in vinds:
        iv = int(ivert)
        if iv in ignored:
            continue
        edges, _ = graph.vstruc[iv]
        edges = np.asarray(edges, dtype=int).reshape(-1)
        if edges.size < 2:
            continue
        local_kernel = _select_vertex_kernel(kernel, edges)
        rmat, saved = Rcompchunk(
            graph.echnks,
            edges,
            local_kernel,
            dimension,
            graph.verts[:, iv],
            options=options,
        )
        used_vertices.append(iv)
        edge_indices.append(edges.copy())
        rmats.append(rmat)
        saved_list.append(saved)
        kernels.append(local_kernel)

    return RCIPChunkGraphResult(
        vertices=np.array(used_vertices, dtype=int),
        edge_indices=edge_indices,
        R=rmats,
        saved=saved_list,
        kernels=kernels,
    )


def _select_vertex_kernel(kernel: Any, edges: np.ndarray) -> Any:
    arr = (
        np.asarray(kernel, dtype=object) if isinstance(kernel, (list, tuple, np.ndarray)) else None
    )
    if arr is None or arr.ndim != 2:
        return kernel
    if arr.shape == (edges.size, edges.size):
        return arr
    if edges.size and (np.max(edges) >= arr.shape[0] or np.max(edges) >= arr.shape[1]):
        raise ValueError("global RCIP block kernel matrix is too small for selected vertex edges")
    return arr[np.ix_(edges, edges)]


def _normalize_vertex_list(vertices: ArrayLike, nvert: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=int).reshape(-1)
    if arr.size and np.max(arr) >= int(nvert):
        if np.min(arr) >= 1 and np.max(arr) <= int(nvert):
            arr = arr - 1
        else:
            raise ValueError("vertex index out of range")
    if np.any(arr < 0) or np.any(arr >= int(nvert)):
        raise ValueError("vertex index out of range")
    return arr
