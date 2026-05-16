"""ChunkGraph construction and public conversion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ._chunkgraph_refine import _subchunker
from ._chunkgraph_regions import _points_in_poly, _region_loop_points, _region_polygons
from .chunker import Chunker, ChunkerPref, _legacy_options, chunkerfunc

if TYPE_CHECKING:
    from .chunkgraph import ChunkGraph


def find_edge_regions(graph: ChunkGraph) -> np.ndarray:
    """Return the region on each side of every graph edge.

    The output matches MATLAB's ``find_edge_regions`` shape and one-based
    region ids. Python region loops use zero-based positive edge ids and
    ``-(edge + 1)`` for reversed edges.
    """

    edge_regs = np.zeros((2, graph.edgesendverts.shape[1]), dtype=int)
    for ireg, region in enumerate(graph.regions, start=1):
        for loop in region:
            for item in loop:
                if item >= 0:
                    edge_regs[0, int(item)] = ireg
                else:
                    edge_regs[1, -int(item) - 1] = ireg
    return edge_regs


def tochunkgraph(chunker: Chunker) -> Any:
    """Convert sorted open/closed chunker components into graph edges."""

    sorted_chunker, info = chunker.sort()
    verts: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    specs: list[Chunker] = []
    start = 0
    for nch, closed in zip(info["nchs"], info["ifclosed"], strict=False):
        sub = _subchunker(sorted_chunker, start, int(nch), bool(closed))
        rend, _ = sub.chunkends([0, sub.nch - 1] if sub.nch > 1 else [0])
        if closed:
            verts.append(rend[:, 0, 0])
            edges.append((len(verts) - 1, len(verts) - 1))
        else:
            left = rend[:, 0, 0]
            right = rend[:, 1, -1]
            verts.extend([left, right])
            edges.append((len(verts) - 2, len(verts) - 1))
        specs.append(sub)
        start += int(nch)
    from .chunkgraph import ChunkGraph

    return ChunkGraph(np.column_stack(verts), np.array(edges, dtype=int).T, specs)


def chunkgraphinregion(
    graph: ChunkGraph,
    points: ArrayLike | tuple[ArrayLike, ArrayLike] | list[ArrayLike],
) -> np.ndarray:
    """Return one-based MATLAB-style region ids for target points."""

    grid_shape = None
    if isinstance(points, (tuple, list)) and len(points) == 2:
        x = np.asarray(points[0], dtype=float)
        y = np.asarray(points[1], dtype=float)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel()))
        grid_shape = xx.shape
    else:
        arr = np.asarray(points, dtype=float)
        pts = arr.reshape(arr.shape[0], -1)

    regions = graph.regions
    if regions and not regions[0]:
        ids = np.ones(pts.shape[1], dtype=int)
        polygons = _region_polygons(graph)
        for idx, poly in enumerate(polygons, start=2):
            inside = _points_in_poly(pts, poly)
            ids[inside] = idx
    else:
        ids = np.full(pts.shape[1], np.nan)
        for idx, region in enumerate(regions, start=1):
            inside = np.zeros(pts.shape[1], dtype=bool)
            for loop in region:
                if loop:
                    inside |= _points_in_poly(pts, _region_loop_points(graph, loop))
            if idx == 1:
                ids[~inside] = idx
            else:
                ids[inside] = idx
        ids = ids.astype(int)
    return ids.reshape(grid_shape) if grid_shape is not None else ids


def _edge_count_hint(edges: np.ndarray, nverts: int) -> int:
    if edges.ndim != 2:
        raise ValueError("edgesendverts must be a 2 x nedge array or incidence matrix")
    has_nan = np.issubdtype(edges.dtype, np.floating) and np.isnan(edges).any()
    finite = edges[~np.isnan(edges)] if has_nan else edges
    if edges.shape[0] == 2 and (has_nan or np.all(finite >= 0)):
        return edges.shape[1]
    if edges.shape[1] != nverts:
        raise ValueError("incidence matrix must have one column per vertex")
    return edges.shape[0]


def _normalize_edges_with_closed_vertices(
    edges: np.ndarray,
    verts: np.ndarray,
    edge_specs: Sequence[Any],
    cparams: Sequence[dict[str, Any]] | dict[str, Any] | None,
    pref: ChunkerPref,
) -> tuple[np.ndarray, np.ndarray, dict[int, Chunker]]:
    if not (np.issubdtype(edges.dtype, np.floating) and np.isnan(edges).any()):
        return verts, _normalize_edges(edges, verts.shape[1]), {}
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("NaN closed-edge notation requires a 2 x nedge edgesendverts array")

    raw = edges.astype(float, copy=True)
    nan_cols = np.unique(np.nonzero(np.isnan(raw))[1])
    for col in nan_cols:
        if not np.all(np.isnan(raw[:, col])):
            raise ValueError("NaN closed-edge columns must have both endpoints set to NaN")

    finite = raw[~np.isnan(raw)]
    if finite.size:
        if np.max(finite) >= verts.shape[1]:
            if np.min(finite) >= 1 and np.max(finite) <= verts.shape[1]:
                raw[~np.isnan(raw)] -= 1
            else:
                raise ValueError("edge vertex index out of range")
        elif np.min(finite) < 0:
            raise ValueError("edge vertex index out of range")

    out_edges = np.zeros(raw.shape, dtype=int)
    prebuilt: dict[int, Chunker] = {}
    verts_out = verts.copy()
    for iedge in range(raw.shape[1]):
        if iedge in nan_cols:
            spec = edge_specs[iedge]
            if spec is None:
                raise ValueError(
                    "NaN closed-edge notation requires a callable or Chunker edge spec"
                )
            cp = _edge_cparams(cparams, iedge)
            cp.setdefault("ifclosed", True)
            chnkr = _edge_chunker_from_spec(spec, cp, pref)
            new_vert = verts_out.shape[1]
            verts_out = np.column_stack((verts_out, chnkr.r[:, 0, 0]))
            out_edges[:, iedge] = new_vert
            prebuilt[iedge] = chnkr
        else:
            out_edges[:, iedge] = raw[:, iedge].astype(int)
    return verts_out, out_edges, prebuilt


def _normalize_edges(edges: ArrayLike, nverts: int) -> np.ndarray:
    arr = np.asarray(edges, dtype=int)
    if arr.ndim != 2:
        raise ValueError("edgesendverts must be a 2 x nedge array or incidence matrix")
    if arr.shape[0] == 2 and np.all(arr >= 0):
        if np.max(arr) >= nverts:
            if np.min(arr) >= 1 and np.max(arr) <= nverts:
                arr = arr - 1
            else:
                raise ValueError("edge vertex index out of range")
        return arr
    if arr.shape[1] != nverts:
        raise ValueError("incidence matrix must have one column per vertex")
    out = np.zeros((2, arr.shape[0]), dtype=int)
    for iedge, row in enumerate(arr):
        starts = np.flatnonzero(row == -1)
        ends = np.flatnonzero(row == 1)
        if starts.size != 1 or ends.size != 1:
            raise ValueError("incidence edge rows must contain one -1 and one 1")
        out[:, iedge] = [starts[0], ends[0]]
    return out


def _edge_chunker_from_spec(spec: Any, cparams: dict[str, Any], pref: ChunkerPref) -> Chunker:
    if isinstance(spec, Chunker):
        return spec.copy().sort()[0]
    if callable(spec):
        chnkr, _ = chunkerfunc(spec, cparams, pref)
        return chnkr.sort()[0]
    raise TypeError("edge spec must be None, callable, or Chunker")


def _expand_edge_specs(specs: Any, nedge: int) -> list[Any]:
    if specs is None:
        return [None] * nedge
    if isinstance(specs, Chunker) or callable(specs):
        return [specs] * nedge
    out = list(specs)
    if len(out) != nedge:
        raise ValueError("fchnks must have one item per edge")
    return out


def _normalize_graph_cparams(cparams: Any) -> Any:
    if cparams is None:
        return None
    if isinstance(cparams, dict):
        return _legacy_options(cparams, "chunkgraph cparams")
    return [_legacy_options(cp, "chunkgraph cparams") for cp in cparams]


def _edge_cparams(cparams: Any, iedge: int) -> dict[str, Any]:
    if cparams is None:
        return _legacy_options(None, "chunkgraph cparams")
    if isinstance(cparams, dict):
        return _legacy_options(cparams, "chunkgraph cparams")
    return _legacy_options(cparams[iedge], "chunkgraph cparams")


def _normalize_index_list(value: Any, nitems: int) -> list[int]:
    arr = np.asarray(value, dtype=int).reshape(-1)
    out: list[int] = []
    for item in arr:
        idx = int(item)
        if idx < 0 or idx >= nitems:
            raise IndexError("graph edge index out of range")
        out.append(idx)
    return out


def _graph_splitchunks(value: Any, nedge: int) -> list[np.ndarray]:
    if value is None:
        return [np.zeros(0, dtype=int) for _ in range(nedge)]
    if (
        isinstance(value, (list, tuple))
        and len(value) == nedge
        and any(isinstance(item, (list, tuple, np.ndarray)) for item in value)
    ):
        return [np.asarray(item, dtype=int).reshape(-1) for item in value]
    chunks = np.asarray(value, dtype=int).reshape(-1)
    return [chunks.copy() for _ in range(nedge)]
