"""Chunkgraph data structure for graph-like collections of chunker edges."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix, as_boundary_vector

from .chunker import (
    Chunker,
    ChunkerPref,
    _legacy_options,
    _set_option,
    chunkerfunc,
    merge,
)
from .curves import linefunc


@dataclass
class SourceInfo:
    """Flattened source fields for graph-wide operator evaluation."""

    r: np.ndarray
    n: np.ndarray
    d: np.ndarray
    d2: np.ndarray
    w: np.ndarray


class ChunkGraph:
    """Region-aware collection of chunker edges joined at vertices.

    ``ChunkGraph`` is the preferred geometry container for multi-region or
    multiply connected BVPs. Each edge is a ``Chunker``; vertices encode corner
    incidence and orientation; ``regions`` records the signed edge loops found
    by graph traversal. Scalar kernels operate on the merged geometry, while
    edge-by-edge block kernel matrices can express coupled interface systems
    with different physics or operator dimensions on each edge.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        verts: ArrayLike | None = None,
        edgesendverts: ArrayLike | None = None,
        fchnks: Sequence[Callable[[np.ndarray], Any] | Chunker]
        | Callable[[np.ndarray], Any]
        | Chunker
        | None = None,
        cparams: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        pref: ChunkerPref | dict[str, Any] | None = None,
    ) -> None:
        if verts is None or edgesendverts is None:
            self.verts = np.zeros((2, 0))
            self.edgesendverts = np.zeros((2, 0), dtype=int)
            self.echnks: list[Chunker] = []
            self.v2emat = np.zeros((0, 0), dtype=int)
            self.vstruc: list[tuple[np.ndarray, np.ndarray]] = []
            self.regions: list[list[list[int]]] = []
            return

        self.verts = np.asarray(verts, dtype=float)
        if self.verts.ndim != 2 or self.verts.shape[0] != 2:
            raise ValueError("verts must have shape (2, nverts)")
        p = ChunkerPref.from_any(pref)

        raw_edges = np.asarray(edgesendverts)
        nedge_hint = _edge_count_hint(raw_edges, self.verts.shape[1])
        cparams = _normalize_graph_cparams(cparams)
        edge_specs = _expand_edge_specs(fchnks, nedge_hint)
        self.verts, self.edgesendverts, prebuilt = _normalize_edges_with_closed_vertices(
            raw_edges, self.verts, edge_specs, cparams, p
        )
        nedge = self.edgesendverts.shape[1]
        self.echnks = []
        for iedge in range(nedge):
            cp = _edge_cparams(cparams, iedge)
            cp.setdefault("ifclosed", False)
            cp.setdefault("nchmin", 4)
            cp.setdefault("ta", 0.0)
            cp.setdefault("tb", 1.0)
            v0 = self.verts[:, self.edgesendverts[0, iedge]]
            v1 = self.verts[:, self.edgesendverts[1, iedge]]
            if iedge in prebuilt:
                self.echnks.append(_fit_edge_chunker(prebuilt[iedge], v0, v1))
                continue
            spec = edge_specs[iedge]
            if spec is None:
                chnkr, _ = chunkerfunc(lambda t, a=v0, b=v1: linefunc(t, a, b), cp, p)
                self.echnks.append(chnkr.sort()[0])
                continue
            if isinstance(spec, Chunker):
                chnkr = spec.copy().sort()[0]
            elif callable(spec):
                chnkr, _ = chunkerfunc(spec, cp, p)
                chnkr = chnkr.sort()[0]
            else:
                raise TypeError("edge spec must be None, callable, or Chunker")
            self.echnks.append(_fit_edge_chunker(chnkr, v0, v1))

        self.v2emat = self.build_v2emat()
        self.vstruc = self.procverts()
        self.regions = self.findregions()

    @property
    def npt(self) -> int:
        return sum(edge.npt for edge in self.echnks)

    @property
    def point_count(self) -> int:
        return self.npt

    @property
    def k(self) -> int:
        return self.echnks[0].k if self.echnks else 0

    @property
    def quadrature_order(self) -> int:
        return self.k

    @property
    def dim(self) -> int:
        return 2

    @property
    def coordinate_dim(self) -> int:
        return self.dim

    @property
    def datadim(self) -> int:
        return self.merged().datadim if self.echnks else 0

    @property
    def r(self) -> np.ndarray:
        return self.merged().r

    @property
    def positions(self) -> np.ndarray:
        return self.r

    @property
    def d(self) -> np.ndarray:
        return self.merged().d

    @property
    def derivatives(self) -> np.ndarray:
        return self.d

    @property
    def d2(self) -> np.ndarray:
        return self.merged().d2

    @property
    def second_derivatives(self) -> np.ndarray:
        return self.d2

    @property
    def n(self) -> np.ndarray:
        return self.merged().n

    @property
    def normal_vectors(self) -> np.ndarray:
        return self.n

    @property
    def wts(self) -> np.ndarray:
        return self.merged().wts

    @property
    def quadrature_weights(self) -> np.ndarray:
        return self.wts

    @property
    def data(self) -> np.ndarray:
        return self.merged().data

    @property
    def adj(self) -> np.ndarray:
        return self.merged().adj

    @property
    def adjacency(self) -> np.ndarray:
        return self.adj

    @property
    def sourceinfo(self) -> SourceInfo:
        chnkr = self.merged()
        return SourceInfo(
            r=as_boundary_point_matrix(chnkr.r, chnkr.dim, chnkr.npt, name="positions"),
            n=as_boundary_point_matrix(chnkr.n, chnkr.dim, chnkr.npt, name="normals"),
            d=as_boundary_point_matrix(chnkr.d, chnkr.dim, chnkr.npt, name="derivatives"),
            d2=as_boundary_point_matrix(chnkr.d2, chnkr.dim, chnkr.npt, name="second derivatives"),
            w=as_boundary_vector(chnkr.wts, name="weights"),
        )

    def merged(self) -> Chunker:
        return merge(self.echnks)

    def build_v2emat(self) -> np.ndarray:
        mat = np.zeros((self.edgesendverts.shape[1], self.verts.shape[1]), dtype=int)
        for iedge, (start, end) in enumerate(self.edgesendverts.T):
            if start == end:
                mat[iedge, start] = 2
            else:
                mat[iedge, start] = -1
                mat[iedge, end] = 1
        return mat

    def procverts(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for ivert in range(self.verts.shape[1]):
            edges: list[int] = []
            signs: list[int] = []
            tangents: list[np.ndarray] = []
            for iedge, (start, end) in enumerate(self.edgesendverts.T):
                if end == ivert:
                    edges.append(iedge)
                    signs.append(1)
                    tangents.append(-self.echnks[iedge].d[:, -1, -1])
                if start == ivert:
                    edges.append(iedge)
                    signs.append(-1)
                    tangents.append(self.echnks[iedge].d[:, 0, 0])
            if edges:
                angles = np.array([np.arctan2(vec[1], vec[0]) for vec in tangents])
                order = np.argsort(angles)
                out.append((np.array(edges, dtype=int)[order], np.array(signs, dtype=int)[order]))
            else:
                out.append((np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
        return out

    def findregions(self) -> list[list[list[int]]]:
        regions = _matlab_style_regions(self)
        self._signed_regions = _regions_to_matlab_indices(regions)
        return regions

    def slicegraph(self, edges: ArrayLike) -> ChunkGraph:
        keep = np.asarray(edges, dtype=int).reshape(-1)
        old_edges = self.edgesendverts[:, keep]
        old_verts = np.unique(old_edges)
        new_verts = self.verts[:, old_verts]
        remap = {old: new for new, old in enumerate(old_verts)}
        new_edges = np.vectorize(remap.__getitem__)(old_edges)
        return ChunkGraph(new_verts, new_edges, [self.echnks[i] for i in keep])

    def edgeids(self, edges: ArrayLike) -> np.ndarray:
        keep = np.asarray(edges, dtype=int).reshape(-1)
        starts = np.cumsum([0] + [edge.npt for edge in self.echnks])
        ids: list[int] = []
        for edge in keep:
            ids.extend(range(starts[edge], starts[edge + 1]))
        return np.array(ids, dtype=int)

    def refine(
        self,
        options: dict[str, Any] | None = None,
        *,
        refine_edges: ArrayLike | None = None,
        ignore_edges: ArrayLike | None = None,
        split_chunks: Any | None = None,
        last_length: float | None = None,
        max_chunk_length: float | None = None,
        level_restrict: str | None = None,
        level_restrict_factor: float | None = None,
        oversample: int | None = None,
        split_type: str | None = None,
        max_chunks: int | None = None,
    ) -> ChunkGraph:
        options = _legacy_options(options, "chunkgraph refine options")
        _set_option(options, "dlist", refine_edges)
        _set_option(options, "ilist", ignore_edges)
        _set_option(options, "splitchunks", split_chunks)
        _set_option(options, "last_len", last_length)
        _set_option(options, "maxchunklen", max_chunk_length)
        _set_option(options, "lvlr", level_restrict)
        _set_option(options, "lvlrfac", level_restrict_factor)
        _set_option(options, "nover", oversample)
        _set_option(options, "stype", split_type)
        _set_option(options, "nchmax", max_chunks)
        out = self.copy()
        nedge = len(out.echnks)
        dlist = _normalize_index_list(options.get("dlist", np.arange(nedge)), nedge)
        ilist = set(_normalize_index_list(options.get("ilist", []), nedge))
        splitchunks = _graph_splitchunks(options.get("splitchunks", []), nedge)
        edge_options = {
            key: val
            for key, val in options.items()
            if key not in {"dlist", "ilist", "splitchunks", "last_len"}
        }
        for iedge in dlist:
            if iedge in ilist:
                continue
            edge_options_for_refine = dict(edge_options)
            edge_options_for_refine["splitchunks"] = splitchunks[iedge]
            out.echnks[iedge] = out.echnks[iedge].refine(edge_options_for_refine).sort()[0]
        _balance_graph(out)
        out.vstruc = out.procverts()
        out.regions = out.findregions()
        if "last_len" in options and options["last_len"] not in (None, ""):
            _refine_graph_last_len(out, float(options["last_len"]), edge_options)
            _balance_graph(out)
            out.vstruc = out.procverts()
            out.regions = out.findregions()
        return out

    def copy(self) -> ChunkGraph:
        out = ChunkGraph()
        out.verts = self.verts.copy()
        out.edgesendverts = self.edgesendverts.copy()
        out.echnks = [edge.copy() for edge in self.echnks]
        out.v2emat = self.v2emat.copy()
        out.vstruc = [(e.copy(), s.copy()) for e, s in self.vstruc]
        out.regions = [[list(cycle) for cycle in region] for region in self.regions]
        out._signed_regions = [
            [list(cycle) for cycle in region]
            for region in getattr(self, "_signed_regions", out.regions)
        ]
        return out

    def translate(self, vector: ArrayLike) -> ChunkGraph:
        vec = np.asarray(vector, dtype=float).reshape(2)
        out = self.copy()
        out.verts = out.verts + vec[:, None]
        out.echnks = [edge.translate(vec) for edge in out.echnks]
        return out

    def transform(self, matrix: ArrayLike) -> ChunkGraph:
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim == 0:
            mat = mat * np.eye(2)
        out = self.copy()
        out.verts = mat @ out.verts
        out.echnks = [mat @ edge for edge in out.echnks]
        return out

    def rotate(
        self, theta: float, r0: ArrayLike | None = None, r1: ArrayLike | None = None
    ) -> ChunkGraph:
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        out = self.copy()
        out.verts = rot @ (out.verts - center0[:, None]) + center1[:, None]
        out.echnks = [edge.rotate(theta, center0, center1) for edge in out.echnks]
        return out

    def reflect(
        self, theta: float, r0: ArrayLike | None = None, r1: ArrayLike | None = None
    ) -> ChunkGraph:
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        mat = np.array(
            [
                [np.cos(2.0 * theta), np.sin(2.0 * theta)],
                [np.sin(2.0 * theta), -np.cos(2.0 * theta)],
            ]
        )
        out = self.copy()
        out.verts = mat @ (out.verts - center0[:, None]) + center1[:, None]
        out.echnks = [edge.reflect(theta, center0, center1) for edge in out.echnks]
        return out

    def min(self) -> np.ndarray:
        return self.merged().min()

    def max(self) -> np.ndarray:
        return self.merged().max()

    def onesmat(self) -> np.ndarray:
        return self.merged().onesmat()

    def normonesmat(self) -> np.ndarray:
        return self.merged().normonesmat()

    def flagnear(
        self,
        points: ArrayLike,
        options: dict[str, Any] | None = None,
        *,
        fac: float | None = None,
    ) -> np.ndarray:
        options = _legacy_options(options, "chunkgraph flagnear options")
        _set_option(options, "fac", fac)
        return self.merged().flagnear(points, options)

    def flagnear_rectangle(
        self,
        points: ArrayLike,
        options: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        options = _legacy_options(options, "chunkgraph flagnear_rectangle options")
        _set_option(options, "rho", rho)
        return self.merged().flagnear_rectangle(points, options)

    def flagnear_rectangle_grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        options: dict[str, Any] | None = None,
        *,
        rho: float | None = None,
    ) -> np.ndarray:
        options = _legacy_options(options, "chunkgraph flagnear_rectangle_grid options")
        _set_option(options, "rho", rho)
        return self.merged().flagnear_rectangle_grid(x, y, options)

    def __add__(self, other: ArrayLike) -> ChunkGraph:
        return self.translate(other)

    def __radd__(self, other: ArrayLike) -> ChunkGraph:
        return self.translate(other)

    def __mul__(self, other: Any) -> ChunkGraph:
        if np.isscalar(other):
            return self.transform(other)
        raise TypeError("product of chunkgraph and matrix only defined for matrix on left")

    def __rmul__(self, other: Any) -> ChunkGraph:
        return self.transform(other)

    def __rmatmul__(self, other: Any) -> ChunkGraph:
        return self.transform(other)


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


def tochunkgraph(chunker: Chunker) -> ChunkGraph:
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


def _refine_graph_last_len(graph: ChunkGraph, last_len: float, options: dict[str, Any]) -> None:
    if last_len <= 0.0:
        raise ValueError("last_len must be positive")
    stype = str(options.get("stype", "a"))
    tol = 1e-10 * max(last_len, 1.0)
    for _ in range(int(options.get("maxiter_last_len", 20))):
        changed = False
        graph.vstruc = graph.procverts()
        for edges, signs in graph.vstruc:
            if edges.size == 0:
                continue
            endpoint_info: list[tuple[int, int, int, float]] = []
            for edge, sign in zip(edges, signs, strict=True):
                iedge = int(edge)
                isign = int(sign)
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                length = float(graph.echnks[iedge].chunklen([ichunk])[0])
                endpoint_info.append((iedge, isign, ichunk, length))
            target = _last_len_target([item[3] for item in endpoint_info], last_len)
            for iedge, isign, ichunk, length in endpoint_info:
                if abs(length - target) <= tol:
                    continue
                ratio = target / length
                if ratio <= 1e-8 or ratio >= 1.0 - 1e-8:
                    continue
                frac = ratio if isign < 0 else 1.0 - ratio
                graph.echnks[iedge] = (
                    graph.echnks[iedge].split(ichunk, frac=frac, stype=stype).sort()[0]
                )
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("graph last_len refinement did not converge")
    graph.vstruc = graph.procverts()
    graph.regions = graph.findregions()


def _balance_graph(graph: ChunkGraph) -> None:
    for _ in range(1000):
        changed = False
        graph.vstruc = graph.procverts()
        for edges, signs in graph.vstruc:
            if edges.size == 0:
                continue
            endpoint_info: list[tuple[int, int, float]] = []
            for edge, sign in zip(edges, signs, strict=True):
                iedge = int(edge)
                isign = int(sign)
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                length = float(graph.echnks[iedge].chunklen([ichunk])[0])
                endpoint_info.append((iedge, isign, length))
            lengths = np.asarray([item[2] for item in endpoint_info], dtype=float)
            amin = float(np.min(lengths))
            amax_idx = int(np.argmax(lengths))
            if amin <= 0.0:
                continue
            nsplit = int(np.floor(np.log2(float(lengths[amax_idx]) / amin)))
            if nsplit <= 0:
                continue
            iedge, isign, _ = endpoint_info[amax_idx]
            for _ in range(nsplit):
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                graph.echnks[iedge] = graph.echnks[iedge].split(ichunk).sort()[0]
            changed = True
        if not changed:
            graph.vstruc = graph.procverts()
            graph.regions = graph.findregions()
            return
    raise RuntimeError("graph balance did not converge")


def _last_len_target(lengths: Sequence[float], last_len: float) -> float:
    lens = np.asarray(lengths, dtype=float)
    level = -np.log2(float(np.min(lens)) / last_len)
    if level - np.round(level) < 1e-8:
        level = float(np.round(level))
    else:
        level = float(np.ceil(level))
    level = max(level, 1.0)
    return float(last_len * 2.0 ** (-level))


def _fit_edge_chunker(chunker: Chunker, v0: np.ndarray, v1: np.ndarray) -> Chunker:
    rend, _ = chunker.chunkends([0, chunker.nch - 1] if chunker.nch > 1 else [0])
    r0 = rend[:, 0, 0]
    r1 = rend[:, 1, -1]
    if np.linalg.norm(v1 - v0) <= 1e-14:
        return chunker.translate(v0 - r0)
    scale = np.linalg.norm(v1 - v0) / np.linalg.norm(r1 - r0)
    theta = np.arctan2(*(v1 - v0)[::-1]) - np.arctan2(*(r1 - r0)[::-1])
    return chunker.move(r0=r0, r1=v0, trotat=theta, scale=scale)


def _subchunker(chunker: Chunker, start: int, chunk_count: int, closed: bool) -> Chunker:
    sub = Chunker(
        {"k": chunker.k, "dim": chunker.dim, "nchstor": chunk_count, "nchmax": chunk_count}
    ).addchunk(chunk_count)
    sl = slice(start, start + chunk_count)
    sub.r = chunker.r[:, :, sl]
    sub.d = chunker.d[:, :, sl]
    sub.d2 = chunker.d2[:, :, sl]
    sub.adj = np.vstack((np.arange(0, chunk_count), np.arange(2, chunk_count + 2)))
    if closed:
        sub.adj[0, 0] = chunk_count
        sub.adj[1, -1] = 1
    else:
        sub.adj[0, 0] = -1
        sub.adj[1, -1] = -1
    sub.recompute_geometry()
    return sub


def _bounded_face_cycles(cg: ChunkGraph) -> list[list[int]]:
    cycles = _oriented_face_cycles(cg)
    by_component: dict[int, list[list[int]]] = {}
    comp = _edge_components(cg.edgesendverts, cg.verts.shape[1])
    for cycle in cycles:
        by_component.setdefault(comp[abs(cycle[0]) - 1], []).append(cycle)

    bounded: list[list[int]] = []
    for comp_cycles in by_component.values():
        if len(comp_cycles) == 2 and {abs(edge) for edge in comp_cycles[0]} == {
            abs(edge) for edge in comp_cycles[1]
        }:
            chosen = max(comp_cycles, key=lambda cyc: _signed_cycle_area(cg, cyc))
            bounded.append(chosen)
            continue
        unbounded = max(
            range(len(comp_cycles)), key=lambda idx: abs(_signed_cycle_area(cg, comp_cycles[idx]))
        )
        for idx, cycle in enumerate(comp_cycles):
            if idx != unbounded:
                bounded.append(cycle)
    return _outer_faces_first(cg, bounded)


def _oriented_face_cycles(cg: ChunkGraph) -> list[list[int]]:
    nedge = cg.edgesendverts.shape[1]
    remaining = list(range(1, nedge + 1)) + list(range(-1, -nedge - 1, -1))
    cycles: list[list[int]] = []
    while remaining:
        current = remaining.pop(0)
        start = current
        cycle = [current]
        vertex = _oriented_edge_end(cg.edgesendverts, current)
        for _ in range(2 * nedge + 1):
            edges, signs = cg.vstruc[vertex]
            loc = np.flatnonzero((edges == abs(current) - 1) & (signs == np.sign(current)))
            if loc.size == 0:
                cycle = []
                break
            next_idx = (int(loc[0]) + 1) % edges.size
            next_edge = int(edges[next_idx]) + 1
            next_sign = int(signs[next_idx])
            current = -next_sign * next_edge
            if current == start:
                break
            cycle.append(current)
            if current in remaining:
                remaining.remove(current)
            vertex = _oriented_edge_end(cg.edgesendverts, current)
        else:
            cycle = []
        if cycle:
            cycles.append(cycle)
    return cycles


def _oriented_edge_end(edges: np.ndarray, signed_edge: int) -> int:
    edge = abs(signed_edge) - 1
    return int(edges[1, edge] if signed_edge > 0 else edges[0, edge])


def _edge_components(edges: np.ndarray, nverts: int) -> list[int]:
    parent = list(range(nverts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for start, end in edges.T:
        union(int(start), int(end))
    roots = {root: idx for idx, root in enumerate(sorted({find(i) for i in range(nverts)}))}
    return [roots[find(int(start))] for start in edges[0]]


def _matlab_style_regions(cg: ChunkGraph) -> list[list[list[int]]]:
    cycles = _oriented_face_cycles(cg)
    if not cycles:
        return []

    edge_components = _edge_components(cg.edgesendverts, cg.verts.shape[1])
    component_regions: list[list[list[list[int]]]] = []
    for comp in sorted({edge_components[abs(cycle[0]) - 1] for cycle in cycles}):
        comp_cycles = [cycle for cycle in cycles if edge_components[abs(cycle[0]) - 1] == comp]
        if not comp_cycles:
            continue
        regions = [[_to_python_signed_cycle(cycle)] for cycle in comp_cycles]
        iunbounded = _find_unbounded_cycle_index(cg, comp_cycles)
        regions[0], regions[iunbounded] = regions[iunbounded], regions[0]
        component_regions.append(regions)

    if not component_regions:
        return []

    containment = np.zeros((len(component_regions), len(component_regions)), dtype=bool)
    for ireg, region in enumerate(component_regions):
        containing: list[int] = []
        for jreg, candidate in enumerate(component_regions):
            if ireg != jreg and _regioninside(cg, candidate, region):
                containing.append(jreg)
        for jreg in containing:
            containment[ireg, jreg] = True
            containment[jreg, ireg] = True

    labels = _component_labels_from_adjacency(containment)
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    component_regions = [component_regions[int(idx)] for idx in order]

    for ireg in range(labels.size):
        label = labels[ireg]
        for jreg in range(labels.size - 1):
            if label == labels[jreg] and _regioninside(
                cg, component_regions[jreg], component_regions[jreg + 1]
            ):
                component_regions[jreg], component_regions[jreg + 1] = (
                    component_regions[jreg + 1],
                    component_regions[jreg],
                )

    grouped: list[list[list[int]]] = []
    for label in sorted(set(int(item) for item in labels)):
        indices = [idx for idx, item in enumerate(labels) if int(item) == label]
        merged = component_regions[indices[0]]
        for idx in indices[1:]:
            merged = _mergeregions(cg, merged, component_regions[idx])
        grouped.append(merged)

    regions = grouped[0]
    for region in grouped[1:]:
        regions = _mergeregions(cg, regions, region)
    return regions


def _find_unbounded_cycle_index(cg: ChunkGraph, cycles: list[list[int]]) -> int:
    iunbounded = 0
    for idx, cycle in enumerate(cycles):
        if _cycle_turning_angle(cg, cycle) > np.pi:
            iunbounded = idx
    return iunbounded


def _cycle_turning_angle(cg: ChunkGraph, cycle: list[int]) -> float:
    theta = 0.0
    tangents: list[np.ndarray] = []
    for signed_edge in cycle:
        edge = abs(signed_edge) - 1
        echnk = cg.echnks[edge]
        start_angles = np.arctan2(echnk.d[1, 0, :], echnk.d[0, 0, :])
        end_angles = np.arctan2(echnk.d[1, -1, :], echnk.d[0, -1, :])
        diffs = end_angles - start_angles
        diffs = np.where(diffs > np.pi, diffs - 2.0 * np.pi, diffs)
        diffs = np.where(diffs < -np.pi, diffs + 2.0 * np.pi, diffs)
        theta += float(np.sign(signed_edge) * np.sum(diffs))

        tangent_start = echnk.d[:, 0, 0]
        tangent_end = echnk.d[:, -1, -1]
        if signed_edge > 0:
            tangents.append(np.concatenate((tangent_start, tangent_end)))
        else:
            tangents.append(np.concatenate((-tangent_end, -tangent_start)))

    angle_sum = 0.0
    for current, next_item in zip(tangents, tangents[1:] + tangents[:1], strict=False):
        tail = current[2:4]
        head = next_item[0:2]
        angle_diff = np.arctan2(head[1], head[0]) - np.arctan2(tail[1], tail[0])
        if angle_diff < -np.pi:
            angle_diff += 2.0 * np.pi
        if angle_diff >= np.pi:
            angle_diff -= 2.0 * np.pi
        angle_sum += float(angle_diff)
    return angle_sum + theta


def _to_python_signed_cycle(cycle: list[int]) -> list[int]:
    return [edge - 1 if edge > 0 else edge for edge in cycle]


def _regions_to_matlab_indices(regions: list[list[list[int]]]) -> list[list[list[int]]]:
    return [
        [[_python_region_edge_to_matlab(edge) for edge in loop] for loop in region]
        for region in regions
    ]


def _python_region_edge_to_matlab(edge: int) -> int:
    return edge + 1 if edge >= 0 else edge


def _component_labels_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    nitems = adjacency.shape[0]
    labels = np.zeros(nitems, dtype=int)
    label = 0
    for start in range(nitems):
        if labels[start] != 0:
            continue
        label += 1
        queue: deque[int] = deque([start])
        labels[start] = label
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[current]):
                idx = int(neighbor)
                if labels[idx] == 0:
                    labels[idx] = label
                    queue.append(idx)
    return labels


def _regioninside(cg: ChunkGraph, rgn1: list[list[list[int]]], rgn2: list[list[list[int]]]) -> bool:
    seed = _region_seed_point(cg, rgn2)
    for region in rgn1[1:]:
        nin = _pointinregion(cg, region, seed)
        if nin > 0 and nin % 2 == 1:
            return True
    return False


def _mergeregions(
    cg: ChunkGraph,
    rgn1: list[list[list[int]]],
    rgn2: list[list[list[int]]],
) -> list[list[list[int]]]:
    seed2 = _region_seed_point(cg, rgn2)
    containing = 0
    for idx in range(1, len(rgn1)):
        nin = _pointinregion(cg, rgn1[idx], seed2)
        if nin > 0 and nin % 2 == 1:
            containing = idx
    if containing != 0:
        out = _copy_regions(rgn1)
        out.extend(_copy_regions(rgn2[1:]))
        out[containing].extend(_copy_regions(rgn2[:1])[0])
        return out

    seed1 = _region_seed_point(cg, rgn1)
    containing = 0
    for idx in range(1, len(rgn2)):
        nin = _pointinregion(cg, rgn2[idx], seed1)
        if nin > 0 and nin % 2 == 1:
            containing = idx
    if containing != 0:
        out = _copy_regions(rgn2)
        out.extend(_copy_regions(rgn1[1:]))
        out[containing].extend(_copy_regions(rgn1[:1])[0])
        return out

    out = _copy_regions(rgn1)
    out.extend(_copy_regions(rgn2[1:]))
    out[0].extend(_copy_regions(rgn2[:1])[0])
    return out


def _pointinregion(cg: ChunkGraph, region: list[list[int]], point: np.ndarray) -> int:
    pts = np.asarray(point, dtype=float).reshape(2, 1)
    count = 0
    for loop in region:
        if loop and _points_in_poly(pts, _region_loop_points(cg, loop))[0]:
            count += 1
    return count


def _region_seed_point(cg: ChunkGraph, regions: list[list[list[int]]]) -> np.ndarray:
    for region in regions:
        for loop in region:
            if loop:
                edge, _ = _decode_region_edge(loop[0])
                return cg.verts[:, cg.edgesendverts[1, edge]]
    raise ValueError("region list does not contain any edges")


def _region_loop_points(cg: ChunkGraph, loop: list[int]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for item in loop:
        edge, reversed_edge = _decode_region_edge(item)
        echnk = cg.echnks[edge].sort()[0]
        pts = as_boundary_point_matrix(echnk.r, 2, echnk.npt, name="edge positions")
        if reversed_edge:
            pts = pts[:, ::-1]
        pieces.append(pts)
    return np.hstack(pieces)


def _decode_region_edge(edge: int) -> tuple[int, bool]:
    edge_int = int(edge)
    if edge_int < 0:
        return -edge_int - 1, True
    return edge_int, False


def _copy_regions(regions: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[list(loop) for loop in region] for region in regions]


def _signed_cycle_vertices(edges: np.ndarray, cycle: list[int]) -> list[int]:
    verts: list[int] = []
    for signed_edge in cycle:
        edge = abs(signed_edge) - 1
        verts.append(int(edges[0, edge] if signed_edge > 0 else edges[1, edge]))
    return verts


def _signed_cycle_area(cg: ChunkGraph, cycle: list[int]) -> float:
    verts = _signed_cycle_vertices(cg.edgesendverts, cycle)
    return _poly_area(cg.verts[:, verts])


def _unsigned_cycle(cycle: list[int]) -> list[int]:
    return [abs(edge) - 1 for edge in cycle]


def _outer_faces_first(cg: ChunkGraph, cycles: list[list[int]]) -> list[list[int]]:
    if len(cycles) < 2:
        return cycles
    polys = [cg.verts[:, _signed_cycle_vertices(cg.edgesendverts, cycle)] for cycle in cycles]
    depths: list[int] = []
    for ipoly, poly in enumerate(polys):
        depth = 0
        for jpoly, other in enumerate(polys):
            if ipoly == jpoly:
                continue
            if np.all(_points_in_poly(poly, other)):
                depth += 1
        depths.append(depth)
    return [
        cycle for _, cycle in sorted(enumerate(cycles), key=lambda item: (depths[item[0]], item[0]))
    ]


def _cycle_vertices(edges: np.ndarray, cycle: list[int]) -> list[int]:
    return [int(edges[0, edge]) for edge in cycle]


def _region_polygons(cg: ChunkGraph) -> list[np.ndarray]:
    polys: list[np.ndarray] = []
    regions = getattr(cg, "_signed_regions", None)
    signed = regions is not None
    if regions is None:
        regions = cg.regions
    for region in regions[1:]:
        if not region:
            continue
        cycle = region[0]
        if signed:
            verts = _signed_cycle_vertices(cg.edgesendverts, cycle)
        else:
            verts = _cycle_vertices(cg.edgesendverts, [abs(edge) for edge in cycle])
        polys.append(cg.verts[:, verts])
    return polys


def _poly_area(poly: Sequence[int] | np.ndarray) -> float:
    if isinstance(poly, np.ndarray):
        x = poly[0]
        y = poly[1]
    else:
        return 0.0
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _points_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x = pts[0]
    y = pts[1]
    xp = poly[0]
    yp = poly[1]
    inside = np.zeros(pts.shape[1], dtype=bool)
    for xa, ya, xb, yb in zip(xp, yp, np.roll(xp, -1), np.roll(yp, -1), strict=False):
        crosses = (ya > y) != (yb > y)
        xhit = (xb - xa) * (y - ya) / (yb - ya + np.finfo(float).eps) + xa
        inside ^= crosses & (x < xhit)
    return inside
