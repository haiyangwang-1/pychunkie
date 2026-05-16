"""Chunkgraph data structure for graph-like collections of chunker edges."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix, as_boundary_vector

from ._chunkgraph_build import (
    _edge_count_hint,
    _edge_cparams,
    _expand_edge_specs,
    _graph_splitchunks,
    _normalize_edges_with_closed_vertices,
    _normalize_graph_cparams,
    _normalize_index_list,
    chunkgraphinregion,
    find_edge_regions,
    tochunkgraph,
)
from ._chunkgraph_refine import _balance_graph, _fit_edge_chunker, _refine_graph_last_len
from ._chunkgraph_regions import _matlab_style_regions, _regions_to_matlab_indices
from .chunker import Chunker, ChunkerPref, _legacy_options, _set_option, chunkerfunc, merge
from .curves import linefunc

__all__ = ["ChunkGraph", "SourceInfo", "chunkgraphinregion", "find_edge_regions", "tochunkgraph"]


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
