"""Chunkgraph data structure for graph-like collections of chunker edges."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .chunker import Chunker, ChunkerPref, chunkerfunc, chunkerpoints, merge
from .chnk.curves import linefunc


@dataclass
class SourceInfo:
    r: np.ndarray
    n: np.ndarray
    d: np.ndarray
    d2: np.ndarray
    w: np.ndarray


class ChunkGraph:
    """A graph whose edges are chunkers and whose vertices mark corners."""

    __array_priority__ = 1000

    def __init__(
        self,
        verts: ArrayLike | None = None,
        edgesendverts: ArrayLike | None = None,
        fchnks: Sequence[Callable[[np.ndarray], Any] | Chunker] | Callable[[np.ndarray], Any] | Chunker | None = None,
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
        self.edgesendverts = _normalize_edges(edgesendverts, self.verts.shape[1])
        nedge = self.edgesendverts.shape[1]
        p = ChunkerPref.from_any(pref)

        edge_specs = _expand_edge_specs(fchnks, nedge)
        self.echnks = []
        for iedge in range(nedge):
            cp = _edge_cparams(cparams, iedge)
            cp.setdefault("ifclosed", False)
            cp.setdefault("nchmin", 4)
            cp.setdefault("ta", 0.0)
            cp.setdefault("tb", 1.0)
            v0 = self.verts[:, self.edgesendverts[0, iedge]]
            v1 = self.verts[:, self.edgesendverts[1, iedge]]
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
    def k(self) -> int:
        return self.echnks[0].k if self.echnks else 0

    @property
    def dim(self) -> int:
        return 2

    @property
    def datadim(self) -> int:
        return self.merged().datadim if self.echnks else 0

    @property
    def r(self) -> np.ndarray:
        return self.merged().r

    @property
    def d(self) -> np.ndarray:
        return self.merged().d

    @property
    def d2(self) -> np.ndarray:
        return self.merged().d2

    @property
    def n(self) -> np.ndarray:
        return self.merged().n

    @property
    def wts(self) -> np.ndarray:
        return self.merged().wts

    @property
    def data(self) -> np.ndarray:
        return self.merged().data

    @property
    def adj(self) -> np.ndarray:
        return self.merged().adj

    @property
    def sourceinfo(self) -> SourceInfo:
        chnkr = self.merged()
        return SourceInfo(
            r=chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F"),
            n=chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F"),
            d=chnkr.d.reshape(chnkr.dim, chnkr.npt, order="F"),
            d2=chnkr.d2.reshape(chnkr.dim, chnkr.npt, order="F"),
            w=chnkr.wts.reshape(-1, order="F"),
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
        cycles = _simple_cycles(self.edgesendverts)
        regions: list[list[list[int]]] = [[]]
        for cyc in cycles:
            regions.append([cyc])
        return regions

    def slicegraph(self, edges: ArrayLike) -> "ChunkGraph":
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

    def refine(self, opts: dict[str, Any] | None = None) -> "ChunkGraph":
        out = self.copy()
        out.echnks = [edge.refine(opts).sort()[0] for edge in out.echnks]
        out.vstruc = out.procverts()
        out.regions = out.findregions()
        return out

    def copy(self) -> "ChunkGraph":
        out = ChunkGraph()
        out.verts = self.verts.copy()
        out.edgesendverts = self.edgesendverts.copy()
        out.echnks = [edge.copy() for edge in self.echnks]
        out.v2emat = self.v2emat.copy()
        out.vstruc = [(e.copy(), s.copy()) for e, s in self.vstruc]
        out.regions = [[list(cycle) for cycle in region] for region in self.regions]
        return out

    def translate(self, vector: ArrayLike) -> "ChunkGraph":
        vec = np.asarray(vector, dtype=float).reshape(2)
        out = self.copy()
        out.verts = out.verts + vec[:, None]
        out.echnks = [edge.translate(vec) for edge in out.echnks]
        return out

    def transform(self, matrix: ArrayLike) -> "ChunkGraph":
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim == 0:
            mat = mat * np.eye(2)
        out = self.copy()
        out.verts = mat @ out.verts
        out.echnks = [mat @ edge for edge in out.echnks]
        return out

    def rotate(self, theta: float, r0: ArrayLike | None = None, r1: ArrayLike | None = None) -> "ChunkGraph":
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        out = self.copy()
        out.verts = rot @ (out.verts - center0[:, None]) + center1[:, None]
        out.echnks = [edge.rotate(theta, center0, center1) for edge in out.echnks]
        return out

    def reflect(self, theta: float, r0: ArrayLike | None = None, r1: ArrayLike | None = None) -> "ChunkGraph":
        center0 = np.zeros(2) if r0 is None else np.asarray(r0, dtype=float).reshape(2)
        center1 = np.zeros(2) if r1 is None else np.asarray(r1, dtype=float).reshape(2)
        mat = np.array([[np.cos(2.0 * theta), np.sin(2.0 * theta)], [np.sin(2.0 * theta), -np.cos(2.0 * theta)]])
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

    def flagnear(self, pts: ArrayLike, opts: dict[str, Any] | None = None) -> np.ndarray:
        return self.merged().flagnear(pts, opts)

    def flagnear_rectangle(self, pts: ArrayLike, opts: dict[str, Any] | None = None) -> np.ndarray:
        return self.merged().flagnear_rectangle(pts, opts)

    def flagnear_rectangle_grid(self, x: ArrayLike, y: ArrayLike, opts: dict[str, Any] | None = None) -> np.ndarray:
        return self.merged().flagnear_rectangle_grid(x, y, opts)

    def __add__(self, other: ArrayLike) -> "ChunkGraph":
        return self.translate(other)

    def __radd__(self, other: ArrayLike) -> "ChunkGraph":
        return self.translate(other)

    def __mul__(self, other: Any) -> "ChunkGraph":
        if np.isscalar(other):
            return self.transform(other)
        raise TypeError("product of chunkgraph and matrix only defined for matrix on left")

    def __rmul__(self, other: Any) -> "ChunkGraph":
        return self.transform(other)

    def __rmatmul__(self, other: Any) -> "ChunkGraph":
        return self.transform(other)


def chunkgraph(*args: Any, **kwargs: Any) -> ChunkGraph:
    return ChunkGraph(*args, **kwargs)


def tochunkgraph(chnkr: Chunker) -> ChunkGraph:
    sorted_chnkr, info = chnkr.sort()
    verts: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    specs: list[Chunker] = []
    start = 0
    for nch, closed in zip(info["nchs"], info["ifclosed"]):
        sub = _subchunker(sorted_chnkr, start, int(nch), bool(closed))
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


def chunkgraphinregion(cg: ChunkGraph, ptsobj: ArrayLike | tuple[ArrayLike, ArrayLike] | list[ArrayLike]) -> np.ndarray:
    grid_shape = None
    if isinstance(ptsobj, (tuple, list)) and len(ptsobj) == 2:
        x = np.asarray(ptsobj[0], dtype=float)
        y = np.asarray(ptsobj[1], dtype=float)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel()))
        grid_shape = xx.shape
    else:
        arr = np.asarray(ptsobj, dtype=float)
        pts = arr.reshape(arr.shape[0], -1)

    ids = np.ones(pts.shape[1], dtype=int)
    polygons = _region_polygons(cg)
    for idx, poly in enumerate(polygons, start=2):
        inside = _points_in_poly(pts, poly)
        ids[inside] = idx
    return ids.reshape(grid_shape) if grid_shape is not None else ids


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


def _expand_edge_specs(specs: Any, nedge: int) -> list[Any]:
    if specs is None:
        return [None] * nedge
    if isinstance(specs, Chunker) or callable(specs):
        return [specs] * nedge
    out = list(specs)
    if len(out) != nedge:
        raise ValueError("fchnks must have one item per edge")
    return out


def _edge_cparams(cparams: Any, iedge: int) -> dict[str, Any]:
    if cparams is None:
        return {}
    if isinstance(cparams, dict):
        return dict(cparams)
    return dict(cparams[iedge])


def _fit_edge_chunker(chnkr: Chunker, v0: np.ndarray, v1: np.ndarray) -> Chunker:
    rend, _ = chnkr.chunkends([0, chnkr.nch - 1] if chnkr.nch > 1 else [0])
    r0 = rend[:, 0, 0]
    r1 = rend[:, 1, -1]
    if np.linalg.norm(v1 - v0) <= 1e-14:
        return chnkr.translate(v0 - r0)
    scale = np.linalg.norm(v1 - v0) / np.linalg.norm(r1 - r0)
    theta = np.arctan2(*(v1 - v0)[::-1]) - np.arctan2(*(r1 - r0)[::-1])
    return chnkr.move(r0=r0, r1=v0, trotat=theta, scale=scale)


def _subchunker(chnkr: Chunker, start: int, nch: int, closed: bool) -> Chunker:
    sub = Chunker({"k": chnkr.k, "dim": chnkr.dim, "nchstor": nch, "nchmax": nch}).addchunk(nch)
    sl = slice(start, start + nch)
    sub.r = chnkr.r[:, :, sl]
    sub.d = chnkr.d[:, :, sl]
    sub.d2 = chnkr.d2[:, :, sl]
    sub.adj = np.vstack((np.arange(0, nch), np.arange(2, nch + 2)))
    if closed:
        sub.adj[0, 0] = nch
        sub.adj[1, -1] = 1
    else:
        sub.adj[0, 0] = -1
        sub.adj[1, -1] = -1
    sub.recompute_geometry()
    return sub


def _simple_cycles(edges: np.ndarray) -> list[list[int]]:
    unused = set(range(edges.shape[1]))
    cycles: list[list[int]] = []
    while unused:
        start_edge = min(unused)
        start_vertex = int(edges[0, start_edge])
        current_vertex = int(edges[1, start_edge])
        cycle = [start_edge]
        unused.remove(start_edge)
        while current_vertex != start_vertex:
            candidates = [edge for edge in unused if int(edges[0, edge]) == current_vertex]
            if not candidates:
                cycle = []
                break
            edge = min(candidates)
            cycle.append(edge)
            unused.remove(edge)
            current_vertex = int(edges[1, edge])
        if cycle:
            cycles.append(cycle)
    cycles.sort(key=lambda c: abs(_poly_area(_cycle_vertices(edges, c))), reverse=True)
    return cycles


def _cycle_vertices(edges: np.ndarray, cycle: list[int]) -> list[int]:
    return [int(edges[0, edge]) for edge in cycle]


def _region_polygons(cg: ChunkGraph) -> list[np.ndarray]:
    polys: list[np.ndarray] = []
    for region in cg.regions[1:]:
        if not region:
            continue
        verts = _cycle_vertices(cg.edgesendverts, [abs(edge) for edge in region[0]])
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
    for xa, ya, xb, yb in zip(xp, yp, np.roll(xp, -1), np.roll(yp, -1)):
        crosses = (ya > y) != (yb > y)
        xhit = (xb - xa) * (y - ya) / (yb - ya + np.finfo(float).eps) + xa
        inside ^= crosses & (x < xhit)
    return inside
