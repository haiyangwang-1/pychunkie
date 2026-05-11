"""Top-level geometry and domain helpers from MATLAB ``chunkIE``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from .chunkgraph import ChunkGraph


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


def ellipse(t: ArrayLike, a: float = 1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    dxs = (-(radius) * st - narms * amp * snt * ct) * scale
    dys = (radius * ct - narms * amp * snt * st) * scale
    d2xs = (-dys - narms * amp * (narms * cnt * ct - snt * st)) * scale
    d2ys = (dxs - narms * amp * (narms * cnt * st + snt * ct)) * scale
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
    ypp = d * expfac * (-2.0 * a * b * flat * np.cos(phase) + (a * a * flat**2 - a - b * b) * np.sin(phase))
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


@dataclass
class HypOctNode:
    ctr: np.ndarray
    xi: np.ndarray
    prnt: int | None
    chld: list[int]
    nbor: list[int]


@dataclass
class HypOctTree:
    nlvl: int
    lvp: np.ndarray
    lrt: float
    nodes: list[HypOctNode]


def hypoct_uni(
    x: ArrayLike,
    bl: float,
    lvlmax: int | float = np.inf,
    ext: ArrayLike | None = None,
) -> HypOctTree:
    """Build a uniform-depth hyperoctree over points.

    Indices in the returned tree are zero-based, following the rest of the
    Python port rather than MATLAB's one-based struct arrays.
    """

    points = np.asarray(x, dtype=float)
    if points.ndim != 2:
        raise ValueError("x must have shape (dim, n)")
    if bl < 0:
        raise ValueError("target box size must be non-negative")
    if lvlmax < 1:
        raise ValueError("maximum tree depth must be at least 1")

    dim, npt = points.shape
    if ext is None:
        if npt == 0:
            extent = np.zeros((dim, 2), dtype=float)
        else:
            extent = np.column_stack((np.min(points, axis=1), np.max(points, axis=1)))
    else:
        extent = np.asarray(ext, dtype=float)
        if extent.shape != (dim, 2):
            raise ValueError("ext must have shape (dim, 2)")

    root_len = float(np.max(extent[:, 1] - extent[:, 0])) if dim else 0.0
    root_ctr = 0.5 * (extent[:, 0] + extent[:, 1])
    nodes = [HypOctNode(root_ctr, np.arange(npt, dtype=int), None, [], [])]
    lvp = [0, 1]
    level = 1
    side = root_len
    max_level = np.inf if np.isinf(lvlmax) else int(lvlmax)

    while level < max_level:
        next_side = 0.5 * side
        if next_side <= bl:
            break
        start, stop = lvp[level - 1], lvp[level]
        before = len(nodes)
        for inode in range(start, stop):
            parent = nodes[inode]
            if parent.xi.size == 0:
                continue
            child_codes = _child_codes(points[:, parent.xi], parent.ctr)
            for code in np.unique(child_codes):
                mask = child_codes == code
                bits = ((int(code) >> np.arange(dim)) & 1).astype(float)
                child_ctr = parent.ctr + next_side * (bits - 0.5)
                child = HypOctNode(child_ctr, parent.xi[mask].copy(), inode, [], [])
                nodes.append(child)
                parent.chld.append(len(nodes) - 1)
            parent.xi = np.zeros(0, dtype=int)
        if len(nodes) == before:
            break
        level += 1
        lvp.append(len(nodes))
        side = next_side

    tree = HypOctTree(level, np.asarray(lvp, dtype=int), root_len, nodes)
    _populate_hypoct_neighbors(tree)
    return tree


def pointinregion(cgrph: ChunkGraph, rgn: list[list[int]], r0: ArrayLike) -> int:
    """Count loops in ``rgn`` that contain point ``r0``."""

    point = np.asarray(r0, dtype=float).reshape(2)
    nin = 0
    for loop in rgn:
        if len(loop) == 0:
            continue
        poly = _region_loop_points(cgrph, loop)
        if _point_in_poly(point, poly):
            nin += 1
    return nin


def regioninside(cgrph: ChunkGraph, rgn1: list[list[list[int]]], rgn2: list[list[list[int]]]) -> bool:
    """Return whether a representative point of ``rgn2`` lies inside ``rgn1``."""

    seed = _region_seed_point(cgrph, rgn2)
    for region in _interior_regions(rgn1):
        nin = pointinregion(cgrph, region, seed)
        if nin > 0 and nin % 2 == 1:
            return True
    return False


def mergeregions(
    cgrph: ChunkGraph,
    rgn1: list[list[list[int]]],
    rgn2: list[list[list[int]]],
) -> list[list[list[int]]]:
    """Merge two MATLAB-style chunkgraph region lists."""

    out = _copy_regions(rgn1)
    seed2 = _region_seed_point(cgrph, rgn2)
    for idx, region in _indexed_interior_regions(rgn1):
        nin = pointinregion(cgrph, region, seed2)
        if nin > 0 and nin % 2 == 1:
            out.extend(_copy_regions(rgn2[1:]))
            out[idx] = out[idx] + _copy_regions(rgn2[:1])[0]
            return out

    out2 = _copy_regions(rgn2)
    seed1 = _region_seed_point(cgrph, rgn1)
    for idx, region in _indexed_interior_regions(rgn2):
        nin = pointinregion(cgrph, region, seed1)
        if nin > 0 and nin % 2 == 1:
            out2.extend(_copy_regions(rgn1[1:]))
            out2[idx] = out2[idx] + _copy_regions(rgn1[:1])[0]
            return out2

    out.extend(_copy_regions(rgn2[1:]))
    if out and rgn2:
        out[0] = out[0] + _copy_regions(rgn2[:1])[0]
    return out


def _reshape_curve_outputs(
    t: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (2,) + t.shape
    return r.reshape(shape), d.reshape(shape), d2.reshape(shape)


def _child_codes(points: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    dim = points.shape[0]
    codes = np.zeros(points.shape[1], dtype=int)
    for axis in range(dim):
        codes += (points[axis] > ctr[axis]).astype(int) << axis
    return codes


def _populate_hypoct_neighbors(tree: HypOctTree) -> None:
    levels = np.zeros(len(tree.nodes), dtype=int)
    side_by_level = np.zeros(tree.nlvl, dtype=float)
    side = tree.lrt
    for lvl in range(tree.nlvl):
        levels[tree.lvp[lvl] : tree.lvp[lvl + 1]] = lvl
        side_by_level[lvl] = side
        side *= 0.5

    for lvl in range(1, tree.nlvl):
        side = side_by_level[lvl]
        start, stop = tree.lvp[lvl], tree.lvp[lvl + 1]
        for i in range(start, stop):
            node = tree.nodes[i]
            neighbors: set[int] = set()
            if node.prnt is not None:
                siblings = tree.nodes[node.prnt].chld
                neighbors.update(j for j in siblings if j != i)
                parent_neighbors = tree.nodes[node.prnt].nbor
            else:
                parent_neighbors = []

            for j in parent_neighbors:
                other = tree.nodes[j]
                if other.xi.size:
                    if _boxes_adjacent(node.ctr, side, other.ctr, side_by_level[levels[j]]):
                        neighbors.add(j)
                for child in other.chld:
                    other_child = tree.nodes[child]
                    if _boxes_adjacent(node.ctr, side, other_child.ctr, side_by_level[levels[child]]):
                        neighbors.add(child)
            node.nbor = sorted(neighbors)


def _boxes_adjacent(ctr1: np.ndarray, side1: float, ctr2: np.ndarray, side2: float) -> bool:
    return bool(np.all(np.abs(ctr1 - ctr2) <= 0.5 * (side1 + side2) + 1e-14))


def _interior_regions(regions: list[list[list[int]]]) -> list[list[list[int]]]:
    if regions and not regions[0]:
        return regions[1:]
    return regions


def _indexed_interior_regions(regions: list[list[list[int]]]) -> list[tuple[int, list[list[int]]]]:
    start = 1 if regions and not regions[0] else 0
    return [(idx, regions[idx]) for idx in range(start, len(regions))]


def _copy_regions(regions: list[list[list[int]]]) -> list[list[list[int]]]:
    return [[list(loop) for loop in region] for region in regions]


def _region_seed_point(cgrph: ChunkGraph, regions: list[list[list[int]]]) -> np.ndarray:
    for region in regions:
        for loop in region:
            if loop:
                edge, _ = _decode_edge(loop[0])
                return cgrph.verts[:, cgrph.edgesendverts[1, edge]]
    raise ValueError("region list does not contain any edges")


def _region_loop_points(cgrph: ChunkGraph, loop: list[int]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for item in loop:
        edge, reversed_edge = _decode_edge(item)
        chnkr = cgrph.echnks[edge].sort()[0]
        pts = chnkr.r.reshape(2, chnkr.npt, order="F")
        if reversed_edge:
            pts = pts[:, ::-1]
        pieces.append(pts)
    return np.hstack(pieces)


def _decode_edge(edge: int) -> tuple[int, bool]:
    edge_int = int(edge)
    if edge_int < 0:
        return -edge_int - 1, True
    return edge_int, False


def _point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = point
    xp = poly[0]
    yp = poly[1]
    inside = False
    for xa, ya, xb, yb in zip(xp, yp, np.roll(xp, -1), np.roll(yp, -1)):
        if ((ya > y) != (yb > y)) and (x < (xb - xa) * (y - ya) / (yb - ya + np.finfo(float).eps) + xa):
            inside = not inside
    return inside
