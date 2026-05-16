"""Region and face-cycle helpers for :class:`chunkie.geometry.ChunkGraph`."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from chunkie._layout import as_boundary_point_matrix


def _bounded_face_cycles(cg: Any) -> list[list[int]]:
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


def _oriented_face_cycles(cg: Any) -> list[list[int]]:
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


def _matlab_style_regions(cg: Any) -> list[list[list[int]]]:
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


def _find_unbounded_cycle_index(cg: Any, cycles: list[list[int]]) -> int:
    iunbounded = 0
    for idx, cycle in enumerate(cycles):
        if _cycle_turning_angle(cg, cycle) > np.pi:
            iunbounded = idx
    return iunbounded


def _cycle_turning_angle(cg: Any, cycle: list[int]) -> float:
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


def _regioninside(cg: Any, rgn1: list[list[list[int]]], rgn2: list[list[list[int]]]) -> bool:
    seed = _region_seed_point(cg, rgn2)
    for region in rgn1[1:]:
        nin = _pointinregion(cg, region, seed)
        if nin > 0 and nin % 2 == 1:
            return True
    return False


def _mergeregions(
    cg: Any,
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


def _pointinregion(cg: Any, region: list[list[int]], point: np.ndarray) -> int:
    pts = np.asarray(point, dtype=float).reshape(2, 1)
    count = 0
    for loop in region:
        if loop and _points_in_poly(pts, _region_loop_points(cg, loop))[0]:
            count += 1
    return count


def _region_seed_point(cg: Any, regions: list[list[list[int]]]) -> np.ndarray:
    for region in regions:
        for loop in region:
            if loop:
                edge, _ = _decode_region_edge(loop[0])
                return cg.verts[:, cg.edgesendverts[1, edge]]
    raise ValueError("region list does not contain any edges")


def _region_loop_points(cg: Any, loop: list[int]) -> np.ndarray:
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


def _signed_cycle_area(cg: Any, cycle: list[int]) -> float:
    verts = _signed_cycle_vertices(cg.edgesendverts, cycle)
    return _poly_area(cg.verts[:, verts])


def _unsigned_cycle(cycle: list[int]) -> list[int]:
    return [abs(edge) - 1 for edge in cycle]


def _outer_faces_first(cg: Any, cycles: list[list[int]]) -> list[list[int]]:
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


def _region_polygons(cg: Any) -> list[np.ndarray]:
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
