"""Nonsmooth-corner RCIP system integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import Chunker, ChunkGraph
from chunkie.geometry.chunker import right_normals
from chunkie.geometry.points import PointInfoView
from chunkie.kernels.base import flat_normals, flat_positions
from chunkie.quadrature import apply_panel_potential, dense_panel_operator_matrix
from chunkie.rcip import (
    RCIPCornerState,
    RCIPSaved,
    RCIPState,
    build_block_prolongation,
    build_local_corner_geometry,
    build_split_panel_prolongation,
)
from chunkie.rcip.algebra import schur_banachiewicz, setup

from .config import SystemConfig
from .trace import BoundaryTrace


@dataclass(frozen=True)
class _RCIPTraceInfo:
    equation_name: str
    unknown_name: str
    boundary: object
    trace: BoundaryTrace


@dataclass(frozen=True)
class _TargetView:
    positions: NDArray[np.floating]
    normals: NDArray[np.floating] | None = None

    @property
    def flat_positions(self) -> NDArray[np.floating]:
        return self.positions

    @property
    def flat_normals(self) -> NDArray[np.floating]:
        if self.normals is None:
            raise ValueError("target normals are required for this kernel selector")
        return self.normals


def build_rcip_state(system, *, config: SystemConfig | None = None) -> RCIPState:
    """Build old-style RCIP corner state for graph-backed second-kind systems."""

    active_config = SystemConfig() if config is None else config
    graph = system.geometry
    if not isinstance(graph, ChunkGraph):
        return RCIPState(metadata={"active": False, "reason": "geometry is not a ChunkGraph"})
    if not graph.edges:
        return RCIPState(metadata={"active": False, "reason": "graph has no edges"})

    trace_template = _first_trace_term(system)
    trace_info = _rcip_trace_info(system)
    nodes = graph.edges[0].chunker._legendre_nodes
    weights = graph.edges[0].chunker._legendre_weights
    _, _, interpolation, weighted_transfer = build_split_panel_prolongation(nodes, weights)

    corners: list[RCIPCornerState] = []
    for vertex in graph.vertices:
        edge_ids, signs = _corner_edges_and_signs(graph, vertex.id)
        if edge_ids.size < 2:
            continue
        directions = _corner_directions(graph, vertex.id, edge_ids, signs)
        local_geometry = build_local_corner_geometry(
            vertex.position,
            directions,
            quadrature_order=graph.quadrature_order,
            levels=max(1, int(active_config.rcip_subdivisions)),
        )
        block_prolongation = build_block_prolongation(
            interpolation,
            edge_count=edge_ids.size,
            component_count=1,
        )
        weighted_block = build_block_prolongation(
            weighted_transfer,
            edge_count=edge_ids.size,
            component_count=1,
        )
        star_size = edge_ids.size * graph.quadrature_order
        compressed_inverse: NDArray[np.generic] = np.eye(star_size)
        star_indices = None
        saved = None
        if trace_info is not None:
            compressed_inverse, saved, star_indices = _compress_corner(
                graph,
                trace_info.trace,
                vertex_id=vertex.id,
                edge_ids=edge_ids,
                signs=signs,
                config=active_config,
            )
        corners.append(
            RCIPCornerState(
                vertex_id=vertex.id,
                edge_ids=tuple(int(edge_id) for edge_id in edge_ids),
                boundary_part=graph.boundary_part(edge_ids),
                local_geometry=local_geometry,
                prolongation=block_prolongation,
                weighted_prolongation=weighted_block,
                compressed_inverse=compressed_inverse,
                local_operator=(
                    _local_corner_trace_operator(local_geometry, trace_template)
                    if trace_template is not None
                    else None
                ),
                star_indices=star_indices,
                saved=saved,
            )
        )

    return RCIPState(
        corners=tuple(corners),
        metadata={
            "active": bool(corners),
            "matrix_replacement_active": trace_info is not None
            and any(corner.saved is not None for corner in corners),
            "subdivisions": int(active_config.rcip_subdivisions),
            "quadrature_order": graph.quadrature_order,
            "density": None if trace_info is None else trace_info.unknown_name,
            "boundary": None if trace_info is None else trace_info.boundary,
            "equation": None if trace_info is None else trace_info.equation_name,
        },
    )


def apply_rcip_to_matrix(
    system,
    matrix: NDArray[np.generic],
    *,
    row_slices: dict[str, slice],
    column_slices: dict[str, slice],
    config: SystemConfig,
) -> tuple[NDArray[np.generic], RCIPState]:
    """Replace coarse corner star blocks with old RCIP-compressed blocks."""

    state = build_rcip_state(system, config=config)
    if not state.metadata.get("matrix_replacement_active"):
        return matrix, state

    equation_name = state.metadata["equation"]
    density_name = state.metadata["density"]
    row_slice = row_slices[equation_name]
    column_slice = column_slices[density_name]
    out = np.array(matrix, copy=True)
    for corner in state.corners:
        if corner.star_indices is None or corner.saved is None:
            continue
        star = np.asarray(corner.star_indices, dtype=np.int64)
        replacement = np.linalg.inv(corner.compressed_inverse)
        rows = row_slice.start + star
        columns = column_slice.start + star
        out[np.ix_(rows, columns)] = replacement
    return out, state


def evaluate_rcip_layer(layer, density, targets, *, config: SystemConfig, state: RCIPState):
    """Evaluate a layer potential with the old coarse-plus-local RCIP split."""

    if not state.metadata.get("matrix_replacement_active"):
        return None
    if layer.density != state.metadata.get("density"):
        return None
    if layer.source is not state.metadata.get("boundary"):
        return None
    if layer.kernel.input_dim != 1:
        return None

    source = layer.source.pointinfo
    values = density.component_values
    if values.shape[0] != 1:
        return None
    coarse_density = values.swapaxes(1, 2).reshape(-1).copy()
    for corner in state.corners:
        if corner.star_indices is not None:
            coarse_density[np.asarray(corner.star_indices, dtype=np.int64)] = 0.0

    contribution = layer.coefficient * apply_panel_potential(
        source,
        targets,
        layer.kernel,
        _vector_to_panel_density(coarse_density, source),
        close_correction=config.close_correction,
        near_rho=config.near_rho,
        tolerance=config.tolerance,
    )

    eval_depth = config.rcip_eval_depth
    for corner in state.corners:
        if corner.saved is None or corner.star_indices is None:
            continue
        saved = corner.saved
        density_star = values.swapaxes(1, 2).reshape(-1)[
            np.asarray(corner.star_indices, dtype=np.int64)
        ]
        local_densities, local_sources = _interpolate_saved_density(
            density_star,
            saved,
            depth=saved.saved_depth if eval_depth is None else int(eval_depth),
        )
        shifted_targets = _shift_targets(targets, saved.center)
        for local_density, local_source in zip(local_densities, local_sources, strict=True):
            contribution = contribution + layer.coefficient * apply_panel_potential(
                local_source,
                shifted_targets,
                layer.kernel,
                _vector_to_panel_density(local_density, local_source),
                close_correction=config.close_correction,
                near_rho=config.near_rho,
                tolerance=config.tolerance,
            )
    return contribution


def _rcip_trace_info(system) -> _RCIPTraceInfo | None:
    graph = system.geometry
    if not isinstance(graph, ChunkGraph):
        return None
    if len(system.unknowns) != 1 or len(system.equations) != 1 or system.constraints:
        return None
    unknown = system.unknowns[0]
    if unknown.component_count != 1:
        return None
    equation = system.equations[0]
    if len(equation.terms) != 1 or not isinstance(equation.terms[0], BoundaryTrace):
        return None
    trace = equation.terms[0]
    if trace.jump is None or trace.jump.density != unknown.name:
        return None
    if not np.isclose(trace.jump.coefficient, 1.0):
        return None
    if trace.layer.density != unknown.name:
        return None
    if trace.layer.source is not unknown.geometry or trace.target is not unknown.geometry:
        return None
    if not _is_rcip_second_kind_kernel(trace.layer.kernel):
        return None
    boundary = unknown.geometry
    if not hasattr(boundary, "edges") or not hasattr(boundary, "orientation"):
        return None
    if tuple(int(edge.id) for edge in graph.edges) != tuple(int(edge) for edge in boundary.edges):
        return None
    if np.any(np.asarray(boundary.orientation) != 1):
        return None
    return _RCIPTraceInfo(
        equation_name=equation.name,
        unknown_name=unknown.name,
        boundary=boundary,
        trace=trace,
    )


def _is_rcip_second_kind_kernel(kernel) -> bool:
    return kernel.family in {"laplace", "helmholtz"} and kernel.selector in {"d", "sp"}


def _corner_edges_and_signs(
    graph: ChunkGraph,
    vertex_id: int,
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    edges: list[int] = []
    signs: list[int] = []
    tangents: list[NDArray[np.floating]] = []
    for edge in graph.edges:
        if edge.end_vertex == vertex_id:
            edges.append(edge.id)
            signs.append(1)
            tangents.append(-edge.chunker.derivatives[:, -1, -1])
        if edge.start_vertex == vertex_id:
            edges.append(edge.id)
            signs.append(-1)
            tangents.append(edge.chunker.derivatives[:, 0, 0])
    if not edges:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    angles = np.asarray([np.arctan2(tangent[1], tangent[0]) for tangent in tangents])
    order = np.argsort(angles)
    return np.asarray(edges, dtype=np.int64)[order], np.asarray(signs, dtype=np.int64)[order]


def _corner_directions(
    graph: ChunkGraph,
    vertex_id: int,
    edge_ids: NDArray[np.integer],
    signs: NDArray[np.integer],
) -> NDArray[np.floating]:
    directions = []
    vertex_position = graph.vertices[vertex_id].position
    for edge_id, sign in zip(edge_ids, signs, strict=True):
        edge = graph.edge(int(edge_id))
        other_vertex = edge.end_vertex if int(sign) < 0 else edge.start_vertex
        directions.append(graph.vertices[other_vertex].position - vertex_position)
    return np.column_stack(directions)


def _compress_corner(
    graph: ChunkGraph,
    trace: BoundaryTrace,
    *,
    vertex_id: int,
    edge_ids: NDArray[np.integer],
    signs: NDArray[np.integer],
    config: SystemConfig,
) -> tuple[NDArray[np.generic], RCIPSaved, NDArray[np.integer]]:
    quadrature_order = graph.quadrature_order
    dimension = 1
    edge_count = edge_ids.size
    starts_at_corner = signs < 0
    (
        prolongation,
        weighted_prolongation,
        star_l,
        circ_l,
        star_s,
        circ_s,
        ilist,
        star_l_scalar,
        circ_l_scalar,
    ) = setup(quadrature_order, dimension, edge_count, starts_at_corner)

    subdivisions = max(1, int(config.rcip_subdivisions))
    saved_depth = subdivisions
    system_size = 3 * quadrature_order * edge_count * dimension
    saved_inverses: list[NDArray[np.generic] | None] = [None] * (subdivisions + 1)
    saved_blocks: list[NDArray[np.generic] | None] = [None] * subdivisions
    saved_geometries: list[Chunker | None] = [None] * subdivisions
    compressed_inverse: NDArray[np.generic] | None = None

    for level in range(1, subdivisions + 1):
        local_edges = [
            _local_edge_chunker(
                graph,
                int(edge_id),
                int(sign),
                level=level,
                subdivisions=subdivisions,
                vertex_id=vertex_id,
            )
            for edge_id, sign in zip(edge_ids, signs, strict=True)
        ]
        local_geometry = _merge_chunkers(local_edges)
        local_matrix = np.eye(system_size, dtype=complex) + _local_trace_matrix(
            local_geometry,
            trace,
        )
        if level == 1:
            compressed_inverse = np.linalg.inv(local_matrix[np.ix_(star_l, star_l)])
            saved_inverses[0] = compressed_inverse
        assert compressed_inverse is not None
        compressed_inverse = schur_banachiewicz(
            prolongation,
            weighted_prolongation,
            local_matrix,
            compressed_inverse,
            star_l,
            circ_l,
            star_s,
            circ_s,
        )
        saved_inverses[level] = compressed_inverse
        saved_blocks[level - 1] = local_matrix[np.ix_(star_l, circ_l)]
        saved_geometries[level - 1] = local_geometry

    assert compressed_inverse is not None
    star_indices = _corner_star_indices(graph, edge_ids, signs)
    saved = RCIPSaved(
        quadrature_order=quadrature_order,
        dimension=dimension,
        edge_count=edge_count,
        prolongation=prolongation,
        weighted_prolongation=weighted_prolongation,
        star_l=star_l,
        circ_l=circ_l,
        star_s=star_s,
        circ_s=circ_s,
        ilist=ilist,
        star_l_scalar=star_l_scalar,
        circ_l_scalar=circ_l_scalar,
        subdivisions=subdivisions,
        saved_depth=saved_depth,
        inverses=tuple(item for item in saved_inverses if item is not None),
        local_blocks=tuple(item for item in saved_blocks if item is not None),
        local_geometries=tuple(item for item in saved_geometries if item is not None),
        star_indices=star_indices,
        center=graph.vertices[vertex_id].position.copy(),
        starts_at_corner=starts_at_corner.copy(),
    )
    return compressed_inverse, saved, star_indices


def _local_trace_matrix(local_geometry: Chunker, trace: BoundaryTrace) -> NDArray[np.generic]:
    matrix = dense_panel_operator_matrix(
        local_geometry.pointinfo,
        local_geometry.pointinfo,
        trace.layer.kernel,
    )
    if trace.layer.kernel.family == "laplace" and trace.layer.kernel.selector in {"d", "sp"}:
        diagonal = (-local_geometry.signed_curvature / (4.0 * np.pi)).T.reshape(-1)
        np.fill_diagonal(matrix, diagonal * local_geometry.pointinfo.flat_weights)
    return trace.layer.coefficient * matrix


def _local_edge_chunker(
    graph: ChunkGraph,
    edge_id: int,
    sign: int,
    *,
    level: int,
    subdivisions: int,
    vertex_id: int,
) -> Chunker:
    edge = graph.edge(edge_id).chunker
    vertex = graph.vertices[vertex_id].position
    nodes = edge._legendre_nodes
    reference_weights = edge._legendre_weights
    if sign < 0:
        tangent = _unit(edge.derivatives[:, 0, 0])
        base_length = float(np.sum(edge.weights[:, 0]))
        scale = base_length / (2 ** (subdivisions - level))
        split = _line_interval_chunker(
            tangent,
            [(0.0, 0.5 * scale), (0.5 * scale, scale)]
            if level == subdivisions
            else [(0.0, 0.5 * scale), (0.5 * scale, scale), (scale, 2.0 * scale)],
            nodes,
            reference_weights,
        )
        if level != subdivisions:
            return split
        return _merge_chunkers((split, _panel_relative_chunker(edge, 1, vertex)))

    tangent = _unit(edge.derivatives[:, 0, -1])
    base_length = float(np.sum(edge.weights[:, -1]))
    scale = base_length / (2 ** (subdivisions - level))
    split = _line_interval_chunker(
        tangent,
        [(-scale, -0.5 * scale), (-0.5 * scale, 0.0)]
        if level == subdivisions
        else [(-2.0 * scale, -scale), (-scale, -0.5 * scale), (-0.5 * scale, 0.0)],
        nodes,
        reference_weights,
    )
    if level != subdivisions:
        return split
    return _merge_chunkers((_panel_relative_chunker(edge, edge.panel_count - 2, vertex), split))


def _line_interval_chunker(
    tangent: NDArray[np.floating],
    intervals: list[tuple[float, float]],
    nodes: NDArray[np.floating],
    reference_weights: NDArray[np.floating],
) -> Chunker:
    panel_count = len(intervals)
    positions = np.empty((2, nodes.size, panel_count), dtype=float)
    derivatives = np.empty_like(positions)
    second = np.zeros_like(positions)
    weights = np.empty((nodes.size, panel_count), dtype=float)
    tangent = np.asarray(tangent, dtype=float).reshape(2)
    for panel_id, (left, right) in enumerate(intervals):
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        signed_distance = midpoint + half_width * nodes
        positions[:, :, panel_id] = tangent[:, None] * signed_distance[None, :]
        derivatives[:, :, panel_id] = tangent[:, None] * half_width
        weights[:, panel_id] = reference_weights * abs(half_width)
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second,
        normals=right_normals(derivatives),
        weights=weights,
        _legendre_nodes=nodes,
        _legendre_weights=reference_weights,
        adjacency=_open_adjacency(panel_count),
        closed=False,
        orientation="open",
    )


def _panel_relative_chunker(edge: Chunker, panel_id: int, vertex: NDArray[np.floating]) -> Chunker:
    if not 0 <= panel_id < edge.panel_count:
        raise ValueError("each RCIP edge needs at least two coarse panels")
    return Chunker(
        positions=edge.positions[:, :, panel_id : panel_id + 1] - vertex[:, None, None],
        derivatives=edge.derivatives[:, :, panel_id : panel_id + 1].copy(),
        second_derivatives=edge.second_derivatives[:, :, panel_id : panel_id + 1].copy(),
        normals=edge.normals[:, :, panel_id : panel_id + 1].copy(),
        weights=edge.weights[:, panel_id : panel_id + 1].copy(),
        _legendre_nodes=edge._legendre_nodes.copy(),
        _legendre_weights=edge._legendre_weights.copy(),
        adjacency=_open_adjacency(1),
        closed=False,
        orientation="open",
    )


def _merge_chunkers(chunks: tuple[Chunker, ...] | list[Chunker]) -> Chunker:
    if not chunks:
        raise ValueError("at least one chunker is required")
    first = chunks[0]
    panel_count = sum(chunk.panel_count for chunk in chunks)
    return Chunker(
        positions=np.concatenate([chunk.positions for chunk in chunks], axis=2),
        derivatives=np.concatenate([chunk.derivatives for chunk in chunks], axis=2),
        second_derivatives=np.concatenate([chunk.second_derivatives for chunk in chunks], axis=2),
        normals=np.concatenate([chunk.normals for chunk in chunks], axis=2),
        weights=np.concatenate([chunk.weights for chunk in chunks], axis=1),
        _legendre_nodes=first._legendre_nodes.copy(),
        _legendre_weights=first._legendre_weights.copy(),
        adjacency=_open_adjacency(panel_count),
        closed=False,
        orientation="open",
    )


def _open_adjacency(panel_count: int) -> NDArray[np.integer]:
    adjacency = np.full((2, int(panel_count)), -1, dtype=np.int64)
    for panel_id in range(int(panel_count)):
        if panel_id > 0:
            adjacency[0, panel_id] = panel_id - 1
        if panel_id + 1 < int(panel_count):
            adjacency[1, panel_id] = panel_id + 1
    return adjacency


def _corner_star_indices(
    graph: ChunkGraph,
    edge_ids: NDArray[np.integer],
    signs: NDArray[np.integer],
) -> NDArray[np.integer]:
    offsets = _edge_point_offsets(graph)
    width = 2 * graph.quadrature_order
    out: list[NDArray[np.integer]] = []
    for edge_id, sign in zip(edge_ids, signs, strict=True):
        edge = graph.edge(int(edge_id))
        start = offsets[int(edge_id)]
        stop = start + edge.chunker.point_count
        if stop - start < width:
            raise ValueError("each RCIP edge needs at least two coarse panels")
        out.append(
            np.arange(start, start + width, dtype=np.int64)
            if int(sign) < 0
            else np.arange(stop - width, stop, dtype=np.int64)
        )
    return np.concatenate(out)


def _edge_point_offsets(graph: ChunkGraph) -> dict[int, int]:
    offsets: dict[int, int] = {}
    cursor = 0
    for edge in graph.edges:
        offsets[edge.id] = cursor
        cursor += edge.chunker.point_count
    return offsets


def _interpolate_saved_density(
    rhohat,
    saved: RCIPSaved,
    *,
    depth: int,
) -> tuple[list[NDArray[np.generic]], list[PointInfoView]]:
    rho = np.asarray(rhohat).reshape(-1)
    if saved.subdivisions <= 0:
        return [rho.copy()], []
    active_depth = min(int(depth), int(saved.subdivisions), int(saved.saved_depth))
    if active_depth <= 0:
        return [rho.copy()], []
    if (
        len(saved.inverses) < active_depth + 1
        or len(saved.local_blocks) < active_depth
        or len(saved.local_geometries) < active_depth
    ):
        raise ValueError("RCIP saved state does not contain enough recursion data")

    star_s = np.sort(saved.star_s)
    circ_s = np.sort(saved.circ_s)
    star_l_scalar = np.sort(saved.star_l_scalar)
    circ_l_scalar = np.sort(saved.circ_l_scalar)
    unknown_count = circ_s.size + star_s.size
    if rho.size % unknown_count != 0:
        raise ValueError("density has incompatible size for RCIP saved data")
    density_count = rho.size // unknown_count
    rhohat0 = rho.reshape(unknown_count, density_count)

    circ_s_edges = _split_edge_indices(circ_s, saved.edge_count)
    star_s_edges = _split_edge_indices(star_s, saved.edge_count)
    circ_l_edges = _split_edge_indices(circ_l_scalar, saved.edge_count)
    star_l_edges = _split_edge_indices(star_l_scalar, saved.edge_count)

    local_geometry = saved.local_geometries[-1]
    rho_cells = [rhohat0[indices, :].copy() for indices in circ_s_edges]
    source_cells = [
        _pointinfo_subset(local_geometry.pointinfo, indices) for indices in circ_l_edges
    ]

    r0 = saved.inverses[-1]
    for interpolation_depth in range(1, active_depth + 1):
        r1 = saved.inverses[-interpolation_depth - 1]
        local_block = saved.local_blocks[-interpolation_depth]
        temporary = np.linalg.solve(r0, rhohat0)
        rhohat0 = r1 @ (
            saved.prolongation @ temporary[star_s, :] - local_block @ rhohat0[circ_s, :]
        )
        if interpolation_depth == active_depth:
            for edge_id in range(saved.edge_count):
                if saved.starts_at_corner[edge_id]:
                    order = np.concatenate((star_s_edges[edge_id], circ_s_edges[edge_id]))
                else:
                    order = np.concatenate((circ_s_edges[edge_id], star_s_edges[edge_id]))
                rho_cells[edge_id] = np.vstack((rho_cells[edge_id], rhohat0[order, :]))
                source_cells[edge_id] = _pointinfo_append(
                    source_cells[edge_id],
                    _pointinfo_subset(local_geometry.pointinfo, star_l_edges[edge_id]),
                )
        else:
            local_geometry = saved.local_geometries[-interpolation_depth - 1]
            for edge_id in range(saved.edge_count):
                rho_cells[edge_id] = np.vstack(
                    (rho_cells[edge_id], rhohat0[circ_s_edges[edge_id], :])
                )
                source_cells[edge_id] = _pointinfo_append(
                    source_cells[edge_id],
                    _pointinfo_subset(local_geometry.pointinfo, circ_l_edges[edge_id]),
                )
        r0 = r1

    if density_count == 1:
        return [cell[:, 0] for cell in rho_cells], source_cells
    return [cell.T.reshape(-1) for cell in rho_cells], source_cells


def _split_edge_indices(indices: NDArray[np.integer], edge_count: int) -> list[NDArray[np.integer]]:
    if indices.size % int(edge_count) != 0:
        raise ValueError("RCIP indices are not evenly split by edge")
    per_edge = indices.size // int(edge_count)
    return [
        indices[edge_id * per_edge : (edge_id + 1) * per_edge] for edge_id in range(int(edge_count))
    ]


def _pointinfo_subset(source: PointInfoView, indices: NDArray[np.integer]) -> PointInfoView:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    quadrature_order = source.nodes.size
    if indices.size % quadrature_order != 0:
        raise ValueError("RCIP pointinfo subsets must contain whole panels")
    panel_ids = indices.reshape(-1, quadrature_order)[:, 0] // quadrature_order
    return PointInfoView(
        positions=source.positions[:, :, panel_ids],
        derivatives=source.derivatives[:, :, panel_ids],
        second_derivatives=source.second_derivatives[:, :, panel_ids],
        normals=source.normals[:, :, panel_ids],
        weights=source.weights[:, panel_ids],
        nodes=source.nodes,
        panel_ids=np.arange(panel_ids.size, dtype=np.int64),
    )


def _pointinfo_append(left: PointInfoView, right: PointInfoView) -> PointInfoView:
    panel_count = left.positions.shape[2] + right.positions.shape[2]
    return PointInfoView(
        positions=np.concatenate((left.positions, right.positions), axis=2),
        derivatives=np.concatenate((left.derivatives, right.derivatives), axis=2),
        second_derivatives=np.concatenate(
            (left.second_derivatives, right.second_derivatives),
            axis=2,
        ),
        normals=np.concatenate((left.normals, right.normals), axis=2),
        weights=np.concatenate((left.weights, right.weights), axis=1),
        nodes=left.nodes,
        panel_ids=np.arange(panel_count, dtype=np.int64),
    )


def _vector_to_panel_density(vector, source: PointInfoView) -> NDArray[np.generic]:
    values = np.asarray(vector)
    return values.reshape(source.positions.shape[2], source.nodes.size).T[None, :, :]


def _shift_targets(targets, center: NDArray[np.floating]):
    positions = flat_positions(targets) - np.asarray(center, dtype=float).reshape(2, 1)
    normals = None
    try:
        normals = flat_normals(targets, label="target")
    except ValueError:
        normals = None
    return _TargetView(positions=positions, normals=normals)


def _unit(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero-length RCIP edge tangent")
    return np.asarray(vector, dtype=float) / norm


def _first_trace_term(system) -> BoundaryTrace | None:
    for equation in system.equations:
        for term in equation.terms:
            if isinstance(term, BoundaryTrace):
                return term
    return None


def _local_corner_trace_operator(local_geometry, trace: BoundaryTrace) -> NDArray[np.generic]:
    matrix = dense_panel_operator_matrix(
        local_geometry.pointinfo,
        local_geometry.pointinfo,
        trace.layer.kernel,
    )
    if trace.layer.kernel.family == "laplace" and trace.layer.kernel.selector in {"d", "sp"}:
        diagonal = (-local_geometry.signed_curvature / (4.0 * np.pi)).T.reshape(-1)
        np.fill_diagonal(matrix, diagonal * local_geometry.pointinfo.flat_weights)
    if trace.jump is not None:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("local RCIP jump terms require square trace operators")
        matrix = matrix + trace.jump.coefficient * np.eye(matrix.shape[0])
    return matrix
