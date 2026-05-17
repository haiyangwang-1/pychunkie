"""Nonsmooth corner system integration."""

from __future__ import annotations

import numpy as np

from chunkie.geometry import ChunkGraph
from chunkie.quadrature import dense_panel_operator_matrix
from chunkie.rcip import (
    RCIPCornerState,
    RCIPState,
    build_block_prolongation,
    build_local_corner_geometry,
    build_split_panel_prolongation,
)

from .config import SystemConfig
from .trace import BoundaryTrace


def build_rcip_state(system, *, config: SystemConfig | None = None) -> RCIPState:
    """Build dense-reference RCIP corner metadata for a graph-backed system."""

    active_config = SystemConfig() if config is None else config
    geometry = system.geometry
    if not isinstance(geometry, ChunkGraph):
        return RCIPState(metadata={"active": False, "reason": "geometry is not a ChunkGraph"})
    if not geometry.edges:
        return RCIPState(metadata={"active": False, "reason": "graph has no edges"})

    nodes = geometry.edges[0].chunker.nodes
    weights = geometry.edges[0].chunker.reference_weights
    _, _, interpolation, weighted_transfer = build_split_panel_prolongation(nodes, weights)
    trace_template = _first_trace_term(system)
    corners: list[RCIPCornerState] = []
    for vertex in geometry.vertices:
        edge_ids = tuple(int(edge_id) for edge_id in vertex.incident_edges)
        if len(edge_ids) < 2:
            continue
        directions = _corner_directions(geometry, vertex.id, edge_ids)
        local_geometry = build_local_corner_geometry(
            vertex.position,
            directions,
            quadrature_order=geometry.quadrature_order,
            levels=max(1, int(active_config.rcip_subdivisions)),
        )
        block_prolongation = build_block_prolongation(interpolation, edge_count=len(edge_ids), component_count=1)
        weighted_block = build_block_prolongation(weighted_transfer, edge_count=len(edge_ids), component_count=1)
        # The first system integration layer records the local star layout and
        # starts from an identity compressed inverse; recursive Schur updates
        # consume the same object once local block assembly is wired in.
        star_size = len(edge_ids) * geometry.quadrature_order
        corners.append(
            RCIPCornerState(
                vertex_id=vertex.id,
                edge_ids=edge_ids,
                boundary_part=geometry.boundary_part(edge_ids),
                local_geometry=local_geometry,
                prolongation=block_prolongation,
                weighted_prolongation=weighted_block,
                compressed_inverse=np.eye(star_size),
                local_operator=(
                    _local_corner_trace_operator(local_geometry, trace_template)
                    if trace_template is not None
                    else None
                ),
            )
        )

    return RCIPState(
        corners=tuple(corners),
        metadata={
            "active": bool(corners),
            "subdivisions": int(active_config.rcip_subdivisions),
            "quadrature_order": geometry.quadrature_order,
        },
    )


def _corner_directions(graph: ChunkGraph, vertex_id: int, edge_ids: tuple[int, ...]) -> np.ndarray:
    directions = []
    vertex_position = graph.vertices[vertex_id].position
    for edge_id in edge_ids:
        edge = graph.edge(edge_id)
        if edge.start_vertex == vertex_id:
            other = graph.vertices[edge.end_vertex].position
        elif edge.end_vertex == vertex_id:
            other = graph.vertices[edge.start_vertex].position
        else:
            raise ValueError("incident edge does not touch vertex")
        directions.append(other - vertex_position)
    return np.column_stack(directions)


def _first_trace_term(system) -> BoundaryTrace | None:
    for equation in system.equations:
        for term in equation.terms:
            if isinstance(term, BoundaryTrace):
                return term
    return None


def _local_corner_trace_operator(local_geometry, trace: BoundaryTrace) -> np.ndarray:
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
        # This local matrix is the dense reference object used by Schur
        # compression. Global insertion is separate; here the one-sided jump is
        # stored in the same local ordering as the refined corner panels.
        matrix = matrix + trace.jump.coefficient * np.eye(matrix.shape[0])
    return matrix
