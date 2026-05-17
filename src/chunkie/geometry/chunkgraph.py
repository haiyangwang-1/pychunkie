"""Region-aware collections of chunker edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .chunker import Chunker
from .constructors import chunker_from_polygon
from .points import PointInfoView


@dataclass(frozen=True)
class SignedEdge:
    edge_id: int
    orientation: int


@dataclass(frozen=True)
class RegionCycle:
    edges: tuple[SignedEdge, ...]


@dataclass(frozen=True)
class GraphVertex:
    id: int
    position: NDArray[np.floating]
    incident_edges: tuple[int, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    id: int
    chunker: Chunker
    start_vertex: int
    end_vertex: int
    orientation: int
    left_region: int | None
    right_region: int | None
    label: str | None = None


@dataclass(frozen=True)
class GraphRegion:
    id: int
    boundary_cycles: tuple[RegionCycle, ...]
    bounded: bool
    label: str | None = None


@dataclass(frozen=True)
class BoundaryPart:
    graph: ChunkGraph | None
    edges: tuple[int, ...]
    point_indices: NDArray[np.integer]
    side: Literal["left", "right", "interior", "exterior"] | int | None
    orientation: NDArray[np.integer]
    points: PointInfoView


@dataclass
class ChunkGraph:
    vertices: list[GraphVertex]
    edges: list[GraphEdge]
    regions: list[GraphRegion]

    @classmethod
    def from_chunker(cls, chunker: Chunker, *, exterior_region: int = 0, interior_region: int = 1) -> ChunkGraph:
        start = GraphVertex(0, chunker.positions[:, 0, 0])
        edge = GraphEdge(
            id=0,
            chunker=chunker,
            start_vertex=0,
            end_vertex=0,
            orientation=1,
            left_region=interior_region,
            right_region=exterior_region,
            label="boundary",
        )
        exterior = GraphRegion(exterior_region, (), bounded=False, label="exterior")
        interior = GraphRegion(
            interior_region,
            (RegionCycle((SignedEdge(edge_id=0, orientation=1),)),),
            bounded=True,
            label="interior",
        )
        return cls(vertices=[start], edges=[edge], regions=[exterior, interior])

    @classmethod
    def from_vertices(
        cls,
        vertices,
        edge_vertices,
        *,
        quadrature_order: int = 16,
        exterior_region: int = 0,
        interior_region: int = 1,
    ) -> ChunkGraph:
        vertex_array = _as_vertex_array(vertices)
        edge_array = _as_edge_array(edge_vertices)
        graph_vertices = [
            GraphVertex(id=index, position=vertex_array[:, index], incident_edges=tuple(np.where(edge_array == index)[1]))
            for index in range(vertex_array.shape[1])
        ]
        graph_edges: list[GraphEdge] = []
        for edge_id in range(edge_array.shape[1]):
            start_vertex = int(edge_array[0, edge_id])
            end_vertex = int(edge_array[1, edge_id])
            edge_chunker = chunker_from_polygon(
                vertex_array[:, [start_vertex, end_vertex]],
                quadrature_order=quadrature_order,
                closed=False,
            )
            # Directed region metadata follows the usual planar convention:
            # walking an interface in its stored orientation, the bounded
            # material region is on the left and the exterior is on the right.
            graph_edges.append(
                GraphEdge(
                    id=edge_id,
                    chunker=edge_chunker,
                    start_vertex=start_vertex,
                    end_vertex=end_vertex,
                    orientation=1,
                    left_region=interior_region,
                    right_region=exterior_region,
                )
            )
        exterior = GraphRegion(exterior_region, (), bounded=False, label="exterior")
        interior = GraphRegion(
            interior_region,
            (RegionCycle(tuple(SignedEdge(edge_id=edge_id, orientation=1) for edge_id in range(edge_array.shape[1]))),),
            bounded=True,
            label="interior",
        )
        return cls(vertices=graph_vertices, edges=graph_edges, regions=[exterior, interior])

    @property
    def point_count(self) -> int:
        return sum(edge.chunker.point_count for edge in self.edges)

    @property
    def panel_count(self) -> int:
        return sum(edge.chunker.panel_count for edge in self.edges)

    @property
    def quadrature_order(self) -> int:
        if not self.edges:
            raise ValueError("empty ChunkGraph has no quadrature order")
        order = self.edges[0].chunker.quadrature_order
        if any(edge.chunker.quadrature_order != order for edge in self.edges):
            raise ValueError("merged ChunkGraph point views require a common quadrature order")
        return order

    @property
    def pointinfo(self) -> PointInfoView:
        return self.merged_points()

    def edge(self, edge_id: int) -> GraphEdge:
        return self.edges[edge_id]

    def edge_points(self, edge_id: int) -> PointInfoView:
        return self.edge(edge_id).chunker.pointinfo

    def region(self, region_id: int) -> GraphRegion:
        for region in self.regions:
            if region.id == region_id:
                return region
        raise KeyError(region_id)

    def merged_points(self) -> PointInfoView:
        if not self.edges:
            raise ValueError("empty ChunkGraph has no points")
        order = self.quadrature_order
        first = self.edges[0].chunker
        nodes = first.nodes
        positions = np.concatenate([edge.chunker.positions for edge in self.edges], axis=2)
        derivatives = np.concatenate([edge.chunker.derivatives for edge in self.edges], axis=2)
        second_derivatives = np.concatenate([edge.chunker.second_derivatives for edge in self.edges], axis=2)
        normals = np.concatenate([edge.chunker.normals for edge in self.edges], axis=2)
        weights = np.concatenate([edge.chunker.weights for edge in self.edges], axis=1)

        # A merged graph view uses global panel-major ids. Edge-to-panel offsets
        # are tracked by ChunkGraph methods, not by PointInfoView itself.
        return PointInfoView(
            positions=positions,
            derivatives=derivatives,
            second_derivatives=second_derivatives,
            normals=normals,
            weights=weights,
            nodes=nodes,
            panel_ids=np.arange(positions.shape[2], dtype=np.int64),
            point_map=first.point_map.__class__(order, positions.shape[2]),
        )

    def boundary(
        self,
        region_id: int,
        *,
        side: Literal["left", "right", "interior", "exterior"] | int | None = "interior",
    ) -> BoundaryPart:
        region = self.region(region_id)
        signed_edges = tuple(edge for cycle in region.boundary_cycles for edge in cycle.edges)
        edge_ids = tuple(edge.edge_id for edge in signed_edges)
        offsets = self._edge_point_offsets()
        point_indices = np.concatenate(
            [
                np.arange(
                    offsets[edge_id],
                    offsets[edge_id] + self.edge(edge_id).chunker.point_count,
                    dtype=np.int64,
                )
                for edge_id in edge_ids
            ]
        )
        return BoundaryPart(
            graph=self,
            edges=edge_ids,
            point_indices=point_indices,
            side=side,
            orientation=np.asarray([edge.orientation for edge in signed_edges], dtype=np.int64),
            points=self.merged_points(),
        )

    def classify_points(self, points) -> NDArray[np.integer]:
        targets = np.asarray(points, dtype=float).reshape(2, -1)
        labels = np.zeros(targets.shape[1], dtype=np.int64)
        for region in self.regions:
            if not region.bounded:
                continue
            labels[:] = np.where(_point_in_region(self, region, targets), region.id, labels)
        return labels

    def _edge_point_offsets(self) -> dict[int, int]:
        offsets: dict[int, int] = {}
        cursor = 0
        for edge in self.edges:
            offsets[edge.id] = cursor
            cursor += edge.chunker.point_count
        return offsets


def _point_in_region(graph: ChunkGraph, region: GraphRegion, targets: NDArray[np.floating]) -> NDArray[np.bool_]:
    inside = np.zeros(targets.shape[1], dtype=bool)
    for cycle in region.boundary_cycles:
        inside |= _point_in_cycle(graph, cycle, targets)
    return inside


def _point_in_cycle(graph: ChunkGraph, cycle: RegionCycle, targets: NDArray[np.floating]) -> NDArray[np.bool_]:
    vertices = _cycle_vertices(graph, cycle)
    x = targets[0]
    y = targets[1]
    inside = np.zeros(targets.shape[1], dtype=bool)
    x0 = vertices[0]
    y0 = vertices[1]
    xj = x0[-1]
    yj = y0[-1]
    for xi, yi in zip(x0, y0, strict=True):
        crosses = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / (yj - yi + np.finfo(float).eps) + xi
        inside ^= crosses & (x < x_intersect)
        xj = xi
        yj = yi
    return inside


def _cycle_vertices(graph: ChunkGraph, cycle: RegionCycle) -> NDArray[np.floating]:
    parts: list[NDArray[np.floating]] = []
    for signed_edge in cycle.edges:
        edge = graph.edge(signed_edge.edge_id)
        starts = edge.chunker.positions[:, 0, :]
        if signed_edge.orientation < 0:
            starts = starts[:, ::-1]
        parts.append(starts)
    if not parts:
        return np.zeros((2, 0))
    return np.concatenate(parts, axis=1)


def _as_vertex_array(vertices) -> NDArray[np.floating]:
    vertex_array = np.asarray(vertices, dtype=float)
    if vertex_array.ndim != 2:
        raise ValueError("vertices must be a two-dimensional array")
    if vertex_array.shape[0] != 2 and vertex_array.shape[1] == 2:
        vertex_array = vertex_array.T
    if vertex_array.shape[0] != 2:
        raise ValueError("vertices must have shape (2, vertex_count) or (vertex_count, 2)")
    return vertex_array


def _as_edge_array(edge_vertices) -> NDArray[np.integer]:
    edge_array = np.asarray(edge_vertices, dtype=np.int64)
    if edge_array.ndim != 2:
        raise ValueError("edge_vertices must be a two-dimensional array")
    if edge_array.shape[0] != 2 and edge_array.shape[1] == 2:
        edge_array = edge_array.T
    if edge_array.shape[0] != 2:
        raise ValueError("edge_vertices must have shape (2, edge_count) or (edge_count, 2)")
    return edge_array
