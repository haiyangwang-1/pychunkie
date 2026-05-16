"""Region-aware collections of chunker edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .chunker import Chunker
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

    @property
    def point_count(self) -> int:
        return sum(edge.chunker.point_count for edge in self.edges)

    def edge(self, edge_id: int) -> GraphEdge:
        return self.edges[edge_id]

    def region(self, region_id: int) -> GraphRegion:
        for region in self.regions:
            if region.id == region_id:
                return region
        raise KeyError(region_id)
