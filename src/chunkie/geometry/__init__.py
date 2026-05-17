"""Geometry storage and construction."""

from .arclength import (
    ArcLengthParameterization,
    arclength_parameterization,
    evaluate_arclength,
    resample_by_arclength,
)
from .chunker import Chunker
from .chunkgraph import (
    BoundaryPart,
    ChunkGraph,
    GraphEdge,
    GraphRegion,
    GraphVertex,
    RegionCycle,
    SignedEdge,
)
from .constructors import chunker_from_curve, chunker_from_polygon, circle, ellipse
from .near import NearestPoint, flagnear, flagnear_rectangle, flagnear_rectangle_grid, nearest_point
from .points import PanelView, PointInfoView, PointMap
from .transforms import affine, reflect, rotate, scale, translate

__all__ = [
    "BoundaryPart",
    "ArcLengthParameterization",
    "ChunkGraph",
    "Chunker",
    "GraphEdge",
    "GraphRegion",
    "GraphVertex",
    "NearestPoint",
    "PanelView",
    "PointInfoView",
    "PointMap",
    "RegionCycle",
    "SignedEdge",
    "arclength_parameterization",
    "chunker_from_curve",
    "chunker_from_polygon",
    "circle",
    "ellipse",
    "evaluate_arclength",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "nearest_point",
    "resample_by_arclength",
    "affine",
    "reflect",
    "rotate",
    "scale",
    "translate",
]
