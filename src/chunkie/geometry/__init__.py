"""Geometry storage and construction."""

from .arclength import (
    ArcLengthParameterization,
    arclength_parameterization,
    evaluate_arclength,
    resample_by_arclength,
)
from .bernstein import (
    BernsteinPanelImage,
    bernstein_ellipse,
    bernstein_panel_image,
    bernstein_radius,
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
from .refine import change_quadrature_order, refine
from .transforms import affine, reflect, rotate, scale, translate

__all__ = [
    "BoundaryPart",
    "ArcLengthParameterization",
    "BernsteinPanelImage",
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
    "bernstein_ellipse",
    "bernstein_panel_image",
    "bernstein_radius",
    "chunker_from_curve",
    "chunker_from_polygon",
    "change_quadrature_order",
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
    "refine",
    "rotate",
    "scale",
    "translate",
]
