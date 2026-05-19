"""Geometry storage and construction."""

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
from .points import PanelView, PointInfoView
from .refine import refine
from .transforms import affine, reflect, rotate, scale, translate

__all__ = [
    "BoundaryPart",
    "BernsteinPanelImage",
    "ChunkGraph",
    "Chunker",
    "GraphEdge",
    "GraphRegion",
    "GraphVertex",
    "NearestPoint",
    "PanelView",
    "PointInfoView",
    "RegionCycle",
    "SignedEdge",
    "bernstein_ellipse",
    "bernstein_panel_image",
    "bernstein_radius",
    "chunker_from_curve",
    "chunker_from_polygon",
    "circle",
    "ellipse",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "nearest_point",
    "affine",
    "reflect",
    "refine",
    "rotate",
    "scale",
    "translate",
]
