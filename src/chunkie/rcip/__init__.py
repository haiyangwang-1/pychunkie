"""Recursive compressed inverse preconditioning primitives."""

from .compression import RCIPState
from .interpolation import interpolate_density
from .local_geometry import LocalCornerGeometry, build_local_corner_geometry
from .prolongation import build_prolongation

__all__ = [
    "LocalCornerGeometry",
    "RCIPState",
    "build_local_corner_geometry",
    "build_prolongation",
    "interpolate_density",
]
