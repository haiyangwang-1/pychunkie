"""Recursive compressed inverse preconditioning primitives."""

from .compression import RCIPState, schur_compress_block
from .interpolation import interpolate_density
from .local_geometry import LocalCornerGeometry, build_local_corner_geometry
from .prolongation import (
    build_block_prolongation,
    build_prolongation,
    build_split_panel_prolongation,
)

__all__ = [
    "LocalCornerGeometry",
    "RCIPState",
    "build_local_corner_geometry",
    "build_block_prolongation",
    "build_prolongation",
    "build_split_panel_prolongation",
    "interpolate_density",
    "schur_compress_block",
]
