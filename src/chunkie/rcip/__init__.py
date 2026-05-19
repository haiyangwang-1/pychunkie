"""Recursive compressed inverse preconditioning primitives."""

from .compression import (
    RCIPCornerState,
    RCIPSaved,
    RCIPSchurLevel,
    RCIPState,
    RecursiveCompressionResult,
    recursive_schur_compress,
    schur_compress_block,
)
from .interpolation import interpolate_density
from .local_geometry import LocalCornerGeometry, build_local_corner_geometry
from .prolongation import (
    build_block_prolongation,
    build_prolongation,
    build_split_panel_prolongation,
)

__all__ = [
    "LocalCornerGeometry",
    "RCIPCornerState",
    "RCIPSaved",
    "RCIPSchurLevel",
    "RCIPState",
    "RecursiveCompressionResult",
    "build_local_corner_geometry",
    "build_block_prolongation",
    "build_prolongation",
    "build_split_panel_prolongation",
    "interpolate_density",
    "recursive_schur_compress",
    "schur_compress_block",
]
