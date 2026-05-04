"""Utilities mirroring MATLAB ``+chnk``."""

from . import arcparam, curves, elast2d, geometry, helm1d, helm2d, lap2d, quadggq, quadnative, rcip, smoother, spcl, stok2d
from .geometry import (
    chunk_nearparam,
    curvature2d,
    flagnear,
    flagnear_rectangle,
    flagnear_rectangle_grid,
    flagself,
    normal2d,
    perp,
)

__all__ = [
    "chunk_nearparam",
    "arcparam",
    "curvature2d",
    "curves",
    "elast2d",
    "flagnear",
    "flagnear_rectangle",
    "flagnear_rectangle_grid",
    "flagself",
    "geometry",
    "helm1d",
    "helm2d",
    "lap2d",
    "normal2d",
    "perp",
    "quadggq",
    "quadnative",
    "rcip",
    "smoother",
    "spcl",
    "stok2d",
]
