"""Utilities mirroring MATLAB ``+chnk``."""

from . import curves, elast2d, geometry, helm2d, lap2d, quadggq, quadnative, stok2d
from .geometry import chunk_nearparam, curvature2d, flagnear, flagself, normal2d, perp

__all__ = [
    "chunk_nearparam",
    "curvature2d",
    "curves",
    "elast2d",
    "flagnear",
    "flagself",
    "geometry",
    "helm2d",
    "lap2d",
    "normal2d",
    "perp",
    "quadggq",
    "quadnative",
    "stok2d",
]
