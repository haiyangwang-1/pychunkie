"""Utilities mirroring MATLAB ``+chnk``."""

from . import curves, elast2d, geometry, helm2d, lap2d, stok2d
from .geometry import chunk_nearparam, flagnear, flagself

__all__ = [
    "chunk_nearparam",
    "curves",
    "elast2d",
    "flagnear",
    "flagself",
    "geometry",
    "helm2d",
    "lap2d",
    "stok2d",
]
