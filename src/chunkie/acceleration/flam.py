"""Public FLAM callback utility facade."""

from __future__ import annotations

from ._flam_index import kernbyindex, kernbyindexr
from ._flam_proxy import (
    nproxy_square,
    proxy_circ_pts,
    proxy_rect_pts,
    proxy_square_pts,
    proxyfun,
    proxyfunr,
)

__all__ = [
    "kernbyindex",
    "kernbyindexr",
    "proxy_square_pts",
    "proxy_rect_pts",
    "proxy_circ_pts",
    "nproxy_square",
    "proxyfun",
    "proxyfunr",
]
