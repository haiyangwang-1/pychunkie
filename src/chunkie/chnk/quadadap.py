"""Adaptive-special quadrature entry points.

The MATLAB implementation adaptively decides which close interactions need
replacement quadrature. This Python baseline exposes the same build path
and delegates the actual replacement blocks to ``quadggq``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..chunker import Chunker
from . import quadggq


def buildmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    opts: dict[str, Any] | None = None,
) -> np.ndarray:
    options = {} if opts is None else dict(opts)
    qtype = str(options.get("sing", getattr(kern, "sing", "log") or "log")).lower()
    return quadggq.buildmat(
        chnkr,
        kern,
        opdims if opdims is not None else getattr(kern, "opdims", None),
        type=qtype,
        ilist=options.get("ilist", None),
    )
