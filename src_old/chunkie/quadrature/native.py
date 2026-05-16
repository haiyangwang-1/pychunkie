"""Native smooth quadrature builders.

The native rule is the ordinary panelwise Gauss-Legendre rule: evaluate the
kernel at source and target nodes, then multiply source columns by physical
arclength weights. It is correct for smooth interactions, and it is the base
matrix that the special quadrature modules overwrite near singular panels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix, as_boundary_vector, boundary_component_weights
from chunkie.geometry import PointInfo
from chunkie.geometry.chunker import Chunker


def buildmat(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    target_chunks: ArrayLike | None = None,
    source_chunks: ArrayLike | None = None,
    weights: ArrayLike | None = None,
) -> np.ndarray:
    """Build a smooth quadrature submatrix for selected source and target chunks.

    No singular or near-singular correction is applied here. Callers that need
    boundary self-interaction, neighboring panels, or close off-surface targets
    start with this matrix and replace only the affected blocks.
    """

    target_ids = (
        np.arange(chunker.nch)
        if target_chunks is None
        else np.asarray(target_chunks, dtype=int).reshape(-1)
    )
    source_ids = (
        np.arange(chunker.nch)
        if source_chunks is None
        else np.asarray(source_chunks, dtype=int).reshape(-1)
    )
    if (
        np.any(target_ids < 0)
        or np.any(target_ids >= chunker.nch)
        or np.any(source_ids < 0)
        or np.any(source_ids >= chunker.nch)
    ):
        raise IndexError("chunk index out of range")

    srcinfo = _pointinfo_for_chunks(chunker, source_ids)
    targinfo = _pointinfo_for_chunks(chunker, target_ids)
    if hasattr(kernel, "eval") and kernel.eval is not None:
        mat = kernel.eval(srcinfo, targinfo)
        if opdims is None and hasattr(kernel, "opdims"):
            opdims = kernel.opdims
    else:
        mat = kernel(srcinfo, targinfo)

    if opdims is None:
        source_count = chunker.k * source_ids.size
        opdims = (mat.shape[0] // (chunker.k * target_ids.size), mat.shape[1] // source_count)

    base_weights = (
        chunker.wstor if weights is None else np.asarray(weights, dtype=float).reshape(chunker.k)
    )
    speed = np.sqrt(np.sum(np.abs(chunker.d[:, :, source_ids]) ** 2, axis=0))
    smooth_wts = as_boundary_vector(speed * base_weights[:, None], name="smooth weights")
    return mat * boundary_component_weights(smooth_wts, int(opdims[1]))[None, :]


def _pointinfo_for_chunks(chunker: Chunker, chunks: np.ndarray) -> PointInfo:
    point_count = chunker.k * chunks.size
    return PointInfo(
        r=as_boundary_point_matrix(chunker.r[:, :, chunks], chunker.dim, point_count, name="r"),
        d=as_boundary_point_matrix(chunker.d[:, :, chunks], chunker.dim, point_count, name="d"),
        d2=as_boundary_point_matrix(chunker.d2[:, :, chunks], chunker.dim, point_count, name="d2"),
        n=as_boundary_point_matrix(chunker.n[:, :, chunks], chunker.dim, point_count, name="n"),
        data=as_boundary_point_matrix(
            chunker.data[:, :, chunks], chunker.datadim, point_count, name="data"
        )
        if chunker.datadim
        else None,
    )
