"""Public target-evaluation and interior-classification entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse import spmatrix

from .._layout import as_boundary_field_matrix, as_boundary_point_matrix, as_boundary_vector
from ..geometry.chunker import Chunker
from ..geometry.pointinfo import PointInfo
from ._common import (
    _acceleration,
    _as_chunker,
    _eval_kernel,
    _flag,
    _fmm_tol,
    _require_chunker,
    _require_fmm,
    _uses_special_quadrature,
    _weighted_density,
)
from ._flam import _chunkerkerneval_flam, _chunkerkernevalmat_flam
from ._fmm import _chunkerkernevalmat_fmm, _smooth_fmm_options
from ._interior import _chunkerinterior_direct
from ._rcip import _chunkgraph_rcip_eval, _chunkgraph_rcip_eval_context
from ._special import (
    _special_correction_matrix,
    _target_adaptive_correction_matrix,
    _target_adaptive_matrix,
)
from .options import _normalize_public_options, _set_option


def chunkerinterior(
    chunker: Chunker,
    points: Chunker | dict[str, Any] | ArrayLike | tuple[ArrayLike, ArrayLike] | list[ArrayLike],
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    near_factor: float | None = None,
    close_correction: bool | None = None,
    axis_symmetric: bool | None = None,
) -> np.ndarray:
    """Classify target points as inside a closed 2D chunker.

    The default ``acceleration="dense"`` path is a dependency-light direct
    polygon test. With ``acceleration="fmm"`` or ``acceleration="flam"``, the
    Laplace double-layer identity is used for accelerated classification, and
    near-boundary targets are corrected by the direct path.
    """

    boundary = _require_chunker(chunker)
    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        near_factor=near_factor,
    )
    _set_option(operator_options, "closecorr", close_correction)
    _set_option(operator_options, "axissym", axis_symmetric)
    if boundary.dim != 2:
        raise ValueError("interior only well-defined for 2D chunkers")

    grid_shape = None
    if isinstance(points, (tuple, list)) and len(points) == 2:
        x = np.asarray(points[0], dtype=float)
        y = np.asarray(points[1], dtype=float)
        xx, yy = np.meshgrid(x, y)
        pts = np.vstack((xx.ravel(), yy.ravel()))
        grid_shape = xx.shape
    elif _as_chunker(points) is not None:
        target_boundary = _require_chunker(points)
        pts = as_boundary_point_matrix(
            target_boundary.r,
            target_boundary.dim,
            target_boundary.npt,
            name="target positions",
        )
    elif isinstance(points, dict) and "r" in points:
        arr = np.asarray(points["r"], dtype=float)
        pts = arr.reshape(arr.shape[0], -1)
    else:
        arr = np.asarray(points, dtype=float)
        pts = arr.reshape(arr.shape[0], -1)

    if pts.shape[0] != 2:
        raise ValueError("target points must be two-dimensional")

    acceleration = _acceleration(operator_options)
    if acceleration in {"fmm", "flam"}:
        from ..kernels import kernel

        lap_d = kernel("lap", "d")
        density = np.ones(boundary.npt)
        eval_options = dict(operator_options)
        eval_options["acceleration"] = acceleration
        vals = chunkerkerneval(
            boundary,
            lap_d,
            density,
            PointInfo(r=pts),
            eval_options,
        )
        vals = as_boundary_vector(vals, name="interior layer values")
        inside = vals < -0.5
        if _flag(operator_options, "closecorr", _flag(operator_options, "corrections", True)):
            near_fac = float(operator_options.get("near_fac", operator_options.get("fac", 1.0)))
            near = np.any(boundary.flagnear(pts, fac=near_fac), axis=1)
            if np.any(near):
                corrected = _chunkerinterior_direct(
                    boundary, pts[:, near], _flag(operator_options, "axissym")
                )
                inside[near] = corrected
    else:
        inside = _chunkerinterior_direct(boundary, pts, _flag(operator_options, "axissym"))

    if grid_shape is not None:
        return inside.reshape(grid_shape)
    return inside


def chunkerkerneval(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    correction_matrix: ArrayLike | sparse.spmatrix | None = None,
    near_factor: float | None = None,
    side: str | None = None,
    rcip: Any | None = None,
    rcip_context: Any | None = None,
    rcip_subdivisions: int | float | None = None,
    rcip_save_depth: int | None = None,
    rcip_eval_depth: int | None = None,
) -> np.ndarray:
    """Evaluate a layer potential generated by a chunker density at targets.

    ``density`` is converted to the flat boundary vector expected by solver and
    backend adapters, including any vector components implied by
    ``kernel.opdims``. ``target`` may be raw coordinates, a
    ``PointInfo``/mapping, another chunker, or a chunkgraph-like object.
    Options include ``acceleration="fmm"`` for supported FMM kernels,
    ``acceleration="flam"`` for PyFLAM target evaluation, and ``force_adaptive=True``
    for adaptive close-target correction.
    """

    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        correction_matrix=correction_matrix,
        near_factor=near_factor,
        side=side,
        rcip=rcip,
        rcip_context=rcip_context,
        rcip_subdivisions=rcip_subdivisions,
        rcip_save_depth=rcip_save_depth,
        rcip_eval_depth=rcip_eval_depth,
    )
    same_source_target = target is chunker
    rcip_context = _chunkgraph_rcip_eval_context(chunker, operator_options)
    if rcip_context is not None and not same_source_target:
        return _chunkgraph_rcip_eval(
            chunker, kernel, density, target, operator_options, rcip_context, chunkerkerneval
        )

    boundary = _require_chunker(chunker)
    acceleration = _acceleration(operator_options)
    if (
        same_source_target
        and _uses_special_quadrature(kernel, operator_options)
        and acceleration != "fmm"
    ):
        mat_options = dict(operator_options)
        mat_options.pop("acceleration", None)
        from ._assembly import chunkermat

        vals = chunkermat(boundary, kernel, mat_options) @ as_boundary_vector(
            density, name="density"
        )
        opdims = getattr(kernel, "opdims", (1, 1))[0]
        return as_boundary_field_matrix(vals, opdims, boundary.npt, name="kernel values")
    if acceleration == "flam":
        targinfo = PointInfo.from_any(target)
        if _flag(operator_options, "forceadap"):
            smooth_options = dict(operator_options)
            smooth_options.pop("forceadap", None)
            vals = as_boundary_vector(
                _chunkerkerneval_flam(boundary, kernel, density, targinfo, smooth_options),
                name="FLAM values",
            )
            correction_options = dict(operator_options)
            correction_options["recompute_source_normals"] = True
            correction_options.setdefault("transinv", False)
            correction = _target_adaptive_correction_matrix(
                boundary, kernel, targinfo, correction_options
            )
            vals = vals + correction @ as_boundary_vector(density, name="density")
            return as_boundary_field_matrix(
                vals,
                vals.size // targinfo.r.shape[1],
                targinfo.r.shape[1],
                name="corrected FLAM values",
            )
        return _chunkerkerneval_flam(boundary, kernel, density, targinfo, operator_options)
    use_fmm = acceleration == "fmm"
    if use_fmm:
        _require_fmm(kernel)

    srcinfo = PointInfo.from_any(boundary)
    targinfo = PointInfo.from_any(target)
    if same_source_target and _uses_special_quadrature(kernel, operator_options):
        smooth_options = _smooth_fmm_options(operator_options)
        smooth_options["acceleration"] = "fmm"
        vals = as_boundary_vector(
            chunkerkerneval(boundary, kernel, density, boundary, smooth_options),
            name="smooth FMM values",
        )
        vals = vals + _special_correction_matrix(
            boundary, kernel, operator_options
        ) @ as_boundary_vector(
            density,
            name="density",
        )
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected FMM values",
        )
    cormat = operator_options.get("cormat", None)
    if cormat is not None:
        mat = _eval_kernel(kernel, srcinfo, targinfo)
        weighted = _weighted_density(boundary, density)
        density_vec = as_boundary_vector(density, name="density")
        corr_vals = (
            cormat @ density_vec if sparse.issparse(cormat) else np.asarray(cormat) @ density_vec
        )
        vals = mat @ weighted + corr_vals
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected values",
        )
    if _flag(operator_options, "forceadap") and use_fmm:
        weighted = _weighted_density(boundary, density)
        vals = kernel.fmm(_fmm_tol(operator_options), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        vals = as_boundary_vector(vals, name="FMM values")
        correction_options = dict(operator_options)
        correction_options["recompute_source_normals"] = True
        correction_options.setdefault("transinv", False)
        correction = _target_adaptive_correction_matrix(
            boundary, kernel, targinfo, correction_options
        )
        vals = vals + correction @ as_boundary_vector(density, name="density")
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="corrected FMM values",
        )
    if _flag(operator_options, "forceadap"):
        eval_options = dict(operator_options)
        eval_options["recompute_source_normals"] = True
        eval_options.setdefault("transinv", False)
        mat = _target_adaptive_matrix(boundary, kernel, targinfo, eval_options)
        vals = mat @ as_boundary_vector(density, name="density")
        return as_boundary_field_matrix(
            vals,
            vals.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="adaptive values",
        )
    if use_fmm:
        weighted = _weighted_density(boundary, density)
        vals = kernel.fmm(_fmm_tol(operator_options), srcinfo, targinfo, weighted)
        if isinstance(vals, tuple):
            vals = vals[0]
        vals_arr = np.asarray(vals)
        return as_boundary_field_matrix(
            vals_arr,
            vals_arr.size // targinfo.r.shape[1],
            targinfo.r.shape[1],
            name="FMM values",
        )

    mat = _eval_kernel(kernel, srcinfo, targinfo)
    weighted = _weighted_density(boundary, density)
    vals = mat @ weighted
    return as_boundary_field_matrix(
        vals,
        vals.size // targinfo.r.shape[1],
        targinfo.r.shape[1],
        name="kernel values",
    )


def chunkerkernevalmat(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any] | None = None,
    *,
    acceleration: str | None = None,
    quadrature: str | None = None,
    use_panel_quadrature: bool | None = None,
    tol: float | None = None,
    flam_type: str | None = None,
    flam_occupancy: int | None = None,
    rank_or_tol: int | float | None = None,
    proxy: bool | None = None,
    force_adaptive: bool | None = None,
    corrections: bool | None = None,
    near_factor: float | None = None,
    side: str | None = None,
) -> np.ndarray | spmatrix:
    """Build the matrix mapping source densities to target values.

    This is the materialized companion to :func:`chunkerkerneval`. It is useful
    for diagnostics, custom solvers, and adaptive correction matrices. Ordinary
    evaluation matrices are dense; ``corrections=True`` returns the sparse
    near-target correction matrix accepted by legacy ``cormat`` evaluation
    calls. With ``acceleration="fmm"``, this routine materializes the target
    evaluation matrix by applying the kernel FMM evaluator to basis densities.
    """

    same_source_target = target is chunker
    boundary = _require_chunker(chunker)
    operator_options = _normalize_public_options(
        options,
        acceleration=acceleration,
        quadrature=quadrature,
        use_panel_quadrature=use_panel_quadrature,
        tol=tol,
        flam_type=flam_type,
        flam_occupancy=flam_occupancy,
        rank_or_tol=rank_or_tol,
        proxy=proxy,
        force_adaptive=force_adaptive,
        corrections=corrections,
        near_factor=near_factor,
        side=side,
    )
    if _flag(operator_options, "corrections"):
        return _target_adaptive_correction_matrix(
            boundary, kernel, PointInfo.from_any(target), operator_options
        )
    acceleration = _acceleration(operator_options)
    if acceleration == "flam":
        if same_source_target and _uses_special_quadrature(kernel, operator_options):
            from ._assembly import chunkermat

            return np.asarray(chunkermat(boundary, kernel, operator_options))
        if _flag(operator_options, "forceadap"):
            targinfo = PointInfo.from_any(target)
            smooth_options = dict(operator_options)
            smooth_options.pop("forceadap", None)
            return (
                _chunkerkernevalmat_flam(boundary, kernel, targinfo, smooth_options)
                + _target_adaptive_correction_matrix(
                    boundary, kernel, targinfo, operator_options
                ).toarray()
            )
        return _chunkerkernevalmat_flam(boundary, kernel, target, operator_options)
    if acceleration == "fmm":
        _require_fmm(kernel)
        if same_source_target and _uses_special_quadrature(kernel, operator_options):
            from ._assembly import chunkermat

            return np.asarray(chunkermat(boundary, kernel, operator_options))
        return _chunkerkernevalmat_fmm(boundary, kernel, target, operator_options)
    if same_source_target and _uses_special_quadrature(kernel, operator_options):
        from ._assembly import chunkermat

        return chunkermat(boundary, kernel, operator_options)

    srcinfo = PointInfo.from_any(boundary)
    targinfo = PointInfo.from_any(target)
    if _flag(operator_options, "forceadap"):
        return _target_adaptive_matrix(boundary, kernel, targinfo, operator_options)
    mat = _eval_kernel(kernel, srcinfo, targinfo)
    wts = as_boundary_vector(boundary.wts, name="weights")
    if mat.shape[1] == boundary.npt:
        return mat * wts[None, :]
    if mat.shape[1] % boundary.npt != 0:
        raise ValueError("kernel column dimension is incompatible with chunker points")
    opdims_col = mat.shape[1] // boundary.npt
    return mat * np.repeat(wts, opdims_col)[None, :]
