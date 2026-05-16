"""RCIP integration helpers for nonsmooth chunkgraph operators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._layout import as_boundary_field_matrix, as_boundary_vector
from ..geometry.chunker import Chunker
from ..geometry.pointinfo import PointInfo
from ._common import (
    _KERNEL_PROBE_EXCEPTIONS,
    _acceleration,
    _apply_weights_to_density,
    _as_chunker,
    _chunker_from_pointinfo,
    _eval_kernel,
    _flag,
    _is_chunkgraph_like,
    _kernel_opdims,
    _l2scale,
    _merge_pointinfos,
    _shift_pointinfo,
)
from .options import _option_bool
from .types import ChunkerRCIPMatrix, RCIPContext


def _chunkgraph_nonsmooth_vertices(obj: Any, options: dict[str, Any]) -> list[int]:
    if not _is_chunkgraph_like(obj):
        return []
    raw_vertices = options.get("rcip_vertices", options.get("vertices", None))
    if raw_vertices is None:
        candidates = range(np.asarray(obj.verts).shape[1])
    else:
        candidates = _normalize_vertex_indices(raw_vertices, np.asarray(obj.verts).shape[1])
    ignored = set(
        _normalize_vertex_indices(
            options.get("rcip_ignore_vertices", options.get("ignore_vertices", [])),
            np.asarray(obj.verts).shape[1],
        ).tolist()
    )
    out: list[int] = []
    for ivert in candidates:
        if int(ivert) in ignored:
            continue
        edges, _ = obj.vstruc[int(ivert)]
        if np.asarray(edges).size >= 2:
            out.append(int(ivert))
    return out


def _normalize_vertex_indices(vertices: Any, nvert: int) -> np.ndarray:
    arr = np.asarray(vertices, dtype=int).reshape(-1)
    if arr.size and np.max(arr) >= int(nvert):
        if np.min(arr) >= 1 and np.max(arr) <= int(nvert):
            arr = arr - 1
        else:
            raise ValueError("vertex index out of range")
    if np.any(arr < 0) or np.any(arr >= int(nvert)):
        raise ValueError("vertex index out of range")
    return arr


def _rcip_option_enabled(options: dict[str, Any]) -> bool:
    value = options.get("rcip", True)
    if isinstance(value, RCIPContext):
        return True
    return _option_bool(value)


def _chunkgraph_rcip_mat_enabled(
    obj: Any, kernel: Callable[[Any, Any], np.ndarray], options: dict[str, Any]
) -> bool:
    if not _rcip_option_enabled(options):
        return False
    if not _is_chunkgraph_like(obj):
        return False
    if _acceleration(options) != "dense" or _l2scale(options):
        return False
    if not _is_rcip_second_kind_kernel(kernel):
        return False
    merged = _as_chunker(obj)
    if merged is None:
        return False
    try:
        if _kernel_opdims(merged, kernel) != (1, 1):
            return False
    except _KERNEL_PROBE_EXCEPTIONS:
        return False
    return bool(_chunkgraph_nonsmooth_vertices(obj, options))


def _is_rcip_second_kind_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    name = str(getattr(kernel, "name", "")).lower()
    typ = str(getattr(kernel, "type", "")).lower()
    if name not in {"laplace", "helmholtz"}:
        return False
    return typ in {
        "d",
        "double",
        "sp",
        "sprime",
    }


def _chunkgraph_rcip_mat(
    cg: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any],
    mat_builder: Callable[[Any, Callable[[Any, Any], np.ndarray], dict[str, Any]], Any],
) -> tuple[ChunkerRCIPMatrix, RCIPContext]:
    from .. import rcip

    merged = cg.merged()
    base_options = _strip_rcip_options(options)
    mat = np.asarray(mat_builder(merged, kernel, base_options)).copy()
    nsub = _rcip_nsub(options)
    savedepth = _rcip_savedepth(options, nsub)
    saved: list[Any] = []

    for ivert in _chunkgraph_nonsmooth_vertices(cg, options):
        edges, signs = cg.vstruc[ivert]
        edges = np.asarray(edges, dtype=int).reshape(-1)
        signs = np.asarray(signs, dtype=int).reshape(-1)
        isstart = signs < 0
        edge_chunks = _rcip_corner_edge_chunks(cg, edges, signs)
        pbc, pwbc, star_l, circ_l, star_s, circ_s, *_ = rcip.setup(cg.k, 1, edges.size, isstart)
        rmat, rcipsav = rcip.Rcompchunk(
            cg.echnks,
            edge_chunks,
            kernel,
            1,
            cg.verts[:, ivert],
            Pbc=pbc,
            PWbc=pwbc,
            starL=star_l,
            circL=circ_l,
            starS=star_s,
            circS=circ_s,
            options={"nsub": nsub, "rcip_savedepth": savedepth},
        )
        starind = _rcip_corner_star_indices(cg, edges, signs)
        rcipsav.starind = starind
        replacement = np.linalg.inv(rmat) - np.eye(rmat.shape[0], dtype=rmat.dtype)
        mat[np.ix_(starind, starind)] = replacement
        saved.append(rcipsav)

    context = RCIPContext(cg, kernel, saved, nsub, savedepth, 1)
    try:
        cg._last_rcip_context = context
    except AttributeError:
        pass
    return ChunkerRCIPMatrix(mat, context), context


def _rcip_nsub(options: dict[str, Any]) -> int:
    value = options.get("nsub", options.get("rcip_nsub", options.get("nsub_or_tol", 20)))
    value_float = float(value)
    if value_float <= 0:
        raise ValueError("RCIP nsub must be positive")
    if value_float < 1.0:
        return max(int(np.ceil(np.log2(1.0 / value_float**2))), 20)
    return int(value_float)


def _rcip_savedepth(options: dict[str, Any], nsub: int) -> int:
    savedepth = int(options.get("rcip_savedepth", options.get("save_depth", nsub)))
    return min(max(savedepth, 0), int(nsub))


def _strip_rcip_options(options: dict[str, Any]) -> dict[str, Any]:
    out = dict(options)
    for key in (
        "rcip",
        "return_rcip",
        "rcip_context",
        "rcip_saved",
        "rcip_vertices",
        "vertices",
        "rcip_ignore_vertices",
        "ignore_vertices",
        "nsub",
        "rcip_nsub",
        "nsub_or_tol",
        "rcip_savedepth",
        "save_depth",
        "rcip_eval_depth",
        "rcip_ndepth",
    ):
        out.pop(key, None)
    return out


def _rcip_corner_star_indices(cg: Any, edges: np.ndarray, signs: np.ndarray) -> np.ndarray:
    starts = np.cumsum([0] + [edge.npt for edge in cg.echnks])
    width = 2 * int(cg.k)
    out: list[np.ndarray] = []
    for edge, sign in zip(np.asarray(edges, dtype=int), np.asarray(signs, dtype=int), strict=True):
        lo = int(starts[int(edge)])
        hi = int(starts[int(edge) + 1])
        if hi - lo < width:
            raise ValueError("each RCIP edge needs at least two coarse panels")
        out.append(np.arange(lo, lo + width) if int(sign) < 0 else np.arange(hi - width, hi))
    return np.concatenate(out)


def _rcip_corner_edge_chunks(cg: Any, edges: np.ndarray, signs: np.ndarray) -> np.ndarray:
    edge_arr = np.asarray(edges, dtype=int).reshape(-1)
    sign_arr = np.asarray(signs, dtype=int).reshape(-1)
    out = np.zeros((2, edge_arr.size), dtype=int)
    out[0] = edge_arr
    for idx, (edge, sign) in enumerate(zip(edge_arr, sign_arr, strict=True)):
        out[1, idx] = 0 if int(sign) < 0 else cg.echnks[int(edge)].nch - 1
    return out


def _chunkgraph_rcip_eval_context(obj: Any, options: dict[str, Any]) -> RCIPContext | None:
    value = options.get("rcip", None)
    if value is not None and not isinstance(value, RCIPContext) and not _option_bool(value):
        return None
    if isinstance(value, RCIPContext):
        return value
    context = options.get("rcip_context", None)
    if context is not None:
        return context
    if "rcip_saved" in options:
        return RCIPContext(
            obj,
            options.get("rcip_system_kernel", lambda s, t: np.zeros((t.r.shape[1], s.r.shape[1]))),
            list(options["rcip_saved"]),
            _rcip_nsub(options),
            _rcip_savedepth(options, _rcip_nsub(options)),
            1,
        )
    if not _is_chunkgraph_like(obj):
        return None
    return getattr(obj, "_last_rcip_context", None)


def _chunkgraph_rcip_eval(
    cg: Any,
    kernel: Callable[[Any, Any], np.ndarray],
    density: ArrayLike,
    target: Chunker | dict[str, Any] | ArrayLike | PointInfo,
    options: dict[str, Any],
    context: RCIPContext,
    eval_func: Callable[
        [Any, Callable[[Any, Any], np.ndarray], ArrayLike, Any, dict[str, Any]],
        np.ndarray,
    ],
) -> np.ndarray:
    from .. import rcip

    if not context.saved:
        return eval_func(cg.merged(), kernel, density, target, _strip_rcip_options(options))

    dens_vec = as_boundary_vector(density, name="density")
    merged = cg.merged()
    if dens_vec.size != merged.npt * int(context.ndim):
        raise ValueError("density has incompatible size for RCIP context")

    rho_coarse = dens_vec.copy()
    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rho_coarse[starind] = 0.0

    eval_options = _strip_rcip_options(options)
    targinfo = PointInfo.from_any(target)
    vals = as_boundary_vector(
        eval_func(merged, kernel, rho_coarse, targinfo, eval_options),
        name="RCIP coarse values",
    )
    ndepth = int(options.get("rcip_eval_depth", options.get("rcip_ndepth", context.savedepth)))

    for rcipsav in context.saved:
        starind = np.asarray(rcipsav.starind, dtype=int).reshape(-1)
        rhocells, srcinfos, wtscells = rcip.rhohatInterp(dens_vec[starind], rcipsav, ndepth)
        local_targets = _shift_pointinfo(targinfo, np.asarray(rcipsav.ctr)[:, [0]])
        if _flag(eval_options, "forceadap"):
            for rho, srcinfo, wts in zip(rhocells, srcinfos, wtscells, strict=True):
                if srcinfo is None or wts is None:
                    continue
                local_chunker = _chunker_from_pointinfo(srcinfo, wts, rcipsav.k)
                vals = vals + as_boundary_vector(
                    eval_func(local_chunker, kernel, rho, local_targets, eval_options),
                    name="RCIP local values",
                )
        else:
            srcinfo = _merge_pointinfos([src for src in srcinfos if src is not None])
            if srcinfo is None:
                continue
            rho_local = np.concatenate(
                [as_boundary_vector(rho, name="RCIP density") for rho in rhocells]
            )
            wts_local = np.concatenate(
                [
                    as_boundary_vector(wts, name="RCIP weights")
                    for wts in wtscells
                    if wts is not None
                ]
            )
            local_mat = _eval_kernel(kernel, srcinfo, local_targets)
            vals = vals + local_mat @ _apply_weights_to_density(rho_local, wts_local)

    return as_boundary_field_matrix(
        vals,
        vals.size // targinfo.r.shape[1],
        targinfo.r.shape[1],
        name="RCIP values",
    )
