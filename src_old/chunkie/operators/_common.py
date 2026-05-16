"""Shared helpers for operator assembly and evaluation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._layout import (
    as_boundary_chunk_tensor,
    as_boundary_point_matrix,
    as_boundary_vector,
    as_boundary_weight_matrix,
    density_matmul_argument,
    weighted_density_for_boundary,
)
from ..geometry.chunker import Chunker, ChunkerPref, merge
from ..geometry.pointinfo import PointInfo
from .options import _OperatorOptions, _option_bool

_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


def _as_chunker(obj: Any) -> Chunker | None:
    if isinstance(obj, Chunker):
        return obj
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if items and all(isinstance(item, Chunker) for item in items):
            return merge(items)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        items = list(np.ravel(obj))
        if items and all(isinstance(item, Chunker) for item in items):
            return merge(items)
    merged = getattr(obj, "merged", None)
    if callable(merged):
        out = merged()
        if isinstance(out, Chunker):
            return out
    return None


def _as_chunker_sequence(obj: Any) -> list[Chunker] | None:
    if isinstance(obj, Chunker):
        return [obj]
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        items = list(as_boundary_vector(obj, name="chunker sequence"))
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    edges = getattr(obj, "echnks", None)
    if edges is not None:
        edge_chunkers = list(edges)
        if edge_chunkers and all(isinstance(item, Chunker) for item in edge_chunkers):
            return edge_chunkers
    return None


def _require_chunker(obj: Any) -> Chunker:
    out = _as_chunker(obj)
    if out is None:
        raise TypeError("expected a chunker or chunkgraph-like object")
    return out


def _is_chunkgraph_like(obj: Any) -> bool:
    return (
        hasattr(obj, "echnks")
        and hasattr(obj, "vstruc")
        and hasattr(obj, "verts")
        and callable(getattr(obj, "merged", None))
    )


def _merge_pointinfos(infos: list[PointInfo]) -> PointInfo | None:
    if not infos:
        return None
    return PointInfo(
        r=np.column_stack([info.r for info in infos]),
        d=None
        if any(info.d is None for info in infos)
        else np.column_stack([info.d for info in infos]),
        d2=None
        if any(info.d2 is None for info in infos)
        else np.column_stack([info.d2 for info in infos]),
        n=None
        if any(info.n is None for info in infos)
        else np.column_stack([info.n for info in infos]),
        data=None
        if any(info.data is None for info in infos)
        else np.column_stack([info.data for info in infos]),
    )


def _shift_pointinfo(info: PointInfo, center: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r - np.asarray(center).reshape(info.r.shape[0], 1),
        d=info.d,
        d2=info.d2,
        n=info.n,
        data=info.data,
    )


def _chunker_from_pointinfo(info: PointInfo, wts: np.ndarray, quadrature_order: int) -> Chunker:
    npt = int(info.r.shape[1])
    quadrature_order = int(quadrature_order)
    if npt % quadrature_order != 0:
        raise ValueError("RCIP local source points must be whole chunks")
    nch = npt // quadrature_order
    out = Chunker(ChunkerPref(k=quadrature_order, dim=info.r.shape[0], nchstor=nch, nchmax=nch))
    out.addchunk(nch)
    out.r = as_boundary_chunk_tensor(
        info.r, info.r.shape[0], quadrature_order, nch, name="positions"
    )
    if info.d is not None:
        out.d = as_boundary_chunk_tensor(
            info.d, info.r.shape[0], quadrature_order, nch, name="derivatives"
        )
    if info.d2 is not None:
        out.d2 = as_boundary_chunk_tensor(
            info.d2, info.r.shape[0], quadrature_order, nch, name="second derivatives"
        )
    if info.n is not None:
        out.n = as_boundary_chunk_tensor(
            info.n, info.r.shape[0], quadrature_order, nch, name="normals"
        )
    out.wts = as_boundary_weight_matrix(wts, quadrature_order, nch, name="weights")
    if nch:
        out.adj = np.vstack(
            (
                np.concatenate(([-1], np.arange(1, nch, dtype=int))),
                np.concatenate((np.arange(2, nch + 1, dtype=int), [-1])),
            )
        )
    return out


def _apply_weights_to_density(rho: np.ndarray, wts: np.ndarray) -> np.ndarray:
    if rho.size == wts.size:
        return rho * wts
    if rho.size % wts.size != 0:
        raise ValueError("RCIP local density and weights are incompatible")
    return rho * np.repeat(wts, rho.size // wts.size)


def _weighted_density(chunker: Chunker, density: ArrayLike) -> np.ndarray:
    return weighted_density_for_boundary(density, chunker.wts, chunker.npt)


def _density_matmul_arg(ncols: int, density: ArrayLike) -> np.ndarray:
    return density_matmul_argument(density, ncols)


def _eval_kernel(
    kernel: Callable[[Any, Any], np.ndarray], source_info: PointInfo, target_info: PointInfo
) -> np.ndarray:
    if hasattr(kernel, "eval") and kernel.eval is not None:
        return kernel.eval(source_info, target_info)
    return kernel(source_info, target_info)


def _uses_special_quadrature(
    kernel: Callable[[Any, Any], np.ndarray],
    options: dict[str, Any] | _OperatorOptions | None,
) -> bool:
    return _OperatorOptions.from_any(options).uses_special_quadrature(kernel)


def _special_quadrature_type(
    kernel: Callable[[Any, Any], np.ndarray], options: dict[str, Any] | _OperatorOptions
) -> str:
    return _OperatorOptions.from_any(options).special_quadrature_type(kernel)


def _is_laplace_double_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() == "laplace" and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "d",
        "double",
        "double layer",
    }


def _is_laplace_sprime_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() == "laplace" and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "normal derivative of single layer",
        "sp",
        "sprime",
    }


def _is_stokes_strac_kernel(kernel: Callable[[Any, Any], np.ndarray]) -> bool:
    return str(getattr(kernel, "name", "")).lower() in {"stokes", "stok"} and str(
        getattr(kernel, "type", "")
    ).lower() in {
        "strac",
        "straction",
        "traction of single layer",
    }


def _apply_laplace_double_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_double_kernel(kernel) or mat.shape != (chunker.npt, chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chunker.d**2, axis=0))
    curvature = (chunker.d[0] * chunker.d2[1] - chunker.d[1] * chunker.d2[0]) / speed**3
    diag = (
        scale
        * (-as_boundary_vector(curvature, name="curvature") / (4.0 * np.pi))
        * as_boundary_vector(chunker.wts, name="weights")
    )
    np.fill_diagonal(mat, diag)
    return mat


def _apply_laplace_sprime_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_laplace_sprime_kernel(kernel) or mat.shape != (chunker.npt, chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    speed = np.sqrt(np.sum(chunker.d**2, axis=0))
    curvature = (chunker.d[0] * chunker.d2[1] - chunker.d[1] * chunker.d2[0]) / speed**3
    diag = (
        scale
        * (-as_boundary_vector(curvature, name="curvature") / (4.0 * np.pi))
        * as_boundary_vector(chunker.wts, name="weights")
    )
    np.fill_diagonal(mat, diag)
    return mat


def _apply_stokes_strac_self_limit(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    mat: np.ndarray,
) -> np.ndarray:
    if not _is_stokes_strac_kernel(kernel) or mat.shape != (2 * chunker.npt, 2 * chunker.npt):
        return mat
    scale = getattr(kernel, "params", {}).get("_scale", 1.0)
    d = as_boundary_point_matrix(chunker.d, chunker.dim, chunker.npt, name="derivatives")
    speed = np.sqrt(np.sum(d**2, axis=0))
    tangents = d / speed[None, :]
    curvature = as_boundary_vector(chunker.signed_curvature(), name="curvature")
    weights = as_boundary_vector(chunker.wts, name="weights")
    for inode in range(chunker.npt):
        block = (
            scale
            * (-curvature[inode] / (2.0 * np.pi))
            * np.outer(tangents[:, inode], tangents[:, inode])
            * weights[inode]
        )
        rows = slice(2 * inode, 2 * inode + 2)
        mat[rows, rows] = block
    return mat


def _acceleration(options: dict[str, Any] | _OperatorOptions) -> str:
    return _OperatorOptions.from_any(options).acceleration


def _flag(options: dict[str, Any] | _OperatorOptions, name: str, default: bool = False) -> bool:
    return _OperatorOptions.from_any(options).flag(name, default)


def _l2scale(options: dict[str, Any] | _OperatorOptions) -> bool:
    return _OperatorOptions.from_any(options).l2scale


def _flamtype(options: dict[str, Any] | _OperatorOptions) -> str:
    return _OperatorOptions.from_any(options).flamtype


def _flam_occ(options: dict[str, Any] | _OperatorOptions) -> int:
    return _OperatorOptions.from_any(options).flam_occ


def _flam_rank_or_tol(options: dict[str, Any] | _OperatorOptions) -> int | float:
    return _OperatorOptions.from_any(options).flam_rank_or_tol


def _flam_options(
    options: dict[str, Any] | _OperatorOptions, *, store_default: str | None = None
) -> dict[str, Any]:
    return _OperatorOptions.from_any(options).flam_options(store_default=store_default)


def _fmm_tol(options: dict[str, Any] | _OperatorOptions, default: float = 1.0e-12) -> float:
    return _OperatorOptions.from_any(options).fmm_tol(default)


def _pquad_enabled(options: dict[str, Any] | _OperatorOptions) -> bool:
    raw = _OperatorOptions.from_any(options).raw
    if "forcepquad" in raw:
        return _option_bool(raw["forcepquad"])
    return _option_bool(raw.get("usepquad", True))


def _pquad_side(options: dict[str, Any] | _OperatorOptions) -> str | None:
    raw = _OperatorOptions.from_any(options).raw
    if "side" not in raw or raw["side"] is None:
        return None
    side = str(raw["side"]).lower()
    if side not in {"i", "e"}:
        raise ValueError("side must be 'i' or 'e'")
    return side


def _boundary_pquad_enabled(options: dict[str, Any] | _OperatorOptions) -> bool:
    raw = _OperatorOptions.from_any(options).raw
    if "forcepquad" in raw or "usepquad" in raw:
        return _pquad_enabled(options)
    return _pquad_side(options) is not None


def _require_fmm(kernel: Callable[[Any, Any], np.ndarray]) -> None:
    fmm = getattr(kernel, "fmm", None)
    if fmm is None:
        raise NotImplementedError("FMM acceleration requested, but the kernel has no FMM evaluator")
    if bool(getattr(fmm, "_chunkie_direct_fmm_fallback", False)):
        name = getattr(kernel, "name", "custom")
        typ = getattr(kernel, "type", "")
        label = f"{name} {typ}".strip()
        warnings.warn(
            f"FMM acceleration requested for {label}, but no accelerated FMM evaluator is available; "
            "using a direct dense matrix-vector fallback",
            RuntimeWarning,
            stacklevel=3,
        )


def _require_pyflam():
    try:
        import pyflam
    except (
        ImportError,
        OSError,
    ) as exc:  # pragma: no cover - dependency is required in packaged installs.
        raise ImportError("FLAM acceleration requires the pyflam package") from exc
    return pyflam


def _dval_vector(dval: ArrayLike | float | complex, size: int) -> np.ndarray:
    arr = np.asarray(dval)
    if arr.size == 1:
        return np.full(size, arr.reshape(-1)[0], dtype=arr.dtype)
    vec = as_boundary_vector(arr, name="dval")
    if vec.size != size:
        raise ValueError(f"dval must be scalar or length {size}")
    return vec


def _add_diagonal_shift(
    out: np.ndarray, rows: np.ndarray, cols: np.ndarray, dval: np.ndarray
) -> np.ndarray:
    rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows_arr.size == 0 or cols_arr.size == 0:
        return out
    col_positions: dict[int, list[int]] = {}
    for pos, col in enumerate(cols_arr):
        col_positions.setdefault(int(col), []).append(pos)
    touched = False
    shifted = out
    for row_pos, row in enumerate(rows_arr):
        positions = col_positions.get(int(row))
        if positions is None:
            continue
        if not touched:
            shifted = np.array(out, dtype=np.result_type(out.dtype, dval.dtype), copy=True)
            touched = True
        shifted[row_pos, positions] += dval[int(row)]
    return shifted


def _kernel_opdims(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    target_info: PointInfo | None = None,
) -> tuple[int, int]:
    opdims = getattr(kernel, "opdims", None)
    if opdims is not None and tuple(opdims) != (0, 0):
        return int(opdims[0]), int(opdims[1])
    source = _pointinfo_node(chunker, 0)
    target = (
        _pointinfo_first(target_info)
        if target_info is not None
        else _pointinfo_node(chunker, 1 if chunker.npt > 1 else 0)
    )
    mat = _eval_kernel(kernel, source, target)
    return int(mat.shape[0]), int(mat.shape[1])


def _operator_dtype(chunker: Chunker, kernel: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    return _probe_kernel_dtype(
        kernel,
        _pointinfo_node(chunker, 0),
        _pointinfo_node(chunker, 1 if chunker.npt > 1 else 0),
    )


def _probe_kernel_dtype(
    kernel: Callable[[Any, Any], np.ndarray], source: PointInfo, target: PointInfo
) -> np.dtype:
    try:
        return np.asarray(_eval_kernel(kernel, source, target)).dtype
    except _KERNEL_PROBE_EXCEPTIONS:
        return np.dtype(float)


def _pointinfo_node(chunker: Chunker, inode: int) -> PointInfo:
    source = PointInfo.from_any(chunker)
    idx = int(inode)
    return _pointinfo_take(source, np.array([idx], dtype=np.int64))


def _pointinfo_first(info: PointInfo) -> PointInfo:
    return _pointinfo_take(info, np.array([0], dtype=np.int64))


def _pointinfo_take(info: PointInfo, indices: np.ndarray) -> PointInfo:
    return PointInfo(
        r=info.r[:, indices],
        d=info.d[:, indices] if info.d is not None else None,
        d2=info.d2[:, indices] if info.d2 is not None else None,
        n=info.n[:, indices] if info.n is not None else None,
        data=info.data[:, indices] if info.data is not None else None,
    )


def _target_rows(indices: np.ndarray, op0: int) -> np.ndarray:
    return (indices[:, None] * int(op0) + np.arange(int(op0))[None, :]).reshape(-1)
