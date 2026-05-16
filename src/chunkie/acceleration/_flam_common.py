"""Shared helpers for FLAM callbacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix

from .._layout import as_boundary_vector
from ..geometry import PointInfo
from ..geometry.chunker import Chunker, merge

_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


def _as_chunker(obj: Any) -> Chunker:
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
    raise TypeError("expected a chunker or chunkgraph-like object")


def _chunker_sequence(obj: Any) -> list[Chunker]:
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
        items = list(edges)
        if items and all(isinstance(item, Chunker) for item in items):
            return items
    raise TypeError("expected a chunker sequence or chunkgraph-like object")


def _is_block_kernel_matrix(kernel: Any) -> bool:
    if callable(kernel):
        return False
    try:
        arr = np.asarray(kernel, dtype=object)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.size > 0 and all(callable(item) for item in arr.flat)


def _pointinfo(obj: Any) -> Any:
    return PointInfo.from_any(obj)


def _proxy_info(pr: ArrayLike, ptau: ArrayLike, scale: float, center: np.ndarray) -> Any:
    pxy = np.asarray(pr, dtype=float).reshape(2, -1) * scale + center
    tau = np.asarray(ptau, dtype=float).reshape(2, -1)
    return _new_info(r=pxy, d=tau, d2=np.zeros_like(tau), n=_perp_unit(tau))


def _subset_info(info: Any, indices: np.ndarray) -> Any:
    return _new_info(
        r=info.r[:, indices],
        d=None if info.d is None else info.d[:, indices],
        d2=None if info.d2 is None else info.d2[:, indices],
        n=None if info.n is None else info.n[:, indices],
        data=None if info.data is None else info.data[:, indices],
    )


def _opdims(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | ArrayLike | None,
    *,
    target: Any | None = None,
) -> tuple[int, int]:
    if opdims is not None:
        arr = np.asarray(opdims, dtype=int).reshape(-1)
        if arr.size < 2:
            raise ValueError("opdims must contain two entries")
        return int(arr[0]), int(arr[1])
    kernel_dims = getattr(kernel, "opdims", None)
    if kernel_dims is not None and tuple(kernel_dims) != (0, 0):
        return int(kernel_dims[0]), int(kernel_dims[1])
    source = _subset_info(_pointinfo(chunker), np.array([0], dtype=np.int64))
    target_info = _pointinfo(chunker if target is None else target)
    target_subset = _subset_info(target_info, np.array([0], dtype=np.int64))
    mat = np.asarray(_eval_kernel(kernel, source, target_subset))
    return int(mat.shape[0]), int(mat.shape[1])


def _eval_kernel(
    kernel: Callable[[Any, Any], np.ndarray], source_info: Any, target_info: Any
) -> np.ndarray:
    try:
        eval_method = kernel.eval
    except AttributeError:
        eval_method = None
    if eval_method is not None:
        return eval_method(source_info, target_info)
    return kernel(source_info, target_info)


def _overwrite_sparse(
    out: np.ndarray, rows: np.ndarray, cols: np.ndarray, spmat: spmatrix | None
) -> np.ndarray:
    if spmat is None or rows.size == 0 or cols.size == 0:
        return out
    sub = spmat[np.ix_(rows, cols)].tocoo()
    if sub.nnz:
        out = np.array(out, copy=True)
        out[sub.row, sub.col] = sub.data
    return out


def _as_index_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _validate_points(points: np.ndarray, npt: int, label: str) -> None:
    if np.any(points < 0) or np.any(points >= npt):
        raise IndexError(f"{label} FLAM index is out of range")


def _perp_unit(vec: np.ndarray) -> np.ndarray:
    speed = np.sqrt(np.sum(np.abs(vec) ** 2, axis=0))
    return np.vstack((-vec[1], vec[0])) / speed[None, :]


def _new_info(
    r: np.ndarray,
    d: np.ndarray | None = None,
    d2: np.ndarray | None = None,
    n: np.ndarray | None = None,
    data: np.ndarray | None = None,
) -> Any:
    return PointInfo(r=r, d=d, d2=d2, n=n, data=data)
