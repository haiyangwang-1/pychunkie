"""Structured point data passed to kernels and quadrature routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class PointInfo:
    """Flattened source or target point data passed to kernels.

    ``r`` has shape ``(dim, npt)``. Optional ``d``, ``d2``, and ``n`` fields
    carry tangent derivatives and normals for kernels that need source or
    target normal derivatives. ``data`` is an optional user-defined
    ``(datadim, npt)`` array available to custom kernels.
    """

    r: np.ndarray
    d: np.ndarray | None = None
    d2: np.ndarray | None = None
    n: np.ndarray | None = None
    data: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.r = _as_point_matrix("r", self.r)
        self.d = _optional_point_matrix("d", self.d, self.r.shape[1], self.r.shape[0])
        self.d2 = _optional_point_matrix("d2", self.d2, self.r.shape[1], self.r.shape[0])
        self.n = _optional_point_matrix("n", self.n, self.r.shape[1], self.r.shape[0])
        self.data = _optional_point_matrix("data", self.data, self.r.shape[1], None)

    @classmethod
    def from_chunker(cls, chnkr: Any) -> "PointInfo":
        """Build point data from a chunker or chunkgraph-like object."""

        merged = getattr(chnkr, "merged", None)
        if callable(merged):
            chnkr = merged()
        return cls(
            r=np.asarray(chnkr.r).reshape(int(chnkr.dim), int(chnkr.npt), order="F"),
            d=np.asarray(chnkr.d).reshape(int(chnkr.dim), int(chnkr.npt), order="F"),
            d2=np.asarray(chnkr.d2).reshape(int(chnkr.dim), int(chnkr.npt), order="F"),
            n=np.asarray(chnkr.n).reshape(int(chnkr.dim), int(chnkr.npt), order="F"),
            data=np.asarray(chnkr.data).reshape(int(chnkr.datadim), int(chnkr.npt), order="F")
            if int(getattr(chnkr, "datadim", 0))
            else None,
        )

    @classmethod
    def from_points(cls, points: ArrayLike) -> "PointInfo":
        """Build target point data from an explicit ``(dim, npt)`` array."""

        return cls(r=_as_point_matrix("points", points))

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "PointInfo":
        """Build point data from a MATLAB-style mapping with an ``r`` field."""

        if "r" not in mapping:
            raise ValueError("PointInfo mapping must contain an 'r' field")
        return cls(
            r=_as_point_matrix("r", mapping["r"]),
            d=_optional_mapping_field(mapping, "d"),
            d2=_optional_mapping_field(mapping, "d2"),
            n=_optional_mapping_field(mapping, "n"),
            data=_optional_mapping_field(mapping, "data"),
        )

    @classmethod
    def from_any(cls, obj: Any) -> "PointInfo":
        """Normalize legacy internal inputs to ``PointInfo``.

        Public code should prefer the explicit constructors above. This method
        keeps the operator and parity paths compact while the broader API moves
        away from MATLAB-style dictionaries and raw array coercion.
        """

        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_mapping(obj)
        if _looks_like_chunker(obj):
            return cls.from_chunker(obj)
        return cls.from_points(obj)

    def take(self, indices: ArrayLike) -> "PointInfo":
        """Return a point-info subset."""

        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        return PointInfo(
            r=self.r[:, idx],
            d=None if self.d is None else self.d[:, idx],
            d2=None if self.d2 is None else self.d2[:, idx],
            n=None if self.n is None else self.n[:, idx],
            data=None if self.data is None else self.data[:, idx],
        )


def _looks_like_chunker(obj: Any) -> bool:
    if callable(getattr(obj, "merged", None)):
        return True
    return all(hasattr(obj, name) for name in ("r", "d", "d2", "n", "dim", "npt"))


def _as_point_matrix(name: str, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (dim, npt)")
    return arr


def _optional_point_matrix(
    name: str,
    values: ArrayLike | None,
    npt: int,
    dim: int | None,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = _as_point_matrix(name, values)
    if arr.shape[1] != npt:
        raise ValueError(f"{name} must have the same number of points as r")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"{name} must have the same dimension as r")
    return arr


def _optional_mapping_field(mapping: dict[str, Any], name: str) -> np.ndarray | None:
    if name not in mapping or mapping[name] is None:
        return None
    arr = np.asarray(mapping[name])
    if arr.ndim < 1:
        raise ValueError(f"{name} must be array-like")
    return arr.reshape(arr.shape[0], -1)
