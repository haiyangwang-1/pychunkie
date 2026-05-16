"""Core chunker data class composed from private responsibility mixins."""

from __future__ import annotations

from ._chunker_geometry import ChunkerGeometryMixin
from ._chunker_near import ChunkerNearMixin
from ._chunker_pref import ChunkerPref
from ._chunker_refine import ChunkerRefineMixin
from ._chunker_storage import ChunkerStorageMixin
from ._chunker_transform import ChunkerTransformMixin


class Chunker(
    ChunkerStorageMixin,
    ChunkerGeometryMixin,
    ChunkerNearMixin,
    ChunkerRefineMixin,
    ChunkerTransformMixin,
):
    """Curve divided into Legendre-discretized chunks.

    Arrays are stored as chunk tensors with shape ``dim x k x nch``:

    - ``r`` stores node positions.
    - ``d`` and ``d2`` store first and second derivatives with respect to the
      local panel parameter.
    - ``n`` and ``wts`` store outward normals and physical quadrature weights.
    - ``adj`` stores one-based MATLAB-style neighboring chunk labels, with
      nonpositive entries denoting open ends.

    User code usually calls :func:`chunkerfunc`, :func:`chunkerpoly`,
    :func:`chunkerfit`, or :func:`chunkerpoints` instead of filling this storage
    manually. Flat vectors are adapter formats for solvers and backend
    interfaces; internal geometry should stay in tensor form when possible.
    """

    __array_priority__ = 1000
    lvlrfacdefault = 2.1


__all__ = ["Chunker", "ChunkerPref"]
