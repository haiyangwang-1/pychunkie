"""Chunker construction from panel values and component merging."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ._chunker_class import Chunker
from ._chunker_options import _legacy_options, _set_option
from ._chunker_pref import ChunkerPref


def chunkerpoints(
    source: ArrayLike | dict[str, ArrayLike],
    options: dict[str, Any] | None = None,
    *,
    closed: bool | None = None,
) -> Chunker:
    """Create a chunker from panel node values.

    ``source`` may be either a ``(dim, k, nch)`` position array or a mapping
    with ``r`` and optional matching ``d``/``d2`` arrays, mirroring MATLAB
    ``chunkerpoints``.
    """

    geometry_options = _legacy_options(options, "chunkerpoints options")
    _set_option(geometry_options, "ifclosed", closed)
    d_arr = None
    d2_arr = None
    if isinstance(source, dict):
        if "r" not in source:
            raise ValueError("missing field r in chunkerpoints")
        r_arr = np.asarray(source["r"])
        if "d" in source and np.asarray(source["d"]).shape == r_arr.shape:
            d_arr = np.asarray(source["d"], dtype=r_arr.dtype)
        if "d2" in source and np.asarray(source["d2"]).shape == r_arr.shape:
            d2_arr = np.asarray(source["d2"], dtype=r_arr.dtype)
    else:
        r_arr = np.asarray(source)

    if r_arr.ndim != 3:
        raise ValueError("chunkerpoints expects r with shape (dim, k, nch)")
    dim, k, nch = r_arr.shape
    if nch <= 0:
        raise ValueError("chunkerpoints requires at least one chunk")

    pref = ChunkerPref(dim=dim, k=k, nchstor=nch, nchmax=max(nch, 1))
    chnkr = Chunker(pref).addchunk(nch)
    dmat = lege.dermat(k)
    chnkr.r = r_arr
    if d_arr is None:
        chnkr.d = np.einsum("dkn,jk->djn", r_arr, dmat)
    else:
        chnkr.d = d_arr
    if d2_arr is None:
        chnkr.d2 = np.einsum("dkn,jk->djn", chnkr.d, dmat)
    else:
        chnkr.d2 = d2_arr

    adjs = np.zeros((2, nch), dtype=int)
    adjs[0] = np.arange(0, nch)
    adjs[1] = np.arange(2, nch + 2)
    if bool(geometry_options.get("ifclosed", True)):
        adjs[0, 0] = nch
        adjs[1, -1] = 1
    else:
        adjs[0, 0] = -1
        adjs[1, -1] = -1
    chnkr.adj = adjs
    chnkr.recompute_geometry()
    return chnkr


def merge(
    chnkrs: ArrayLike | list[Chunker] | tuple[Chunker, ...],
    pref: ChunkerPref | dict[str, Any] | None = None,
) -> Chunker:
    """Combine chunkers of the same dimension and order.

    The merged chunker keeps each input component's local adjacency, shifting
    positive neighbor labels by the accumulated chunk offset. This is the
    geometry path used when scalar kernels are applied to a ``ChunkGraph`` or
    to an explicit list of chunkers.
    """

    if isinstance(chnkrs, Chunker):
        items = [chnkrs]
    elif isinstance(chnkrs, (list, tuple)):
        items = list(chnkrs)
    else:
        items = list(np.ravel(chnkrs))
    if not items:
        return Chunker(pref)
    if not all(isinstance(item, Chunker) for item in items):
        raise TypeError("input must contain chunker objects")

    first = items[0]
    total_nch = sum(item.nch for item in items)
    p = ChunkerPref.from_any(pref)
    p = ChunkerPref(
        nchmax=max(p.nchmax, total_nch),
        k=first.k,
        dim=first.dim,
        nchstor=max(p.nchstor, total_nch),
        verttol=p.verttol,
    )
    out = Chunker(p, first.tstor, first.wstor).addchunk(total_nch)

    offset = 0
    for item in items:
        if item.dim != first.dim or item.k != first.k:
            raise ValueError("chunkers to merge must have the same dimension and order")
        sl = slice(offset, offset + item.nch)
        out.rstor[:, :, sl] = item.r
        out.dstor[:, :, sl] = item.d
        out.d2stor[:, :, sl] = item.d2
        out.nstor[:, :, sl] = item.n
        out.wtsstor[:, sl] = item.wts
        adj = item.adj.copy()
        adj[adj > 0] += offset
        out.adjstor[:, sl] = adj
        offset += item.nch

    max_data = max((item.datadim for item in items), default=0)
    if max_data > 0:
        out.makedatarows(max_data)
        offset = 0
        for item in items:
            if item.hasdata and item.datadim > 0:
                out.datastor[: item.datadim, :, offset : offset + item.nch] = item.data
            offset += item.nch
    return out
