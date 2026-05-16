"""RCIP compression matrix construction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from chunkie._layout import as_boundary_point_matrix

from ..geometry.chunker import Chunker, ChunkerPref, merge
from ._local import _rcip_edge_records, _shiftedcurve, chunkerfunclocal, shiftedlegbasismats
from .algebra import SchurBana, setup
from .types import RCIPSaved


def Rcompchunk(
    chunker: list[Chunker] | tuple[Chunker, ...] | Chunker,
    edge_chunks: ArrayLike,
    kernel: Any,
    dimension: int,
    vertex: ArrayLike,
    Pbc: ArrayLike | None = None,
    PWbc: ArrayLike | None = None,
    starL: ArrayLike | None = None,
    circL: ArrayLike | None = None,
    starS: ArrayLike | None = None,
    circS: ArrayLike | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, RCIPSaved]:
    """Compute the RCIP compression matrix for chunks adjacent to a corner."""

    options = {} if options is None else dict(options)
    chunks = [chunker] if isinstance(chunker, Chunker) else list(chunker)
    if not chunks:
        raise ValueError("Rcompchunk requires at least one edge chunker")
    k = chunks[0].k
    dim = chunks[0].dim
    ndim = int(dimension)
    vert = np.asarray(vertex, dtype=float).reshape(dim)
    nsub = int(options.get("nsub", options.get("rcip_nsub", 0)))
    if nsub <= 0:
        edge_chunk_indices = np.asarray(edge_chunks, dtype=int)
        nedge0 = int(
            edge_chunk_indices.size
            if edge_chunk_indices.ndim == 1
            else edge_chunk_indices.shape[-1]
        )
        pbc, pwbc, sl, cl, ss, cs, ilist, sl1, cl1 = setup(
            k, ndim, nedge0, np.ones(nedge0, dtype=bool)
        )
        size = 2 * nedge0 * k * ndim
        rmat = np.eye(size)
        saved = RCIPSaved(
            k=k,
            ndim=ndim,
            nedge=nedge0,
            Pbc=pbc,
            PWbc=pwbc,
            starL=sl,
            circL=cl,
            starS=ss,
            circS=cs,
            ilist=ilist,
            starL1=sl1,
            circL1=cl1,
            nsub=0,
            savedepth=0,
            R=[rmat],
            MAT=[],
            chnkrlocals=[],
            starind=np.arange(size, dtype=int),
        )
        return rmat, saved

    sbclmat, sbcrmat, lvmat, rvmat, u = shiftedlegbasismats(k)
    records = _rcip_edge_records(chunks, edge_chunks, vert, sbclmat, sbcrmat, lvmat, rvmat, u)
    nedge = len(records)
    isstart = np.array([rec["ileftright"] == 1 for rec in records], dtype=bool)

    if (
        Pbc is None
        or PWbc is None
        or starL is None
        or circL is None
        or starS is None
        or circS is None
    ):
        pbc, pwbc, sl, cl, ss, cs, ilist, sl1, cl1 = setup(k, ndim, nedge, isstart)
    else:
        pbc = np.asarray(Pbc)
        pwbc = np.asarray(PWbc)
        sl = np.asarray(starL, dtype=int)
        cl = np.asarray(circL, dtype=int)
        ss = np.asarray(starS, dtype=int)
        cs = np.asarray(circS, dtype=int)
        ilist = np.vstack((np.zeros(nedge, dtype=int), np.ones(nedge, dtype=int)))
        sl1 = sl // int(ndim)
        cl1 = cl // int(ndim)

    size = 2 * nedge * k * int(ndim)
    savedepth = int(options.get("rcip_savedepth", options.get("save_depth", 10)))
    savedepth = min(max(savedepth, 0), nsub)
    nsys = 3 * k * nedge * ndim
    pref = ChunkerPref(k=k, dim=dim, nchstor=5, nchmax=5)
    rmat: np.ndarray | None = None
    saved_R: list[np.ndarray | None] = [None] * (nsub + 1)
    saved_MAT: list[np.ndarray | None] = [None] * nsub
    saved_locals: list[Chunker | None] = [None] * nsub

    for level in range(1, nsub + 1):
        h = np.ones(nedge) / (2 ** (nsub - level))
        locals_: list[Chunker] = []
        for iedge, rec in enumerate(records):
            if rec["ileftright"] == -1:
                ts = (
                    np.array([0.0, 0.5, 1.0]) * h[iedge]
                    if level == nsub
                    else np.array([0.0, 0.5, 1.0, 2.0]) * h[iedge]
                )
            else:
                ts = (
                    -np.array([1.0, 0.5, 0.0]) * h[iedge]
                    if level == nsub
                    else -np.array([2.0, 1.0, 0.5, 0.0]) * h[iedge]
                )
            locals_.append(
                chunkerfunclocal(
                    lambda t, rec=rec: _shiftedcurve(
                        t,
                        rec["rcs"],
                        rec["dcs"],
                        rec["dscal"],
                        rec["d2cs"],
                        rec["d2scal"],
                        rec["ileftright"],
                    ),
                    ts,
                    pref,
                    chunks[0].tstor,
                    chunks[0].wstor,
                )
            )

        if level == nsub:
            for iedge, rec in enumerate(records):
                local = locals_[iedge]
                next_chunk = rec["nextchunk"]
                source_chunker = chunks[rec["chunker"]]
                old_nch = local.nch
                local.addchunk(1)
                local.rstor[:, :, old_nch] = (
                    source_chunker.r[:, :, next_chunk] - rec["ctr"][:, None]
                )
                local.dstor[:, :, old_nch] = source_chunker.d[:, :, next_chunk]
                local.d2stor[:, :, old_nch] = source_chunker.d2[:, :, next_chunk]
                if rec["ileftright"] == -1:
                    local.adjstor[0, old_nch] = old_nch
                    local.adjstor[1, old_nch] = -1
                    local.adjstor[1, old_nch - 1] = old_nch + 1
                else:
                    local.adjstor[0, old_nch] = -1
                    local.adjstor[1, old_nch] = 1
                    local.adjstor[0, 0] = old_nch + 1
                    local = local.sort()[0]
                local.recompute_geometry()
                locals_[iedge] = local

        ilistl = None if level == 1 else ilist
        mat = np.eye(nsys) + _local_chunkermat(locals_, kernel, ndim, ilistl)
        if level == 1:
            rmat = np.linalg.inv(mat[np.ix_(sl, sl)])
            if level >= nsub - savedepth + 1:
                saved_R[0] = rmat
        if savedepth < nsub and level == nsub - savedepth + 1:
            saved_R[level - 1] = rmat
        rmat = SchurBana(pbc, pwbc, mat, rmat, sl, cl, ss, cs)
        if level >= nsub - savedepth + 1:
            saved_R[level] = rmat
            saved_MAT[level - 1] = mat[np.ix_(sl, cl)]
            saved_locals[level - 1] = merge(locals_)

    assert rmat is not None
    saved = RCIPSaved(
        k=k,
        ndim=ndim,
        nedge=nedge,
        Pbc=pbc,
        PWbc=pwbc,
        starL=sl,
        circL=cl,
        starS=ss,
        circS=cs,
        ilist=ilist,
        starL1=sl1,
        circL1=cl1,
        nsub=nsub,
        savedepth=savedepth,
        R=[item for item in saved_R if item is not None],
        MAT=[item for item in saved_MAT if item is not None],
        chnkrlocals=[item for item in saved_locals if item is not None],
        starind=np.arange(size, dtype=int),
        ctr=np.column_stack([rec["ctr"] for rec in records]),
        rcs=np.stack([rec["rcs"] for rec in records], axis=2),
        dcs=np.stack([rec["dcs"] for rec in records], axis=2),
        d2cs=np.stack([rec["d2cs"] for rec in records], axis=2),
        dscal=np.array([rec["dscal"] for rec in records]),
        d2scal=np.array([rec["d2scal"] for rec in records]),
        ileftright=np.array([rec["ileftright"] for rec in records], dtype=int),
        glxs=chunks[0].tstor.copy(),
        glws=chunks[0].wstor.copy(),
    )
    return rmat, saved


def _local_chunkermat(
    chunks: list[Chunker],
    kernel: Any,
    dimension: int,
    ilist: np.ndarray | None = None,
) -> np.ndarray:
    from ..operators import chunkerkernevalmat, chunkermat
    from ..quadrature import ggq as quadggq
    from ..quadrature import native as quadnative

    starts = np.cumsum([0] + [ch.npt * int(dimension) for ch in chunks])
    out = np.zeros((starts[-1], starts[-1]))
    for itarg, target in enumerate(chunks):
        rows = slice(starts[itarg], starts[itarg + 1])
        for isrc, source in enumerate(chunks):
            cols = slice(starts[isrc], starts[isrc + 1])
            block_kernel = _select_local_kernel(kernel, itarg, isrc)
            opdims = _kernel_opdims(block_kernel, dimension)
            if itarg == isrc:
                if ilist is not None and getattr(block_kernel, "sing", "") in {"log", "pv", "hs"}:
                    block = quadggq.buildmat(
                        source,
                        block_kernel,
                        opdims,
                        getattr(block_kernel, "sing", "log"),
                        ilist=ilist[:, isrc],
                    )
                elif getattr(block_kernel, "sing", "") in {"log", "pv", "hs"}:
                    block = chunkermat(source, block_kernel)
                else:
                    block = quadnative.buildmat(source, block_kernel, opdims)
            else:
                block = chunkerkernevalmat(source, block_kernel, target, quadrature="smooth")
            block_arr = _zero_coincident_nonfinite_block(
                block,
                source,
                target,
                opdims,
                f"RCIP local matrix block ({itarg}, {isrc})",
            )
            out[rows, cols] = block_arr
    return out


def _zero_coincident_nonfinite_block(
    values: np.ndarray,
    source: Chunker,
    target: Chunker,
    opdims: tuple[int, int],
    context: str,
) -> np.ndarray:
    arr = np.array(values, copy=True)
    nonfinite = ~np.isfinite(arr)
    if not np.any(nonfinite):
        return arr

    expected = _coincident_chunker_mask(source, target, opdims, arr.shape)
    unexpected = nonfinite & ~expected
    if np.any(unexpected):
        raise ValueError(
            f"{context} contains non-finite values away from coincident source/target points"
        )
    arr[nonfinite] = 0.0
    return arr


def _coincident_chunker_mask(
    source: Chunker,
    target: Chunker,
    opdims: tuple[int, int],
    shape: tuple[int, ...],
) -> np.ndarray:
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    src_pts = as_boundary_point_matrix(source.r, source.dim, source.npt, name="source positions")
    targ_pts = as_boundary_point_matrix(target.r, target.dim, target.npt, name="target positions")
    expected_shape = (op0 * target.npt, op1 * source.npt)
    if tuple(shape) != expected_shape:
        return np.zeros(shape, dtype=bool)
    tol = (
        16.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.max(np.abs(src_pts))) if src_pts.size else 0.0,
            float(np.max(np.abs(targ_pts))) if targ_pts.size else 0.0,
        )
    )
    dist2 = np.sum((targ_pts[:, :, None] - src_pts[:, None, :]) ** 2, axis=0)
    coincident = dist2 <= tol**2
    return np.repeat(np.repeat(coincident, op0, axis=0), op1, axis=1)


def _select_local_kernel(kernel: Any, target_index: int, source_index: int) -> Any:
    arr = (
        np.asarray(kernel, dtype=object) if isinstance(kernel, (list, tuple, np.ndarray)) else None
    )
    if arr is not None and arr.ndim == 2:
        return arr[target_index, source_index]
    return kernel


def _kernel_opdims(kernel: Any, dimension: int) -> tuple[int, int]:
    opdims = getattr(kernel, "opdims", None)
    if opdims is None or opdims == (0, 0):
        return (int(dimension), int(dimension))
    return tuple(int(x) for x in opdims)
