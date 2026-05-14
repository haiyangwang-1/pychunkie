"""Special quadrature builders for singular and near-panel interactions.

This module mirrors the MATLAB ``chnk.quadggq`` entry points. MATLAB's
tabulated GGQ rules are vendored as NumPy package data, so runtime quadrature
loading does not depend on a local MATLAB checkout.

GGQ assembly starts from native smooth quadrature and overwrites blocks whose
source and target panels are the same or immediate neighbors. Self blocks use
the per-node split rules in ``xs0/wts0``; neighbor blocks use the one-sided
``xs1/wts1`` rule. This keeps the global matrix dense and MATLAB-compatible
while localizing all singular correction logic to panel blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from chunkie import lege
from chunkie.geometry import PointInfo
from chunkie.geometry.chunker import Chunker

from . import native as quadnative

_QUADGGQ_DATA_PATH = ("data", "quadggq")
_KERNEL_PROBE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    FloatingPointError,
    NotImplementedError,
)


@dataclass
class AuxQuad:
    xs1: np.ndarray
    wts1: np.ndarray
    xs0: list[np.ndarray]
    wts0: list[np.ndarray]
    ainterp1: np.ndarray
    ainterps0: list[np.ndarray]
    type: str = "log"


def setup(
    quadrature_order: int,
    singularity: str = "log",
    nfac_self: int | None = None,
    nfac_near: int | None = None,
) -> AuxQuad:
    """Generate auxiliary quadrature rules for self and neighbor panels."""

    order = int(quadrature_order)
    qtype = singularity.lower()
    if qtype not in {"log", "removable", "pv", "hs"}:
        raise ValueError("quadggq singularity must be one of log, removable, pv, or hs")

    use_matlab_log = nfac_self is None and nfac_near is None
    if use_matlab_log:
        xs1, wts1, xs0, wts0 = getlogquad(order, 2)
    else:
        if nfac_self is None:
            nfac_self = max(4, int(np.ceil(48 / max(order, 1))))
        if nfac_near is None:
            nfac_near = max(4, int(np.ceil(48 / max(order, 1))))
        xs1, wts1 = lege.exps(int(nfac_near * order))[:2]
        xs0, wts0 = getremovablequad(order, nfac_self)

    if qtype == "pv":
        xs0, wts0 = gethqsuppquad(order, 1)
    elif qtype == "hs":
        xs0, wts0 = gethqsuppquad(order, 2)
    elif qtype == "removable":
        xs0, wts0 = getremovablequad(order, 1 if use_matlab_log else int(nfac_self))
    return AuxQuad(
        xs1=xs1,
        wts1=wts1,
        xs0=xs0,
        wts0=wts0,
        ainterp1=lege.matrin(order, xs1)[0],
        ainterps0=[lege.matrin(order, xs)[0] for xs in xs0],
        type=qtype,
    )


def getlogquad(
    quadrature_order: int,
    npolyfac: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return MATLAB GGQ neighbor and self rules for logarithmic kernels."""

    order = int(quadrature_order)
    nfac = int(npolyfac)
    near_order = _log_near_order(order)
    near = _load_near_table(f"ggqnear{near_order}") if near_order is not None else None
    self_rule = _load_cell_table(f"ggqself_nnode{order:03d}_npoly{nfac * order:03d}")
    if near is None or self_rule is None:
        return _generated_logquad(order, nfac)
    xs1, wts1 = near
    xs0, wts0 = self_rule
    return xs1, wts1, xs0, wts0


def logavail() -> np.ndarray:
    """Return panel orders supported by MATLAB log GGQ tables."""

    return np.array([16, 20, 24, 30, 40, 60], dtype=int)


def hqsuppavail() -> np.ndarray:
    """Return MATLAB table orders available for PV/HS self quadrature."""

    return np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20], dtype=int)


def gethqsuppquad(
    quadrature_order: int,
    singularity_code: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return MATLAB GGQ support tables for PV or HS self interactions.

    ``singularity_code=1`` corresponds to principal-value support and
    ``singularity_code=2`` to hypersingular support. When a table is not
    vendored for the requested order, a generated removable split rule is
    returned as a conservative fallback.
    """

    order = int(quadrature_order)
    if order not in set(hqsuppavail().tolist()):
        return getremovablequad(order, 2)
    prefix = "hsupp" if int(singularity_code) == 1 else "hqsupp"
    table = _load_cell_table(f"{prefix}_nnode{order:03d}_npoly{2 * order:03d}")
    if table is None:
        return getremovablequad(order, 2)
    return table


def getremovablequad(
    quadrature_order: int,
    nfac: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return split Gauss rules on each side of every Legendre node."""

    order = int(quadrature_order)
    xleg, _ = lege.exps(order)[:2]
    xover, wover = lege.exps(max(int(np.ceil(order * nfac)), order))[:2]
    x01 = (xover + 1.0) / 2.0
    w01 = wover / 2.0
    xs0: list[np.ndarray] = []
    wts0: list[np.ndarray] = []
    for node in xleg:
        xleft = x01 * (node + 1.0) - 1.0
        wleft = w01 * (node + 1.0)
        xright = x01 * (1.0 - node) + node
        wright = w01 * (1.0 - node)
        xs0.append(np.concatenate((xleft, xright)))
        wts0.append(np.concatenate((wleft, wright)))
    return xs0, wts0


def _generated_logquad(
    quadrature_order: int,
    npolyfac: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    xs1, wts1 = lege.exps(max(int(npolyfac * quadrature_order), quadrature_order))[:2]
    xs0, wts0 = getremovablequad(quadrature_order, npolyfac)
    return xs1, wts1, xs0, wts0


def getpvquad(quadrature_order: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return gethqsuppquad(quadrature_order, 1)


def gethsquad(quadrature_order: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return gethqsuppquad(quadrature_order, 2)


def buildmat(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    singularity: str = "log",
    auxquads: AuxQuad | None = None,
    ilist: ArrayLike | None = None,
    *,
    pquad_side: str | None = None,
    usepquad: bool = False,
) -> np.ndarray:
    """Build a matrix with special self and neighbor quadrature blocks.

    ``quadnative.buildmat`` supplies every smooth interaction. The loop below
    replaces adjacent-panel blocks first and then each self block; those are the
    only places where logarithmic, principal-value, or hypersingular kernels
    need GGQ tables on a chunker boundary.
    """

    chunker = chunker
    kernel = kernel
    if opdims is None:
        opdims = getattr(kernel, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for special quadrature assembly")

    aux = setup(chunker.k, singularity) if auxquads is None else auxquads
    ignored = (
        set() if ilist is None else {int(idx) for idx in np.asarray(ilist, dtype=int).reshape(-1)}
    )
    mat = quadnative.buildmat(chunker, kernel, opdims)

    for src_chunk in range(chunker.nch):
        src_cols = _block_slice(src_chunk, chunker.k, int(opdims[1]))
        left, right = chunker.adj[:, src_chunk]

        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chunker.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            rows = _block_slice(targ_chunk, chunker.k, int(opdims[0]))
            mat[rows, src_cols] = nearbuildmat(
                chunker,
                targ_chunk,
                src_chunk,
                kernel,
                opdims,
                aux,
                pquad_side=pquad_side,
                usepquad=usepquad,
            )

        if src_chunk in ignored:
            continue
        rows = _block_slice(src_chunk, chunker.k, int(opdims[0]))
        mat[rows, src_cols] = diagbuildmat(chunker, src_chunk, kernel, opdims, aux)

    return mat


def buildmattd(
    chunker: Chunker,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    singularity: str = "log",
    auxquads: AuxQuad | None = None,
    ilist: ArrayLike | None = None,
    corrections: bool = False,
    *,
    pquad_side: str | None = None,
    usepquad: bool = False,
) -> sparse.csr_matrix:
    """Build the sparse matrix of special self and neighbor blocks only.

    This is a correction/overwrite view of GGQ. Operator code uses it to patch
    matrix-free paths: the sparse entries are exactly the local special blocks,
    while all far interactions still come from direct evaluation or FMM.
    """

    chunker = chunker
    kernel = kernel
    if opdims is None:
        opdims = getattr(kernel, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for special quadrature assembly")

    aux = setup(chunker.k, singularity) if auxquads is None else auxquads
    ignored = (
        set() if ilist is None else {int(idx) for idx in np.asarray(ilist, dtype=int).reshape(-1)}
    )
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def append_block(target_chunk: int, source_chunk: int, block: np.ndarray) -> None:
        row0 = target_chunk * chunker.k * op0
        col0 = source_chunk * chunker.k * op1
        rr, cc = np.indices(block.shape)
        rows.append((row0 + rr).reshape(-1))
        cols.append((col0 + cc).reshape(-1))
        vals.append(block.reshape(-1))

    for src_chunk in range(chunker.nch):
        left, right = chunker.adj[:, src_chunk]
        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chunker.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            append_block(
                targ_chunk,
                src_chunk,
                nearbuildmat(
                    chunker,
                    targ_chunk,
                    src_chunk,
                    kernel,
                    (op0, op1),
                    aux,
                    corrections=corrections,
                    pquad_side=pquad_side,
                    usepquad=usepquad,
                ),
            )

        if src_chunk in ignored:
            continue
        append_block(
            src_chunk,
            src_chunk,
            diagbuildmat(chunker, src_chunk, kernel, (op0, op1), aux, corrections=corrections),
        )

    shape = (chunker.npt * op0, chunker.npt * op1)
    if not vals:
        return sparse.csr_matrix(shape, dtype=float)
    data = np.concatenate(vals)
    dtype = np.result_type(data, _kernel_dtype(chunker, kernel))
    return sparse.coo_matrix(
        (data.astype(dtype, copy=False), (np.concatenate(rows), np.concatenate(cols))), shape=shape
    ).tocsr()


def diagbuildmat(
    chunker: Chunker,
    i: int,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
    corrections: bool = False,
    wtss: ArrayLike | None = None,
) -> np.ndarray:
    """Build a special self-interaction block for chunk ``i``."""

    aux = setup(chunker.k, "log") if aux is None else aux
    k = chunker.k
    out = np.zeros((int(opdims[0]) * k, int(opdims[1]) * k), dtype=_kernel_dtype(chunker, kernel))
    rs = chunker.r[:, :, i]
    ds = chunker.d[:, :, i]
    d2s = chunker.d2[:, :, i]
    ns = chunker.n[:, :, i]
    dd = chunker.data[:, :, i] if chunker.datadim else None

    for inode in range(k):
        interp = aux.ainterps0[inode]
        source = _interpolated_pointinfo(rs, ds, d2s, ns, dd, interp)
        target = PointInfo(
            r=rs[:, inode : inode + 1],
            d=ds[:, inode : inode + 1],
            d2=d2s[:, inode : inode + 1],
            n=ns[:, inode : inode + 1],
            data=dd[:, inode : inode + 1] if dd is not None else None,
        )
        weights = np.sqrt(np.sum(np.abs(source.d) ** 2, axis=0)) * aux.wts0[inode]
        kvals = _zero_coincident_nonfinite(
            _eval_kernel(kernel, source, target), source, target, opdims, "GGQ self block"
        )
        block = kvals * np.repeat(weights, int(opdims[1]))[None, :]
        rows = slice(int(opdims[0]) * inode, int(opdims[0]) * (inode + 1))
        out[rows, :] = block @ np.kron(interp, np.eye(int(opdims[1])))
    if corrections:
        src0 = PointInfo(r=rs, d=ds, d2=d2s, n=ns, data=dd)
        op0 = int(opdims[0])
        op1 = int(opdims[1])
        smooth = _zero_coincident_nonfinite(
            _eval_kernel(kernel, src0, src0), src0, src0, opdims, "GGQ correction block"
        )
        for inode in range(k):
            row = slice(op0 * inode, op0 * (inode + 1))
            col = slice(op1 * inode, op1 * (inode + 1))
            smooth[row, col] = 0.0
        if wtss is None:
            wtsi = chunker.wts[:, i]
        else:
            wtsi = np.asarray(wtss)[:, i]
        out = out - smooth * np.repeat(wtsi, op1)[None, :]
    return out


def nearbuildmat(
    chunker: Chunker,
    i: int,
    j: int,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
    corrections: bool = False,
    wtss: ArrayLike | None = None,
    *,
    pquad_side: str | None = None,
    usepquad: bool = False,
) -> np.ndarray:
    """Build an oversampled near-neighbor block from source chunk ``j`` to target chunk ``i``."""

    aux = setup(chunker.k, "log") if aux is None else aux
    target = PointInfo(
        r=chunker.r[:, :, i],
        d=chunker.d[:, :, i],
        d2=chunker.d2[:, :, i],
        n=chunker.n[:, :, i],
        data=chunker.data[:, :, i] if chunker.datadim else None,
    )
    if usepquad:
        pquad_block, handled = _pquad_near_block(chunker, j, kernel, opdims, target, pquad_side)
        if pquad_block is not None and np.all(handled):
            if corrections:
                pquad_block = pquad_block - _native_panel_block(chunker, i, j, kernel, opdims, wtss)
            return np.real_if_close(pquad_block)

    interp = aux.ainterp1
    source = _interpolated_pointinfo(
        chunker.r[:, :, j],
        chunker.d[:, :, j],
        chunker.d2[:, :, j],
        chunker.n[:, :, j],
        chunker.data[:, :, j] if chunker.datadim else None,
        interp,
    )
    weights = np.sqrt(np.sum(np.abs(source.d) ** 2, axis=0)) * aux.wts1
    kvals = _zero_coincident_nonfinite(
        _eval_kernel(kernel, source, target), source, target, opdims, "GGQ near block"
    )
    mat = kvals * np.repeat(weights, int(opdims[1]))[None, :]
    out = mat @ np.kron(interp, np.eye(int(opdims[1])))
    if corrections:
        out = out - _native_panel_block(chunker, i, j, kernel, opdims, wtss)
    return out


def _pquad_near_block(
    chunker: Chunker,
    src_chunk: int,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    target: PointInfo,
    side: str | None,
) -> tuple[np.ndarray | None, np.ndarray]:
    from . import panel as pquad

    split_info = pquad.splitinfo_for_kernel(kernel=kernel)
    if split_info is None or tuple(split_info.opdims) != (int(opdims[0]), int(opdims[1])):
        return None, np.zeros(target.r.shape[1], dtype=bool)
    block, handled = pquad.panel_matrix_auto_side(
        chunker=chunker,
        source_chunk=src_chunk,
        target=target,
        split_info=split_info,
        side=side,
    )
    return block, handled


def _native_panel_block(
    chunker: Chunker,
    targ_chunk: int,
    src_chunk: int,
    kernel: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    wtss: ArrayLike | None = None,
) -> np.ndarray:
    wtss_arr = chunker.wts if wtss is None else np.asarray(wtss)
    source = PointInfo(
        r=chunker.r[:, :, src_chunk],
        d=chunker.d[:, :, src_chunk],
        d2=chunker.d2[:, :, src_chunk],
        n=chunker.n[:, :, src_chunk],
        data=chunker.data[:, :, src_chunk] if chunker.datadim else None,
    )
    target = PointInfo(
        r=chunker.r[:, :, targ_chunk],
        d=chunker.d[:, :, targ_chunk],
        d2=chunker.d2[:, :, targ_chunk],
        n=chunker.n[:, :, targ_chunk],
        data=chunker.data[:, :, targ_chunk] if chunker.datadim else None,
    )
    smooth = _eval_kernel(kernel, source, target)
    return smooth * np.repeat(wtss_arr[:, src_chunk], int(opdims[1]))[None, :]


def _zero_coincident_nonfinite(
    values: np.ndarray,
    source: PointInfo,
    target: PointInfo,
    opdims: tuple[int, int],
    context: str,
) -> np.ndarray:
    arr = np.array(values, copy=True)
    nonfinite = ~np.isfinite(arr)
    if not np.any(nonfinite):
        return arr

    expected = _coincident_kernel_mask(source, target, opdims, arr.shape)
    unexpected = nonfinite & ~expected
    if np.any(unexpected):
        raise ValueError(
            f"{context} kernel evaluation returned non-finite values away from coincident source/target points"
        )
    arr[nonfinite] = 0.0
    return arr


def _coincident_kernel_mask(
    source: PointInfo,
    target: PointInfo,
    opdims: tuple[int, int],
    shape: tuple[int, ...],
) -> np.ndarray:
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    nt = target.r.shape[1]
    ns = source.r.shape[1]
    expected_shape = (op0 * nt, op1 * ns)
    if tuple(shape) != expected_shape:
        return np.zeros(shape, dtype=bool)
    tol = (
        16.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.max(np.abs(source.r))) if source.r.size else 0.0,
            float(np.max(np.abs(target.r))) if target.r.size else 0.0,
        )
    )
    dist2 = np.sum((target.r[:, :, None] - source.r[:, None, :]) ** 2, axis=0)
    coincident = dist2 <= tol**2
    return np.repeat(np.repeat(coincident, op0, axis=0), op1, axis=1)


def _interpolated_pointinfo(
    r: np.ndarray,
    d: np.ndarray,
    d2: np.ndarray,
    n: np.ndarray,
    data: np.ndarray | None,
    interp: np.ndarray,
) -> PointInfo:
    ri = (interp @ r.T).T
    di = (interp @ d.T).T
    d2i = (interp @ d2.T).T
    speed = np.sqrt(np.sum(np.abs(di) ** 2, axis=0))
    ni = np.vstack((di[1], -di[0])) / speed[None, :]
    return PointInfo(
        r=ri,
        d=di,
        d2=d2i,
        n=ni if n is not None else None,
        data=(interp @ data.T).T if data is not None else None,
    )


def _eval_kernel(
    kernel: Callable[[Any, Any], np.ndarray], source: PointInfo, target: PointInfo
) -> np.ndarray:
    if hasattr(kernel, "eval") and kernel.eval is not None:
        return kernel.eval(source, target)
    return kernel(source, target)


def _block_slice(chunk: int, quadrature_order: int, opdim: int) -> slice:
    start = chunk * quadrature_order * opdim
    return slice(start, start + quadrature_order * opdim)


def _kernel_dtype(chunker: Chunker, kernel: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    try:
        source = PointInfo(
            r=chunker.r[:, :1, 0],
            d=chunker.d[:, :1, 0],
            d2=chunker.d2[:, :1, 0],
            n=chunker.n[:, :1, 0],
        )
        return np.asarray(_eval_kernel(kernel, source, source)).dtype
    except _KERNEL_PROBE_EXCEPTIONS:
        return np.dtype(float)


def _load_near_table(stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    table = _load_npz_table(stem)
    if table is None:
        return None
    return table["x"].copy(), table["w"].copy()


def _load_cell_table(stem: str) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    table = _load_npz_table(stem)
    if table is None:
        return None
    count = int(np.asarray(table["count"]).item())
    xs = [table[f"xs0_{idx:03d}"].copy() for idx in range(count)]
    ws = [table[f"ws0_{idx:03d}"].copy() for idx in range(count)]
    return xs, ws


@cache
def _load_npz_table(stem: str) -> dict[str, np.ndarray] | None:
    resource = resources.files("chunkie").joinpath(*_QUADGGQ_DATA_PATH, f"{stem}.npz")
    try:
        with resources.as_file(resource) as path:
            with np.load(path) as data:
                return {name: data[name].copy() for name in data.files}
    except FileNotFoundError:
        return None


def _log_near_order(quadrature_order: int) -> int | None:
    if quadrature_order <= 16:
        return 16
    if quadrature_order <= 20:
        return 20
    if quadrature_order <= 24:
        return 24
    if quadrature_order <= 30:
        return 30
    if quadrature_order <= 40:
        return 40
    if quadrature_order <= 60:
        return 60
    return None
