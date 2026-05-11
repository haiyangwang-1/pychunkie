"""Special quadrature builders for singular and near-panel interactions.

This module mirrors the MATLAB ``chnk.quadggq`` entry points. The current
Python rules are generated dynamically from split and oversampled
Gauss-Legendre rules, avoiding tabulated data while still replacing the
native diagonal and neighbor blocks where singular kernels need help.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

from chunkie import lege
from chunkie.chunker import Chunker
from chunkie.operators import PointInfo
from chunkie.chnk import quadnative


@dataclass
class AuxQuad:
    xs1: np.ndarray
    wts1: np.ndarray
    xs0: list[np.ndarray]
    wts0: list[np.ndarray]
    ainterp1: np.ndarray
    ainterps0: list[np.ndarray]
    type: str = "log"


def setup(k: int, type: str = "log", nfac_self: int | None = None, nfac_near: int | None = None) -> AuxQuad:
    """Generate auxiliary quadrature rules for self and neighbor panels."""

    qtype = type.lower()
    if qtype not in {"log", "removable", "pv", "hs"}:
        raise ValueError("quadggq type must be one of log, removable, pv, or hs")

    use_matlab_log = nfac_self is None and nfac_near is None
    if use_matlab_log:
        xs1, wts1, xs0, wts0 = getlogquad(k, 2)
    else:
        if nfac_self is None:
            nfac_self = max(4, int(np.ceil(48 / max(k, 1))))
        if nfac_near is None:
            nfac_near = max(4, int(np.ceil(48 / max(k, 1))))
        xs1, wts1 = lege.exps(int(nfac_near * k))[:2]
        xs0, wts0 = getremovablequad(k, nfac_self)

    if qtype == "pv":
        xs0, wts0 = gethqsuppquad(k, 1)
    elif qtype == "hs":
        xs0, wts0 = gethqsuppquad(k, 2)
    elif qtype == "removable":
        xs0, wts0 = getremovablequad(k, 1 if use_matlab_log else int(nfac_self))
    return AuxQuad(
        xs1=xs1,
        wts1=wts1,
        xs0=xs0,
        wts0=wts0,
        ainterp1=lege.matrin(k, xs1)[0],
        ainterps0=[lege.matrin(k, xs)[0] for xs in xs0],
        type=qtype,
    )


def getlogquad(k: int, npolyfac: int = 2) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return MATLAB GGQ neighbor and self rules for logarithmic kernels."""

    order = int(k)
    nfac = int(npolyfac)
    near_order = _log_near_order(order)
    table_dir = _matlab_quadggq_dir()
    near_table = table_dir / f"ggqnear{near_order}.m" if near_order is not None else None
    self_table = table_dir / f"ggqself_nnode{order:03d}_npoly{nfac * order:03d}.m"
    if near_table is None or not near_table.exists() or not self_table.exists():
        return _generated_logquad(order, nfac)
    xs1 = _parse_matlab_vector_assignment(near_table, "x")
    wts1 = _parse_matlab_vector_assignment(near_table, "w")
    xs0, wts0 = _parse_matlab_cell_table(self_table)
    return xs1, wts1, xs0, wts0


def logavail() -> np.ndarray:
    """Return panel orders supported by MATLAB log GGQ tables."""

    return np.array([*range(1, 17), 20, 24, 28, 32, 36], dtype=int)


def hqsuppavail() -> np.ndarray:
    """Return MATLAB table orders available for PV/HS self quadrature."""

    return np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20], dtype=int)


def gethqsuppquad(k: int, itype: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return MATLAB GGQ support tables for PV or HS self interactions.

    ``itype=1`` corresponds to principal-value support and ``itype=2`` to
    hypersingular support. When the cloned MATLAB reference is unavailable,
    a generated removable split rule is returned as a conservative fallback.
    """

    order = int(k)
    if order not in set(hqsuppavail().tolist()):
        return getremovablequad(order, 2)
    prefix = "hsupp" if int(itype) == 1 else "hqsupp"
    table = _matlab_quadggq_dir() / f"{prefix}_nnode{order:03d}_npoly{2 * order:03d}.m"
    if not table.exists():
        return getremovablequad(order, 2)
    return _parse_matlab_cell_table(table)


def getremovablequad(k: int, nfac: int = 1) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return split Gauss rules on each side of every Legendre node."""

    xleg, _ = lege.exps(k)[:2]
    xover, wover = lege.exps(max(int(np.ceil(k * nfac)), k))[:2]
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


def _generated_logquad(k: int, npolyfac: int = 2) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    xs1, wts1 = lege.exps(max(int(npolyfac * k), k))[:2]
    xs0, wts0 = getremovablequad(k, npolyfac)
    return xs1, wts1, xs0, wts0


def getpvquad(k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return gethqsuppquad(k, 1)


def gethsquad(k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return gethqsuppquad(k, 2)


def buildmat(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    type: str = "log",
    auxquads: AuxQuad | None = None,
    ilist: ArrayLike | None = None,
) -> np.ndarray:
    """Build a matrix with special self and neighbor quadrature blocks."""

    if opdims is None:
        opdims = getattr(kern, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for special quadrature assembly")

    aux = setup(chnkr.k, type) if auxquads is None else auxquads
    ignored = set() if ilist is None else {int(idx) for idx in np.asarray(ilist, dtype=int).reshape(-1)}
    mat = quadnative.buildmat(chnkr, kern, opdims)

    for src_chunk in range(chnkr.nch):
        src_cols = _block_slice(src_chunk, chnkr.k, int(opdims[1]))
        left, right = chnkr.adj[:, src_chunk]

        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chnkr.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            rows = _block_slice(targ_chunk, chnkr.k, int(opdims[0]))
            mat[rows, src_cols] = nearbuildmat(chnkr, targ_chunk, src_chunk, kern, opdims, aux)

        if src_chunk in ignored:
            continue
        rows = _block_slice(src_chunk, chnkr.k, int(opdims[0]))
        mat[rows, src_cols] = diagbuildmat(chnkr, src_chunk, kern, opdims, aux)

    return mat


def buildmattd(
    chnkr: Chunker,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int] | None = None,
    type: str = "log",
    auxquads: AuxQuad | None = None,
    ilist: ArrayLike | None = None,
    corrections: bool = False,
) -> sparse.csr_matrix:
    """Build the sparse matrix of special self and neighbor blocks only."""

    if opdims is None:
        opdims = getattr(kern, "opdims", None)
    if opdims is None or opdims == (0, 0):
        raise ValueError("opdims must be provided for special quadrature assembly")

    aux = setup(chnkr.k, type) if auxquads is None else auxquads
    ignored = set() if ilist is None else {int(idx) for idx in np.asarray(ilist, dtype=int).reshape(-1)}
    op0 = int(opdims[0])
    op1 = int(opdims[1])
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    def append_block(targ_chunk: int, src_chunk: int, block: np.ndarray) -> None:
        row0 = targ_chunk * chnkr.k * op0
        col0 = src_chunk * chnkr.k * op1
        rr, cc = np.indices(block.shape)
        rows.append((row0 + rr).reshape(-1))
        cols.append((col0 + cc).reshape(-1))
        vals.append(block.reshape(-1))

    for src_chunk in range(chnkr.nch):
        left, right = chnkr.adj[:, src_chunk]
        for targ_chunk in (int(left) - 1, int(right) - 1):
            if targ_chunk < 0 or targ_chunk >= chnkr.nch:
                continue
            if src_chunk in ignored and targ_chunk in ignored:
                continue
            append_block(
                targ_chunk,
                src_chunk,
                nearbuildmat(chnkr, targ_chunk, src_chunk, kern, (op0, op1), aux, corrections=corrections),
            )

        if src_chunk in ignored:
            continue
        append_block(
            src_chunk,
            src_chunk,
            diagbuildmat(chnkr, src_chunk, kern, (op0, op1), aux, corrections=corrections),
        )

    shape = (chnkr.npt * op0, chnkr.npt * op1)
    if not vals:
        return sparse.csr_matrix(shape, dtype=float)
    data = np.concatenate(vals)
    dtype = np.result_type(data, _kernel_dtype(chnkr, kern))
    return sparse.coo_matrix((data.astype(dtype, copy=False), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()


def diagbuildmat(
    chnkr: Chunker,
    i: int,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
    corrections: bool = False,
    wtss: ArrayLike | None = None,
) -> np.ndarray:
    """Build a special self-interaction block for chunk ``i``."""

    aux = setup(chnkr.k, "log") if aux is None else aux
    k = chnkr.k
    out = np.zeros((int(opdims[0]) * k, int(opdims[1]) * k), dtype=_kernel_dtype(chnkr, kern))
    rs = chnkr.r[:, :, i]
    ds = chnkr.d[:, :, i]
    d2s = chnkr.d2[:, :, i]
    ns = chnkr.n[:, :, i]
    dd = chnkr.data[:, :, i] if chnkr.datadim else None

    for inode in range(k):
        interp = aux.ainterps0[inode]
        src = _interpolated_pointinfo(rs, ds, d2s, ns, dd, interp)
        targ = PointInfo(
            r=rs[:, inode : inode + 1],
            d=ds[:, inode : inode + 1],
            d2=d2s[:, inode : inode + 1],
            n=ns[:, inode : inode + 1],
            data=dd[:, inode : inode + 1] if dd is not None else None,
        )
        weights = np.sqrt(np.sum(np.abs(src.d) ** 2, axis=0)) * aux.wts0[inode]
        kvals = np.nan_to_num(_eval_kernel(kern, src, targ), nan=0.0, posinf=0.0, neginf=0.0)
        block = kvals * np.repeat(weights, int(opdims[1]))[None, :]
        rows = slice(int(opdims[0]) * inode, int(opdims[0]) * (inode + 1))
        out[rows, :] = block @ np.kron(interp, np.eye(int(opdims[1])))
    if corrections:
        src0 = PointInfo(r=rs, d=ds, d2=d2s, n=ns, data=dd)
        smooth = np.array(_eval_kernel(kern, src0, src0), copy=True)
        op0 = int(opdims[0])
        op1 = int(opdims[1])
        for inode in range(k):
            row = slice(op0 * inode, op0 * (inode + 1))
            col = slice(op1 * inode, op1 * (inode + 1))
            smooth[row, col] = 0.0
        if wtss is None:
            wtsi = chnkr.wts[:, i]
        else:
            wtsi = np.asarray(wtss)[:, i]
        out = out - np.nan_to_num(smooth, nan=0.0, posinf=0.0, neginf=0.0) * np.repeat(wtsi, op1)[None, :]
    return out


def nearbuildmat(
    chnkr: Chunker,
    i: int,
    j: int,
    kern: Callable[[Any, Any], np.ndarray],
    opdims: tuple[int, int],
    aux: AuxQuad | None = None,
    corrections: bool = False,
    wtss: ArrayLike | None = None,
) -> np.ndarray:
    """Build an oversampled near-neighbor block from source chunk ``j`` to target chunk ``i``."""

    aux = setup(chnkr.k, "log") if aux is None else aux
    interp = aux.ainterp1
    src = _interpolated_pointinfo(
        chnkr.r[:, :, j],
        chnkr.d[:, :, j],
        chnkr.d2[:, :, j],
        chnkr.n[:, :, j],
        chnkr.data[:, :, j] if chnkr.datadim else None,
        interp,
    )
    targ = PointInfo(
        r=chnkr.r[:, :, i],
        d=chnkr.d[:, :, i],
        d2=chnkr.d2[:, :, i],
        n=chnkr.n[:, :, i],
        data=chnkr.data[:, :, i] if chnkr.datadim else None,
    )
    weights = np.sqrt(np.sum(np.abs(src.d) ** 2, axis=0)) * aux.wts1
    kvals = np.nan_to_num(_eval_kernel(kern, src, targ), nan=0.0, posinf=0.0, neginf=0.0)
    mat = kvals * np.repeat(weights, int(opdims[1]))[None, :]
    out = mat @ np.kron(interp, np.eye(int(opdims[1])))
    if corrections:
        if wtss is None:
            wtss_arr = chnkr.wts
        else:
            wtss_arr = np.asarray(wtss)
        src0 = PointInfo(
            r=chnkr.r[:, :, j],
            d=chnkr.d[:, :, j],
            d2=chnkr.d2[:, :, j],
            n=chnkr.n[:, :, j],
            data=chnkr.data[:, :, j] if chnkr.datadim else None,
        )
        smooth = _eval_kernel(kern, src0, targ)
        out = out - smooth * np.repeat(wtss_arr[:, j], int(opdims[1]))[None, :]
    return out


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


def _eval_kernel(kern: Callable[[Any, Any], np.ndarray], src: PointInfo, targ: PointInfo) -> np.ndarray:
    if hasattr(kern, "eval") and getattr(kern, "eval") is not None:
        return kern.eval(src, targ)
    return kern(src, targ)


def _block_slice(chunk: int, k: int, opdim: int) -> slice:
    start = chunk * k * opdim
    return slice(start, start + k * opdim)


def _kernel_dtype(chnkr: Chunker, kern: Callable[[Any, Any], np.ndarray]) -> np.dtype:
    try:
        src = PointInfo(r=chnkr.r[:, :1, 0], d=chnkr.d[:, :1, 0], d2=chnkr.d2[:, :1, 0], n=chnkr.n[:, :1, 0])
        return np.asarray(_eval_kernel(kern, src, src)).dtype
    except Exception:
        return np.dtype(float)


def _matlab_quadggq_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "external" / "chunkie-matlab" / "chunkie" / "+chnk" / "+quadggq"


def _parse_matlab_cell_table(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    text = path.read_text(encoding="utf-8")
    xs = _parse_cells(text, "xs0")
    ws = _parse_cells(text, "ws0")
    if len(xs) != len(ws):
        raise ValueError(f"malformed MATLAB GGQ table: {path}")
    return xs, ws


def _parse_matlab_vector_assignment(path: Path, name: str) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"\b{name}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if match is None:
        raise ValueError(f"no {name} vector found in MATLAB table: {path}")
    clean = match.group(1).replace("D", "E").replace("d", "e")
    return np.fromstring(clean, sep=" ")


def _log_near_order(k: int) -> int | None:
    if k <= 16:
        return 16
    if k <= 20:
        return 20
    if k <= 24:
        return 24
    if k <= 30:
        return 30
    if k <= 40:
        return 40
    if k <= 60:
        return 60
    return None


def _parse_cells(text: str, name: str) -> list[np.ndarray]:
    matches = re.findall(rf"{name}\{{\s*(\d+)\s*\}}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if not matches:
        raise ValueError(f"no {name} cells found in MATLAB GGQ table")
    out: list[np.ndarray] = [np.array([])] * len(matches)
    for idx, block in matches:
        clean = block.replace("D", "E").replace("d", "e")
        vals = np.fromstring(clean, sep=" ")
        out[int(idx) - 1] = vals
    return out
