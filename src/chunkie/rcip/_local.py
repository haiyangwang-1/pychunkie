"""Local curve and edge helpers for RCIP."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .. import lege
from ..geometry.chunker import Chunker, ChunkerPref


def shiftedlegbasismats(
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quadrature_order = int(quadrature_order)
    t0, _, u, v = lege.exps(quadrature_order)
    t = (t0 + 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbclmat = vv_t[: quadrature_order - 1, :].T @ (
        np.diag(1.0 / ss[: quadrature_order - 1]) @ uu[:, : quadrature_order - 1].T
    )

    t = (t0 - 1.0) / 2.0
    basis = t[:, None] * v[:, :-1]
    uu, ss, vv_t = np.linalg.svd(basis, full_matrices=False)
    sbcrmat = vv_t[: quadrature_order - 1, :].T @ (
        np.diag(1.0 / ss[: quadrature_order - 1]) @ uu[:, : quadrature_order - 1].T
    )

    pm1 = lege.pols(np.array([-1.0]), quadrature_order - 1)[0].reshape(-1)
    p1 = lege.pols(np.array([1.0]), quadrature_order - 1)[0].reshape(-1)
    leftvalmat = pm1 @ u
    rightvalmat = p1 @ u
    return sbclmat, sbcrmat, leftvalmat, rightvalmat, u


def chunkerfunclocal(
    fcurve: Any,
    ts: ArrayLike,
    pref: ChunkerPref | dict[str, Any] | None = None,
    xs: ArrayLike | None = None,
    ws: ArrayLike | None = None,
) -> Chunker:
    p = ChunkerPref.from_any(pref)
    tbreaks = np.asarray(ts, dtype=float).reshape(-1)
    if tbreaks.size < 2:
        raise ValueError("ts must contain at least two endpoints")
    xnodes = lege.exps(p.k)[0] if xs is None else np.asarray(xs, dtype=float).reshape(-1)
    wnodes = lege.exps(p.k)[1] if ws is None else np.asarray(ws, dtype=float).reshape(-1)
    r0, _, _ = fcurve(np.array([tbreaks[0]]))
    dim = np.asarray(r0).reshape(-1, 1).shape[0]
    out = Chunker(
        ChunkerPref(
            nchmax=max(p.nchmax, tbreaks.size - 1),
            k=p.k,
            dim=dim,
            nchstor=max(p.nchstor, tbreaks.size - 1),
        ),
        xnodes,
        wnodes,
    )
    out.addchunk(tbreaks.size - 1)
    out.adj = np.vstack((np.arange(0, out.nch), np.arange(2, out.nch + 2)))
    out.adj[0, 0] = -1
    out.adj[1, -1] = -1
    for ich in range(out.nch):
        a = tbreaks[ich]
        b = tbreaks[ich + 1]
        ts_panel = a + (b - a) * (xnodes + 1.0) / 2.0
        r, d, d2 = fcurve(ts_panel)
        h = (b - a) / 2.0
        out.rstor[:, :, ich] = np.asarray(r).reshape(dim, p.k)
        out.dstor[:, :, ich] = np.asarray(d).reshape(dim, p.k) * h
        out.d2stor[:, :, ich] = np.asarray(d2).reshape(dim, p.k) * h * h
    out.recompute_geometry()
    return out


def _rcip_edge_records(
    chunks: list[Chunker],
    edge_chunks: ArrayLike,
    vertex: np.ndarray,
    sbclmat: np.ndarray,
    sbcrmat: np.ndarray,
    lvmat: np.ndarray,
    rvmat: np.ndarray,
    u: np.ndarray,
) -> list[dict[str, Any]]:
    pairs = _normalize_edge_chunk_pairs(chunks, edge_chunks, vertex)
    out: list[dict[str, Any]] = []
    for chunker_idx, chunk_idx in pairs:
        ch = chunks[int(chunker_idx)]
        r = ch.r[:, :, int(chunk_idx)].copy()
        d = ch.d[:, :, int(chunk_idx)]
        d2 = ch.d2[:, :, int(chunk_idx)]
        left, right = ch.adj[:, int(chunk_idx)]
        if left > 0 and right < 0:
            nextchunk = int(left) - 1
            ileftright = 1
            ctr = (rvmat @ r.T).reshape(ch.dim)
            rcentered = r - ctr[:, None]
            rcs = sbcrmat @ rcentered.T
        elif left < 0 and right > 0:
            nextchunk = int(right) - 1
            ileftright = -1
            ctr = (lvmat @ r.T).reshape(ch.dim)
            rcentered = r - ctr[:, None]
            rcs = sbclmat @ rcentered.T
        else:
            raise ValueError("RCIP edge chunk must be adjacent to one vertex and one neighbor")
        out.append(
            {
                "chunker": int(chunker_idx),
                "chunk": int(chunk_idx),
                "nextchunk": nextchunk,
                "ctr": ctr,
                "rcs": rcs,
                "dcs": u @ d.T,
                "d2cs": u @ d2.T,
                "dscal": 2.0,
                "d2scal": 4.0,
                "ileftright": ileftright,
            }
        )
    return out


def _normalize_edge_chunk_pairs(
    chunks: list[Chunker], edge_chunks: ArrayLike, vertex: np.ndarray
) -> list[tuple[int, int]]:
    arr = np.asarray(edge_chunks, dtype=int)
    if arr.ndim == 1:
        pairs = []
        for chunker_idx in arr.reshape(-1):
            ch = chunks[int(chunker_idx)]
            pairs.append((int(chunker_idx), _chunk_adjacent_to_vertex(ch, vertex)))
        return pairs
    if arr.shape[0] != 2:
        raise ValueError("edge_chunks must be a 1D chunker list or a 2 x nedge array")
    pairs = []
    for col in range(arr.shape[1]):
        chunker_idx = int(arr[0, col])
        chunk_idx = int(arr[1, col])
        if chunker_idx >= len(chunks) and chunker_idx - 1 >= 0:
            chunker_idx -= 1
        if chunk_idx >= chunks[chunker_idx].nch and chunk_idx - 1 >= 0:
            chunk_idx -= 1
        pairs.append((chunker_idx, chunk_idx))
    return pairs


def _chunk_adjacent_to_vertex(chunker: Chunker, vertex: np.ndarray) -> int:
    rend, _ = chunker.chunkends()
    candidates: list[tuple[float, int]] = []
    for ich in range(chunker.nch):
        if chunker.adj[0, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 0, ich] - vertex)), ich))
        if chunker.adj[1, ich] < 0:
            candidates.append((float(np.linalg.norm(rend[:, 1, ich] - vertex)), ich))
    if not candidates:
        raise ValueError("could not identify a vertex-adjacent chunk")
    return min(candidates, key=lambda item: item[0])[1]


def _shiftedcurve(
    t: ArrayLike,
    rc: np.ndarray,
    dc: np.ndarray,
    scald: float,
    d2c: np.ndarray,
    scald2: float,
    ilr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tt_in = np.asarray(t, dtype=float).reshape(-1)
    tt = 2.0 * tt_in - 1.0 if int(ilr) == -1 else 2.0 * tt_in + 1.0
    pols = lege.pols(tt, dc.shape[0] - 1)[0].T
    r = (tt_in[:, None] * (pols[:, : rc.shape[0]] @ rc)).T
    d = (float(scald) * (pols @ dc)).T
    d2 = (float(scald2) * (pols @ d2c)).T
    return r, d, d2
