"""ChunkGraph refinement helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .chunker import Chunker


def _refine_graph_last_len(graph: Any, last_len: float, options: dict[str, Any]) -> None:
    if last_len <= 0.0:
        raise ValueError("last_len must be positive")
    stype = str(options.get("stype", "a"))
    tol = 1e-10 * max(last_len, 1.0)
    for _ in range(int(options.get("maxiter_last_len", 20))):
        changed = False
        graph.vstruc = graph.procverts()
        for edges, signs in graph.vstruc:
            if edges.size == 0:
                continue
            endpoint_info: list[tuple[int, int, int, float]] = []
            for edge, sign in zip(edges, signs, strict=True):
                iedge = int(edge)
                isign = int(sign)
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                length = float(graph.echnks[iedge].chunklen([ichunk])[0])
                endpoint_info.append((iedge, isign, ichunk, length))
            target = _last_len_target([item[3] for item in endpoint_info], last_len)
            for iedge, isign, ichunk, length in endpoint_info:
                if abs(length - target) <= tol:
                    continue
                ratio = target / length
                if ratio <= 1e-8 or ratio >= 1.0 - 1e-8:
                    continue
                frac = ratio if isign < 0 else 1.0 - ratio
                graph.echnks[iedge] = (
                    graph.echnks[iedge].split(ichunk, frac=frac, stype=stype).sort()[0]
                )
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("graph last_len refinement did not converge")
    graph.vstruc = graph.procverts()
    graph.regions = graph.findregions()


def _balance_graph(graph: Any) -> None:
    for _ in range(1000):
        changed = False
        graph.vstruc = graph.procverts()
        for edges, signs in graph.vstruc:
            if edges.size == 0:
                continue
            endpoint_info: list[tuple[int, int, float]] = []
            for edge, sign in zip(edges, signs, strict=True):
                iedge = int(edge)
                isign = int(sign)
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                length = float(graph.echnks[iedge].chunklen([ichunk])[0])
                endpoint_info.append((iedge, isign, length))
            lengths = np.asarray([item[2] for item in endpoint_info], dtype=float)
            amin = float(np.min(lengths))
            amax_idx = int(np.argmax(lengths))
            if amin <= 0.0:
                continue
            nsplit = int(np.floor(np.log2(float(lengths[amax_idx]) / amin)))
            if nsplit <= 0:
                continue
            iedge, isign, _ = endpoint_info[amax_idx]
            for _ in range(nsplit):
                ichunk = 0 if isign < 0 else graph.echnks[iedge].nch - 1
                graph.echnks[iedge] = graph.echnks[iedge].split(ichunk).sort()[0]
            changed = True
        if not changed:
            graph.vstruc = graph.procverts()
            graph.regions = graph.findregions()
            return
    raise RuntimeError("graph balance did not converge")


def _last_len_target(lengths: Sequence[float], last_len: float) -> float:
    lens = np.asarray(lengths, dtype=float)
    level = -np.log2(float(np.min(lens)) / last_len)
    if level - np.round(level) < 1e-8:
        level = float(np.round(level))
    else:
        level = float(np.ceil(level))
    level = max(level, 1.0)
    return float(last_len * 2.0 ** (-level))


def _fit_edge_chunker(chunker: Chunker, v0: np.ndarray, v1: np.ndarray) -> Chunker:
    rend, _ = chunker.chunkends([0, chunker.nch - 1] if chunker.nch > 1 else [0])
    r0 = rend[:, 0, 0]
    r1 = rend[:, 1, -1]
    if np.linalg.norm(v1 - v0) <= 1e-14:
        return chunker.translate(v0 - r0)
    scale = np.linalg.norm(v1 - v0) / np.linalg.norm(r1 - r0)
    theta = np.arctan2(*(v1 - v0)[::-1]) - np.arctan2(*(r1 - r0)[::-1])
    return chunker.move(r0=r0, r1=v0, trotat=theta, scale=scale)


def _subchunker(chunker: Chunker, start: int, chunk_count: int, closed: bool) -> Chunker:
    sub = Chunker(
        {"k": chunker.k, "dim": chunker.dim, "nchstor": chunk_count, "nchmax": chunk_count}
    ).addchunk(chunk_count)
    sl = slice(start, start + chunk_count)
    sub.r = chunker.r[:, :, sl]
    sub.d = chunker.d[:, :, sl]
    sub.d2 = chunker.d2[:, :, sl]
    sub.adj = np.vstack((np.arange(0, chunk_count), np.arange(2, chunk_count + 2)))
    if closed:
        sub.adj[0, 0] = chunk_count
        sub.adj[1, -1] = 1
    else:
        sub.adj[0, 0] = -1
        sub.adj[1, -1] = -1
    sub.recompute_geometry()
    return sub
