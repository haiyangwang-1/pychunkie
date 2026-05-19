"""Panel refinement helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from chunkie.quadrature.legendre import interpolation_matrix, legendre_rule

from .chunker import Chunker, right_normals
from .constructors import _adjacency


@dataclass(frozen=True)
class _PanelInterval:
    parent_panel: int
    left: float
    right: float


def refine(
    chunker: Chunker,
    *,
    levels: int | None = None,
    splitchunks: Sequence[int] | NDArray[np.integer] | None = None,
    lvlr: Literal["a", "n"] = "a",
    lvlrfac: float = 2.1,
    maxchunklen: float = np.inf,
    nover: int = 0,
    maxiter_lvlr: int = 1000,
    stype: Literal["a", "t"] = "a",
    nchmax: int | None = None,
) -> Chunker:
    """Refine panels by selected splits, length limits, level restriction, and oversampling.

    Parameters mirror the active MATLAB ``refine`` knobs while keeping Python's
    ordered panel storage. ``levels`` remains as the old Python alias for
    ``nover``.
    """

    active_nover = _validate_refine_options(
        chunker,
        levels=levels,
        lvlr=lvlr,
        lvlrfac=lvlrfac,
        maxchunklen=maxchunklen,
        nover=nover,
        maxiter_lvlr=maxiter_lvlr,
        stype=stype,
        nchmax=nchmax,
    )
    max_panel_count = 10 * chunker.panel_count if nchmax is None else int(nchmax)
    intervals = [
        _PanelInterval(parent_panel=panel_id, left=-1.0, right=1.0)
        for panel_id in range(chunker.panel_count)
    ]

    split_ids = _split_id_set(splitchunks, chunker.panel_count)
    if split_ids:
        intervals = _split_marked_intervals(
            chunker,
            intervals,
            [interval.parent_panel in split_ids for interval in intervals],
            stype=stype,
            max_panel_count=max_panel_count,
        )

    if np.isfinite(maxchunklen):
        intervals = _enforce_max_panel_length(
            chunker,
            intervals,
            float(maxchunklen),
            stype=stype,
            max_panel_count=max_panel_count,
        )

    if lvlr.lower() == "a":
        intervals = _enforce_level_restriction(
            chunker,
            intervals,
            factor=float(lvlrfac),
            max_iterations=int(maxiter_lvlr),
            stype=stype,
            max_panel_count=max_panel_count,
        )

    for _ in range(active_nover):
        intervals = _split_marked_intervals(
            chunker,
            intervals,
            [True] * len(intervals),
            stype=stype,
            max_panel_count=max_panel_count,
        )

    if _is_identity_intervals(intervals, chunker.panel_count):
        return replace(chunker)
    return _chunker_from_intervals(
        chunker,
        intervals,
        refinement_metadata={
            "splitchunks": tuple(sorted(split_ids)),
            "lvlr": lvlr.lower(),
            "lvlrfac": float(lvlrfac),
            "maxchunklen": float(maxchunklen),
            "nover": int(active_nover),
            "stype": stype.lower(),
        },
    )


def _validate_refine_options(
    chunker: Chunker,
    *,
    levels: int | None,
    lvlr: str,
    lvlrfac: float,
    maxchunklen: float,
    nover: int,
    maxiter_lvlr: int,
    stype: str,
    nchmax: int | None,
) -> int:
    if levels is not None:
        if nover != 0:
            raise ValueError("specify either levels or nover, not both")
        if int(levels) < 0:
            raise ValueError("levels must be non-negative")
        nover = int(levels)
    if int(nover) < 0:
        raise ValueError("nover must be non-negative")
    if lvlr.lower() not in {"a", "n"}:
        raise ValueError("lvlr must be 'a' or 'n'")
    if stype.lower() not in {"a", "t"}:
        raise ValueError("stype must be 'a' for arclength or 't' for parameter split")
    if float(lvlrfac) <= 1.0:
        raise ValueError("lvlrfac must be greater than 1")
    if float(maxchunklen) <= 0.0:
        raise ValueError("maxchunklen must be positive")
    if int(maxiter_lvlr) < 0:
        raise ValueError("maxiter_lvlr must be non-negative")
    if nchmax is not None and int(nchmax) < chunker.panel_count:
        raise ValueError("nchmax cannot be smaller than the current panel count")
    return int(nover)


def _split_id_set(
    splitchunks: Sequence[int] | NDArray[np.integer] | None,
    panel_count: int,
) -> set[int]:
    if splitchunks is None:
        return set()
    split_ids = {int(panel_id) for panel_id in np.asarray(splitchunks, dtype=np.int64).reshape(-1)}
    if any(panel_id < 0 or panel_id >= panel_count for panel_id in split_ids):
        raise IndexError("splitchunks entries must be valid zero-based panel ids")
    return split_ids


def _split_marked_intervals(
    chunker: Chunker,
    intervals: list[_PanelInterval],
    should_split: Sequence[bool],
    *,
    stype: str,
    max_panel_count: int,
) -> list[_PanelInterval]:
    if len(intervals) + sum(bool(flag) for flag in should_split) > max_panel_count:
        raise ValueError("nchmax exceeded during refinement")

    out: list[_PanelInterval] = []
    for interval, split in zip(intervals, should_split, strict=True):
        if split:
            out.extend(_split_interval(chunker, interval, stype=stype))
        else:
            out.append(interval)
    return out


def _enforce_max_panel_length(
    chunker: Chunker,
    intervals: list[_PanelInterval],
    max_panel_length: float,
    *,
    stype: str,
    max_panel_count: int,
    max_iterations: int = 1000,
) -> list[_PanelInterval]:
    for iteration in range(max_iterations + 1):
        lengths = _interval_lengths(chunker, intervals)
        should_split = [length > max_panel_length for length in lengths]
        if not any(should_split):
            return intervals
        if iteration == max_iterations:
            break
        intervals = _split_marked_intervals(
            chunker,
            intervals,
            should_split,
            stype=stype,
            max_panel_count=max_panel_count,
        )
    raise RuntimeError("maxchunklen refinement did not converge")


def _enforce_level_restriction(
    chunker: Chunker,
    intervals: list[_PanelInterval],
    *,
    factor: float,
    max_iterations: int,
    stype: str,
    max_panel_count: int,
) -> list[_PanelInterval]:
    for iteration in range(max_iterations + 1):
        lengths = _interval_lengths(chunker, intervals)
        should_split = [False] * len(intervals)
        for panel_id, length in enumerate(lengths):
            neighbors = _neighbor_lengths(lengths, panel_id, closed=chunker.closed)
            if neighbors and any(length > factor * neighbor for neighbor in neighbors):
                should_split[panel_id] = True
        if not any(should_split):
            return intervals
        if iteration == max_iterations:
            break
        intervals = _split_marked_intervals(
            chunker,
            intervals,
            should_split,
            stype=stype,
            max_panel_count=max_panel_count,
        )
    raise RuntimeError("level restriction refinement did not converge")


def _neighbor_lengths(
    lengths: NDArray[np.floating],
    panel_id: int,
    *,
    closed: bool,
) -> list[float]:
    out: list[float] = []
    if panel_id > 0:
        out.append(float(lengths[panel_id - 1]))
    elif closed and lengths.size > 1:
        out.append(float(lengths[-1]))
    if panel_id + 1 < lengths.size:
        out.append(float(lengths[panel_id + 1]))
    elif closed and lengths.size > 1:
        out.append(float(lengths[0]))
    return out


def _split_interval(
    chunker: Chunker,
    interval: _PanelInterval,
    *,
    stype: str,
) -> tuple[_PanelInterval, _PanelInterval]:
    split = (
        0.5 * (interval.left + interval.right)
        if stype.lower() == "t"
        else _arclength_midpoint(chunker, interval)
    )
    return (
        _PanelInterval(interval.parent_panel, interval.left, split),
        _PanelInterval(interval.parent_panel, split, interval.right),
    )


def _arclength_midpoint(chunker: Chunker, interval: _PanelInterval) -> float:
    target = 0.5 * _interval_length(chunker, interval)
    left = interval.left
    right = interval.right
    for _ in range(48):
        midpoint = 0.5 * (left + right)
        trial = _PanelInterval(interval.parent_panel, interval.left, midpoint)
        if _interval_length(chunker, trial) < target:
            left = midpoint
        else:
            right = midpoint
    return 0.5 * (left + right)


def _chunker_from_intervals(
    chunker: Chunker,
    intervals: list[_PanelInterval],
    *,
    refinement_metadata: dict[str, object],
) -> Chunker:
    panel_count = len(intervals)
    positions = np.empty(
        (chunker.coordinate_dim, chunker.quadrature_order, panel_count), dtype=float
    )
    derivatives = np.empty_like(positions)
    second_derivatives = np.empty_like(positions)
    weights = np.empty((chunker.quadrature_order, panel_count), dtype=float)

    for out_panel, interval in enumerate(intervals):
        _fill_interval(
            chunker,
            interval,
            out_panel,
            positions,
            derivatives,
            second_derivatives,
            weights,
        )

    metadata = dict(chunker.metadata)
    metadata["refinement"] = refinement_metadata
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second_derivatives,
        normals=right_normals(derivatives),
        weights=weights,
        _legendre_nodes=chunker._legendre_nodes,
        _legendre_weights=chunker._legendre_weights,
        adjacency=_adjacency(panel_count, chunker.closed),
        closed=chunker.closed,
        orientation=chunker.orientation,
        vertices=chunker.vertices,
        metadata=metadata,
    )


def _fill_interval(
    chunker: Chunker,
    interval: _PanelInterval,
    out_panel: int,
    positions: NDArray[np.floating],
    derivatives: NDArray[np.floating],
    second_derivatives: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> None:
    center = 0.5 * (interval.left + interval.right)
    scale = 0.5 * (interval.right - interval.left)
    parent_nodes = center + scale * chunker._legendre_nodes
    interpolation = interpolation_matrix(chunker._legendre_nodes, parent_nodes)

    positions[:, :, out_panel] = np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.positions[:, :, interval.parent_panel],
    )
    # Parent derivatives are with respect to the parent reference coordinate.
    # The child coordinate maps by u_parent = center + scale*u_child, so the
    # chain rule contributes scale and scale**2 to first and second derivatives.
    derivatives[:, :, out_panel] = scale * np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.derivatives[:, :, interval.parent_panel],
    )
    second_derivatives[:, :, out_panel] = scale**2 * np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.second_derivatives[:, :, interval.parent_panel],
    )
    weights[:, out_panel] = chunker._legendre_weights * np.linalg.norm(
        derivatives[:, :, out_panel],
        axis=0,
    )


def _interval_lengths(
    chunker: Chunker,
    intervals: list[_PanelInterval],
) -> NDArray[np.floating]:
    return np.asarray([_interval_length(chunker, interval) for interval in intervals], dtype=float)


def _interval_length(chunker: Chunker, interval: _PanelInterval) -> float:
    nodes, weights = legendre_rule(max(24, chunker.quadrature_order + 8))
    midpoint = 0.5 * (interval.left + interval.right)
    half_width = 0.5 * (interval.right - interval.left)
    references = midpoint + half_width * nodes
    interpolation = interpolation_matrix(chunker._legendre_nodes, references)
    derivatives = np.einsum(
        "qk,Rk->Rq",
        interpolation,
        chunker.derivatives[:, :, interval.parent_panel],
    )
    return float(abs(half_width) * np.sum(weights * np.linalg.norm(derivatives, axis=0)))


def _is_identity_intervals(intervals: list[_PanelInterval], panel_count: int) -> bool:
    if len(intervals) != panel_count:
        return False
    return all(
        interval.parent_panel == panel_id
        and np.isclose(interval.left, -1.0)
        and np.isclose(interval.right, 1.0)
        for panel_id, interval in enumerate(intervals)
    )
