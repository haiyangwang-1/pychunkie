"""RCIP compression state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RCIPCornerState:
    vertex_id: int
    edge_ids: tuple[int, ...]
    boundary_part: object
    local_geometry: object
    prolongation: NDArray[np.floating]
    weighted_prolongation: NDArray[np.floating]
    compressed_inverse: NDArray[np.generic]
    local_operator: NDArray[np.generic] | None = None
    star_indices: NDArray[np.integer] | None = None
    saved: RCIPSaved | None = None


@dataclass(frozen=True)
class RCIPState:
    corners: tuple[RCIPCornerState, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RCIPSaved:
    """Saved old-style RCIP recursion data for field evaluation."""

    quadrature_order: int
    dimension: int
    edge_count: int
    prolongation: NDArray[np.floating]
    weighted_prolongation: NDArray[np.floating]
    star_l: NDArray[np.integer]
    circ_l: NDArray[np.integer]
    star_s: NDArray[np.integer]
    circ_s: NDArray[np.integer]
    ilist: NDArray[np.integer]
    star_l_scalar: NDArray[np.integer]
    circ_l_scalar: NDArray[np.integer]
    subdivisions: int
    saved_depth: int
    inverses: tuple[NDArray[np.generic], ...]
    local_blocks: tuple[NDArray[np.generic], ...]
    local_geometries: tuple[object, ...]
    star_indices: NDArray[np.integer]
    center: NDArray[np.floating]
    starts_at_corner: NDArray[np.bool_]


@dataclass(frozen=True)
class RCIPSchurLevel:
    """Inputs for one dense RCIP Schur compression level."""

    prolongation: ArrayLike
    weighted_prolongation: ArrayLike
    local_matrix: ArrayLike
    fine_star_indices: ArrayLike
    fine_eliminated_indices: ArrayLike
    coarse_star_indices: ArrayLike
    coarse_eliminated_indices: ArrayLike


@dataclass(frozen=True)
class RecursiveCompressionResult:
    """Dense reference output of a multi-level RCIP compression."""

    compressed_inverse: NDArray[np.generic]
    level_inverses: tuple[NDArray[np.generic], ...]

    @property
    def level_count(self) -> int:
        return len(self.level_inverses)


def schur_compress_block(
    prolongation: ArrayLike,
    weighted_prolongation: ArrayLike,
    local_matrix: ArrayLike,
    seed_inverse: ArrayLike,
    fine_star_indices: ArrayLike,
    fine_eliminated_indices: ArrayLike,
    coarse_star_indices: ArrayLike,
    coarse_eliminated_indices: ArrayLike,
) -> NDArray[np.generic]:
    """Apply one dense RCIP Schur update in local corner layout."""

    prolongation_array = np.asarray(prolongation)
    weighted_prolongation_array = np.asarray(weighted_prolongation)
    local_matrix_array = np.asarray(local_matrix)
    compressed = np.array(seed_inverse, copy=True)
    fine_star = np.asarray(fine_star_indices, dtype=np.int64).reshape(-1)
    fine_eliminated = np.asarray(fine_eliminated_indices, dtype=np.int64).reshape(-1)
    coarse_star = np.asarray(coarse_star_indices, dtype=np.int64).reshape(-1)
    coarse_eliminated = np.asarray(coarse_eliminated_indices, dtype=np.int64).reshape(-1)

    # This is the dense reference form of the RCIP local elimination. The
    # prolongation matrices move between coarse star unknowns and the finer
    # corner star, while the eliminated block is inverted explicitly here.
    eliminated_to_star = local_matrix_array[np.ix_(fine_eliminated, fine_star)] @ compressed
    weighted_star = weighted_prolongation_array.T @ compressed
    weighted_coupling = weighted_star @ local_matrix_array[np.ix_(fine_star, fine_eliminated)]
    eliminated_inverse = np.linalg.inv(
        local_matrix_array[np.ix_(fine_eliminated, fine_eliminated)]
        - eliminated_to_star @ local_matrix_array[np.ix_(fine_star, fine_eliminated)]
    )
    eliminated_to_prolonged = eliminated_inverse @ (eliminated_to_star @ prolongation_array)

    compressed[np.ix_(coarse_star, coarse_star)] = (
        weighted_star @ prolongation_array + weighted_coupling @ eliminated_to_prolonged
    )
    compressed[np.ix_(coarse_eliminated, coarse_eliminated)] = eliminated_inverse
    compressed[np.ix_(coarse_eliminated, coarse_star)] = -eliminated_to_prolonged
    compressed[np.ix_(coarse_star, coarse_eliminated)] = -weighted_coupling @ eliminated_inverse
    return compressed


def recursive_schur_compress(
    levels: Sequence[RCIPSchurLevel],
    seed_inverse: ArrayLike,
) -> RecursiveCompressionResult:
    """Apply a sequence of dense RCIP Schur updates.

    This reference driver keeps every intermediate inverse so system-level RCIP
    code can attach diagnostics before the production corner recursion is
    optimized or specialized.
    """

    current = np.asarray(seed_inverse)
    history: list[NDArray[np.generic]] = []
    for level in levels:
        current = schur_compress_block(
            level.prolongation,
            level.weighted_prolongation,
            level.local_matrix,
            current,
            level.fine_star_indices,
            level.fine_eliminated_indices,
            level.coarse_star_indices,
            level.coarse_eliminated_indices,
        )
        history.append(current)
    return RecursiveCompressionResult(
        compressed_inverse=current,
        level_inverses=tuple(history),
    )
