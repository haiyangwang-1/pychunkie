"""Global correction construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import Chunker, flagnear
from chunkie.kernels import Kernel
from chunkie.quadrature import (
    adaptive_panel_matrix,
    build_ggq_self_panel_matrix,
    build_helsing_ojala_panel_matrix,
    operator_matrix_from_weighted_kernel,
)


@dataclass(frozen=True)
class PanelCorrection:
    rows: NDArray[np.integer]
    columns: NDArray[np.integer]
    values: NDArray[np.generic]
    diagnostics: dict[str, object]

    def apply_to(self, matrix: NDArray[np.generic]) -> None:
        if self.values.shape != (self.rows.size, self.columns.size):
            raise ValueError("panel correction values do not match row/column index sizes")
        matrix[np.ix_(self.rows, self.columns)] = self.values


def build_corrections(
    source: Chunker,
    target: Chunker,
    kernel: Kernel,
    *,
    method: str = "auto",
    side: str | None = None,
    near_rho: float = 1.8,
    tolerance: float = 1.0e-12,
    include_self: bool = True,
    include_near: bool = True,
) -> tuple[PanelCorrection, ...]:
    """Select dense local replacement blocks for self and near panel pairs."""

    if not isinstance(source, Chunker) or not isinstance(target, Chunker):
        raise NotImplementedError(
            "automatic corrections currently support Chunker source and target geometry"
        )

    target_points = target.pointinfo.flat_positions
    near_flags = flagnear(source, target_points, rho=near_rho)
    corrections: list[PanelCorrection] = []
    for source_panel_id in range(source.panel_count):
        target_ids = np.flatnonzero(near_flags[:, source_panel_id])
        self_ids = (
            source_panel_id * source.quadrature_order
            + np.arange(source.quadrature_order, dtype=np.int64)
            if source is target
            else np.array([], dtype=np.int64)
        )
        if include_self and self_ids.size:
            corrections.append(
                build_panel_correction(
                    source,
                    target,
                    kernel,
                    source_panel_id=source_panel_id,
                    target_point_ids=self_ids,
                    method=_select_correction_method(
                        kernel, requested=method, self_block=True, side=side
                    ),
                    side=side,
                    tolerance=tolerance,
                )
            )
        if include_near and target_ids.size:
            near_ids = np.setdiff1d(target_ids, self_ids, assume_unique=False)
            if near_ids.size:
                corrections.append(
                    build_panel_correction(
                        source,
                        target,
                        kernel,
                        source_panel_id=source_panel_id,
                        target_point_ids=near_ids,
                        method=_select_correction_method(
                            kernel, requested=method, self_block=False, side=side
                        ),
                        side=side,
                        tolerance=tolerance,
                    )
                )
    return tuple(corrections)


def _select_correction_method(
    kernel: Kernel, *, requested: str, self_block: bool, side: str | None
) -> str:
    method = requested.lower()
    if method != "auto":
        return method
    if self_block and kernel.family == "laplace" and kernel.selector == "s":
        return "ggq"
    if side is not None and kernel.singularity.expansion.terms:
        return "helsing_ojala"
    return "adaptive"


def build_panel_correction(
    source: Chunker,
    target: Chunker,
    kernel: Kernel,
    *,
    source_panel_id: int,
    target_point_ids,
    method: str = "adaptive",
    side: str | None = None,
    tolerance: float = 1.0e-12,
) -> PanelCorrection:
    """Build one dense replacement block for a source panel.

    The correction owns only the adapter boundary: rows and columns are global
    component-major system indices, while ``values`` is a local corrected panel
    matrix. Policy deciding which panels need replacement belongs in later
    correction-selection code.
    """

    if not isinstance(source, Chunker) or not isinstance(target, Chunker):
        raise NotImplementedError(
            "panel corrections currently support Chunker source and target geometry"
        )
    panel = source.panel(source_panel_id)
    point_ids = np.asarray(target_point_ids, dtype=np.int64).reshape(-1)
    target_points = target.pointinfo.flat_positions[:, point_ids]

    method0 = method.lower()
    if method0 == "adaptive":
        tensor = adaptive_panel_matrix(panel, target_points, kernel, tolerance=tolerance)
    elif method0 == "helsing_ojala":
        if side is None:
            raise ValueError("Helsing-Ojala panel corrections require an explicit side")
        tensor = build_helsing_ojala_panel_matrix(panel, target_points, kernel, side=side)
    elif method0 == "ggq":
        expected = source_panel_id * source.quadrature_order + np.arange(source.quadrature_order)
        if not np.array_equal(point_ids, expected):
            raise ValueError(
                "generated GGQ panel corrections currently require the matching self-panel targets"
            )
        tensor = build_ggq_self_panel_matrix(panel, kernel)
    else:
        raise ValueError("panel correction method must be 'adaptive', 'helsing_ojala', or 'ggq'")

    rows, columns = panel_block_indices(
        source,
        target,
        kernel,
        source_panel_id=source_panel_id,
        target_point_ids=point_ids,
    )
    return PanelCorrection(
        rows=rows,
        columns=columns,
        values=operator_matrix_from_weighted_kernel(tensor),
        diagnostics={
            "method": method0,
            "source_panel_id": int(source_panel_id),
            "target_point_count": int(point_ids.size),
            "kernel": f"{kernel.family}:{kernel.selector}",
        },
    )


def panel_block_indices(
    source: Chunker,
    target: Chunker,
    kernel: Kernel,
    *,
    source_panel_id: int,
    target_point_ids,
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    point_ids = np.asarray(target_point_ids, dtype=np.int64).reshape(-1)
    local_nodes = np.arange(source.quadrature_order, dtype=np.int64)
    source_point_ids = source_panel_id * source.quadrature_order + local_nodes

    # System matrices are component-major over panel-major point ids. This is
    # the global counterpart of quadrature's local operator-matrix adapter.
    rows = np.concatenate(
        [field * target.point_count + point_ids for field in range(kernel.output_dim)]
    )
    columns = np.concatenate(
        [component * source.point_count + source_point_ids for component in range(kernel.input_dim)]
    )
    return rows, columns
