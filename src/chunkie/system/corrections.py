"""Global correction construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chunkie.geometry import Chunker
from chunkie.kernels import Kernel
from chunkie.quadrature import (
    adaptive_panel_matrix,
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


def build_corrections(*args, **kwargs):
    raise NotImplementedError("Automatic correction selection is a required rewrite milestone")


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
        raise NotImplementedError("panel corrections currently support Chunker source and target geometry")
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
    else:
        raise ValueError("panel correction method must be 'adaptive' or 'helsing_ojala'")

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
    source_point_ids = source.point_map.to_point_id(source_panel_id, local_nodes)

    # System matrices are component-major over panel-major point ids. This is
    # the global counterpart of quadrature's local operator-matrix adapter.
    rows = np.concatenate([field * target.point_count + point_ids for field in range(kernel.output_dim)])
    columns = np.concatenate(
        [component * source.point_count + source_point_ids for component in range(kernel.input_dim)]
    )
    return rows, columns
