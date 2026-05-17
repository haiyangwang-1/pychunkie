"""Shared smooth-circle Laplace example machinery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chunkie.geometry import Chunker, circle
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)

Side = Literal["interior", "exterior"]
Condition = Literal["dirichlet", "neumann"]

INTERIOR_TARGETS = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
EXTERIOR_TARGETS = np.array([[2.0, 1.4, -1.6], [0.0, 0.8, 0.3]])


@dataclass(frozen=True)
class SmoothLaplaceExampleResult:
    side: Side
    condition: Condition
    boundary: Chunker
    system: IntegralSystem
    target_points: NDArray[np.floating]
    values: NDArray[np.floating]
    truth: NDArray[np.floating]
    additive_constant: float
    residual_norm: float
    max_error: float
    solution_path: Path
    error_path: Path


def make_boundary() -> Chunker:
    return circle(quadrature_order=16, panel_count=24)


def interior_solution(targets: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.asarray(targets, dtype=float)[0]


def exterior_solution(targets: NDArray[np.floating]) -> NDArray[np.floating]:
    points = np.asarray(targets, dtype=float)
    return points[0] / np.sum(points * points, axis=0)


def analytic_solution(side: Side, targets: NDArray[np.floating]) -> NDArray[np.floating]:
    return interior_solution(targets) if side == "interior" else exterior_solution(targets)


def build_laplace_circle_system(side: Side, condition: Condition) -> tuple[Chunker, IntegralSystem]:
    boundary = make_boundary()
    density = DensitySpace("sigma", boundary)

    if condition == "dirichlet":
        trace_kernel = kernel("laplace", selector="d")
        field_kernel = trace_kernel
        # With right normals on a counter-clockwise curve, the double-layer
        # trace is (-1/2 I + D) from the interior and (+1/2 I + D) from the
        # exterior.
        jump = -0.5 if side == "interior" else 0.5
        rhs = analytic_solution(side, boundary.positions)
        layer_name = "double"
        equation_name = "dirichlet"
    else:
        trace_kernel = kernel("laplace", selector="sp")
        field_kernel = kernel("laplace", selector="s")
        # The adjoint double-layer trace of the single layer has the opposite
        # one-sided jump: (+1/2 I + S') inside and (-1/2 I + S') outside.
        jump = 0.5 if side == "interior" else -0.5
        rhs = boundary.normals[0] if side == "interior" else -boundary.positions[0]
        layer_name = "single"
        equation_name = "neumann"

    trace_layer = LayerPotential(f"{layer_name}_trace", boundary, trace_kernel, "sigma")
    field_layer = LayerPotential(layer_name, boundary, field_kernel, "sigma")
    trace = BoundaryTrace(trace_layer, boundary, side, jump=JumpTerm(jump, "sigma"))
    system = IntegralSystem(
        f"smooth_laplace_{side}_{condition}",
        boundary,
        (density,),
        (BoundaryEquation(equation_name, boundary, (trace,), rhs),),
        fields={"u": (field_layer,)},
    )
    return boundary, system


def solve_case(
    side: Side,
    condition: Condition,
    *,
    grid_size: int = 96,
    output_dir: Path | str | None = None,
    stem: str | None = None,
) -> SmoothLaplaceExampleResult:
    boundary, system = build_laplace_circle_system(side, condition)
    solution = system.solve()
    x_grid, y_grid, mask = _field_grid(side, grid_size)
    targets = np.vstack((x_grid[mask], y_grid[mask]))

    raw_values = np.asarray(solution.evaluate(targets).values[0], dtype=complex).real
    truth = analytic_solution(side, targets)
    # Neumann data determines the potential only up to a constant. The examples
    # align that constant on the plotting/evaluation cloud before measuring
    # pointwise error.
    additive_constant = float(np.mean(truth - raw_values)) if condition == "neumann" else 0.0
    values = raw_values + additive_constant
    max_error = float(np.max(np.abs(values - truth)))
    residual_norm = float(np.linalg.norm(solution.residual))

    destination = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_stem = f"smooth_laplace_{side}_{condition}" if stem is None else stem
    solution_path, error_path = write_solution_plots(
        side,
        condition,
        x_grid,
        y_grid,
        mask,
        values,
        truth,
        destination=destination,
        stem=file_stem,
    )

    return SmoothLaplaceExampleResult(
        side=side,
        condition=condition,
        boundary=boundary,
        system=system,
        target_points=targets,
        values=values,
        truth=truth,
        additive_constant=additive_constant,
        residual_norm=residual_norm,
        max_error=max_error,
        solution_path=solution_path,
        error_path=error_path,
    )


def main_case(side: Side, condition: Condition) -> SmoothLaplaceExampleResult:
    result = solve_case(side, condition)
    print(
        f"unit circle: {result.boundary.panel_count} panels, "
        f"{result.boundary.point_count} nodes"
    )
    print(
        f"{side} {condition} residual: {result.residual_norm:.3e}; "
        f"max field error: {result.max_error:.3e}"
    )
    print(f"solution figure: {result.solution_path}")
    print(f"log10 abs error figure: {result.error_path}")
    return result


def write_solution_plots(
    side: Side,
    condition: Condition,
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    mask: NDArray[np.bool_],
    values: NDArray[np.floating],
    truth: NDArray[np.floating],
    *,
    destination: Path,
    stem: str,
) -> tuple[Path, Path]:
    field = np.full(x_grid.shape, np.nan)
    field[mask] = values
    error = np.full(x_grid.shape, np.nan)
    error[mask] = np.log10(np.maximum(np.abs(values - truth), 1.0e-16))

    solution_path = destination / f"{stem}_solution.png"
    error_path = destination / f"{stem}_error_log10.png"
    title_prefix = f"Smooth Laplace {side} {condition}"
    _save_scalar_plot(
        x_grid,
        y_grid,
        field,
        solution_path,
        title=f"{title_prefix}: solution",
        colorbar_label="u",
        cmap="viridis",
    )
    _save_scalar_plot(
        x_grid,
        y_grid,
        error,
        error_path,
        title=f"{title_prefix}: log10 abs error",
        colorbar_label="log10(abs(error))",
        cmap="magma",
    )
    return solution_path, error_path


def _field_grid(side: Side, grid_size: int) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    extent = 1.2 if side == "interior" else 2.0
    axis = np.linspace(-extent, extent, int(grid_size))
    x_grid, y_grid = np.meshgrid(axis, axis)
    radius = np.hypot(x_grid, y_grid)
    mask = radius < 0.95 if side == "interior" else radius > 1.05
    return x_grid, y_grid, mask


def _save_scalar_plot(
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    data: NDArray[np.floating],
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = ax.imshow(
        data,
        extent=(float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())),
        origin="lower",
        cmap=cmap,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="white", linewidth=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)
