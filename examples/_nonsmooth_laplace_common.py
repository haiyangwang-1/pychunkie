"""Shared square-corner Laplace example machinery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chunkie.geometry import BoundaryPart, ChunkGraph
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
    SystemConfig,
)

Side = Literal["interior", "exterior"]
Condition = Literal["dirichlet", "neumann"]

DEFAULT_QUADRATURE_ORDER = 32
DEFAULT_GRID_SIZE = 80
DEFAULT_RCIP_SUBDIVISIONS = 20

SQUARE_VERTICES = np.array(
    [
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ],
)
SQUARE_EDGES = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
INTERIOR_TARGETS = np.array([[0.0, 0.3, -0.2], [0.0, 0.2, 0.4]])
EXTERIOR_TARGETS = np.array([[1.3, 2.0, -1.4, 0.2], [0.2, 0.5, -1.2, 1.5]])

SOLUTION_CMAP = LinearSegmentedColormap.from_list(
    "chunkie_solution",
    ["#313695", "#f7f7f7", "#a50026"],
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "chunkie_error",
    ["#ffffff", "#fee08b", "#f46d43", "#a50026"],
)


@dataclass(frozen=True)
class NonsmoothLaplaceExampleResult:
    side: Side
    condition: Condition
    graph: ChunkGraph
    boundary: BoundaryPart
    system: IntegralSystem
    target_points: NDArray[np.floating]
    values: NDArray[np.floating]
    truth: NDArray[np.floating]
    additive_constant: float
    residual_norm: float
    max_error: float
    rcip_corner_count: int
    solution_path: Path
    error_path: Path


def make_square_graph(*, quadrature_order: int = DEFAULT_QUADRATURE_ORDER) -> ChunkGraph:
    return ChunkGraph.from_vertices(
        SQUARE_VERTICES,
        SQUARE_EDGES,
        quadrature_order=quadrature_order,
    )


def interior_solution(targets: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.asarray(targets, dtype=float)[0]


def exterior_solution(targets: NDArray[np.floating]) -> NDArray[np.floating]:
    points = np.asarray(targets, dtype=float)
    return points[0] / np.sum(points * points, axis=0)


def analytic_solution(side: Side, targets: NDArray[np.floating]) -> NDArray[np.floating]:
    return interior_solution(targets) if side == "interior" else exterior_solution(targets)


def analytic_gradient(side: Side, targets: NDArray[np.floating]) -> NDArray[np.floating]:
    points = np.asarray(targets, dtype=float)
    if side == "interior":
        return np.vstack((np.ones(points.shape[1]), np.zeros(points.shape[1])))
    x = points[0]
    y = points[1]
    radius2 = x * x + y * y
    radius4 = radius2 * radius2
    return np.vstack(((y * y - x * x) / radius4, -2.0 * x * y / radius4))


def build_laplace_square_system(
    side: Side,
    condition: Condition,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
) -> tuple[ChunkGraph, BoundaryPart, IntegralSystem]:
    graph = make_square_graph(quadrature_order=quadrature_order)
    boundary = graph.boundary(1, side="interior")
    density = DensitySpace("sigma", boundary)

    if condition == "dirichlet":
        trace_kernel = kernel("laplace", selector="d")
        field_kernel = trace_kernel
        # Right normals point outside the square, so the same one-sided
        # double-layer jumps used on smooth curves apply on each smooth side.
        jump = -0.5 if side == "interior" else 0.5
        rhs = analytic_solution(side, boundary.pointinfo.positions)
        layer_name = "double"
        equation_name = "dirichlet"
    else:
        trace_kernel = kernel("laplace", selector="sp")
        field_kernel = kernel("laplace", selector="s")
        jump = 0.5 if side == "interior" else -0.5
        rhs = _normal_derivative(side, boundary)
        layer_name = "single"
        equation_name = "neumann"

    trace_layer = LayerPotential(f"{layer_name}_trace", boundary, trace_kernel, "sigma")
    field_layer = LayerPotential(layer_name, boundary, field_kernel, "sigma")
    trace = BoundaryTrace(trace_layer, boundary, side, jump=JumpTerm(jump, "sigma"))
    system = IntegralSystem(
        f"nonsmooth_laplace_{side}_{condition}",
        graph,
        (density,),
        (BoundaryEquation(equation_name, boundary, (trace,), rhs),),
        fields={"u": (field_layer,)},
    )
    return graph, boundary, system


def solve_case(
    side: Side,
    condition: Condition,
    *,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    rcip_subdivisions: int = DEFAULT_RCIP_SUBDIVISIONS,
    grid_size: int = DEFAULT_GRID_SIZE,
    output_dir: Path | str | None = None,
    stem: str | None = None,
) -> NonsmoothLaplaceExampleResult:
    graph, boundary, system = build_laplace_square_system(
        side,
        condition,
        quadrature_order=quadrature_order,
    )
    config = SystemConfig(rcip_subdivisions=rcip_subdivisions)
    solution = system.solve(config=config)
    targets = INTERIOR_TARGETS if side == "interior" else EXTERIOR_TARGETS
    raw_values = np.asarray(solution.evaluate(targets).values[0], dtype=complex).real
    truth = analytic_solution(side, targets)
    # The single-layer Neumann representation fixes the derivative data but not
    # the additive potential constant, so align against off-boundary reference
    # points before measuring the example error.
    additive_constant = float(np.mean(truth - raw_values)) if condition == "neumann" else 0.0
    values = raw_values + additive_constant

    destination = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    file_stem = f"nonsmooth_laplace_{side}_{condition}" if stem is None else stem
    solution_path, error_path = write_solution_plots(
        side,
        condition,
        solution,
        additive_constant,
        grid_size=grid_size,
        destination=destination,
        stem=file_stem,
    )
    rcip_state = solution.operator.diagnostics.get("rcip") if solution.operator is not None else None
    corner_count = len(rcip_state.corners) if rcip_state is not None else 0

    return NonsmoothLaplaceExampleResult(
        side=side,
        condition=condition,
        graph=graph,
        boundary=boundary,
        system=system,
        target_points=targets,
        values=values,
        truth=truth,
        additive_constant=additive_constant,
        residual_norm=float(np.linalg.norm(solution.residual)),
        max_error=float(np.max(np.abs(values - truth))),
        rcip_corner_count=corner_count,
        solution_path=solution_path,
        error_path=error_path,
    )


def main_case(side: Side, condition: Condition) -> NonsmoothLaplaceExampleResult:
    result = solve_case(side, condition)
    print(
        f"square graph: {len(result.graph.edges)} edges, "
        f"{result.boundary.point_count} nodes, {result.rcip_corner_count} RCIP corners"
    )
    print(
        f"{side} {condition} residual: {result.residual_norm:.3e}; "
        f"target max error: {result.max_error:.3e}"
    )
    print(f"solution figure: {result.solution_path}")
    print(f"log10 abs error figure: {result.error_path}")
    return result


def write_solution_plots(
    side: Side,
    condition: Condition,
    solution,
    additive_constant: float,
    *,
    grid_size: int,
    destination: Path,
    stem: str,
) -> tuple[Path, Path]:
    x_grid, y_grid, mask = _field_grid(side, grid_size)
    targets = np.vstack((x_grid[mask], y_grid[mask]))
    values = np.asarray(solution.evaluate(targets).values[0], dtype=complex).real + additive_constant
    truth = analytic_solution(side, targets)

    field = np.full(x_grid.shape, np.nan)
    field[mask] = values
    error = np.full(x_grid.shape, np.nan)
    error[mask] = np.log10(np.maximum(np.abs(values - truth), 1.0e-16))

    solution_path = destination / f"{stem}_solution.png"
    error_path = destination / f"{stem}_error_log10.png"
    title_prefix = f"Nonsmooth Laplace {side} {condition}"
    _save_scalar_plot(
        x_grid,
        y_grid,
        field,
        solution_path,
        title=f"{title_prefix}: solution",
        cmap=SOLUTION_CMAP,
        vmin=-1.0,
        vmax=1.0,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        colorbar_label="u",
    )
    _save_scalar_plot(
        x_grid,
        y_grid,
        error,
        error_path,
        title=f"{title_prefix}: log10 abs error",
        cmap=ERROR_CMAP,
        vmin=-14.0,
        vmax=-3.0,
        ticks=[-14.0, -11.0, -8.0, -5.0, -3.0],
        colorbar_label="log10(abs(error))",
    )
    return solution_path, error_path


def _normal_derivative(side: Side, boundary: BoundaryPart) -> NDArray[np.floating]:
    gradient = analytic_gradient(side, boundary.pointinfo.flat_positions)
    normals = boundary.pointinfo.flat_normals
    values = np.sum(gradient * normals, axis=0)
    return values.reshape(boundary.panel_count, boundary.quadrature_order).T


def _field_grid(side: Side, grid_size: int) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.bool_]]:
    bounds = (-1.15, 1.15) if side == "interior" else (-2.0, 2.0)
    axis = np.linspace(bounds[0], bounds[1], int(grid_size))
    x_grid, y_grid = np.meshgrid(axis, axis)
    if side == "interior":
        mask = (np.abs(x_grid) <= 0.995) & (np.abs(y_grid) <= 0.995)
    else:
        mask = (np.abs(x_grid) >= 1.005) | (np.abs(y_grid) >= 1.005)
    return x_grid, y_grid, mask


def _save_scalar_plot(
    x_grid: NDArray[np.floating],
    y_grid: NDArray[np.floating],
    data: NDArray[np.floating],
    path: Path,
    *,
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    ticks: list[float],
    colorbar_label: str,
) -> None:
    active_cmap = cmap.copy()
    active_cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
    mesh = ax.pcolormesh(
        x_grid[0],
        y_grid[:, 0],
        np.ma.masked_invalid(data),
        shading="auto",
        cmap=active_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    outline = np.column_stack((SQUARE_VERTICES, SQUARE_VERTICES[:, :1]))
    ax.plot(outline[0], outline[1], color="black", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
    ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    colorbar = fig.colorbar(mesh, ax=ax, ticks=ticks, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
