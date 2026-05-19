"""Interior Laplace Neumann solve on a smooth unit circle."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chunkie.geometry import circle
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

CASES = (("interior", "neumann"),)


def analytic_solution(side: str, targets):
    points = np.asarray(targets, dtype=float)
    if side == "interior":
        return points[0]
    return points[0] / np.sum(points * points, axis=0)


def build_system(side: str, condition: str):
    boundary = circle(quadrature_order=16, panel_count=24)
    density = DensitySpace("sigma", boundary)
    if condition == "dirichlet":
        trace_kernel = kernel("laplace", selector="d")
        field_kernel = trace_kernel
        jump = -0.5 if side == "interior" else 0.5
        rhs = analytic_solution(side, boundary.positions)
        layer_name = "double"
    else:
        trace_kernel = kernel("laplace", selector="sp")
        field_kernel = kernel("laplace", selector="s")
        jump = 0.5 if side == "interior" else -0.5
        rhs = boundary.normals[0] if side == "interior" else -boundary.positions[0]
        layer_name = "single"

    trace_layer = LayerPotential(f"{layer_name}_trace", boundary, trace_kernel, "sigma")
    field_layer = LayerPotential(layer_name, boundary, field_kernel, "sigma")
    trace = BoundaryTrace(trace_layer, boundary, side, jump=JumpTerm(jump, "sigma"))
    system = IntegralSystem(
        f"smooth_laplace_{side}_{condition}",
        boundary,
        (density,),
        (BoundaryEquation(condition, boundary, (trace,), rhs),),
        fields={"u": (field_layer,)},
    )
    return boundary, system


def field_grid(side: str, grid_size: int):
    extent = 1.2 if side == "interior" else 2.0
    axis = np.linspace(-extent, extent, int(grid_size))
    x_grid, y_grid = np.meshgrid(axis, axis)
    radius = np.hypot(x_grid, y_grid)
    mask = radius < 0.95 if side == "interior" else radius > 1.05
    return x_grid, y_grid, mask


def save_scalar_plot(
    x_grid, y_grid, data, path: Path, *, title: str, colorbar_label: str, cmap: str
):
    fig, ax = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    image = ax.imshow(
        data,
        extent=(float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())),
        origin="lower",
        cmap=cmap,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linestyle="-", linewidth=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def solve_case(
    side: str, condition: str, *, grid_size: int = 96, output_dir: Path | str | None = None
):
    boundary, system = build_system(side, condition)
    solution = system.solve()
    x_grid, y_grid, mask = field_grid(side, grid_size)
    targets = np.vstack((x_grid[mask], y_grid[mask]))
    values = np.asarray(
        solution.evaluate(targets, config=SystemConfig()).values[0],
        dtype=complex,
    ).real
    truth = analytic_solution(side, targets)
    additive_constant = float(np.mean(truth - values)) if condition == "neumann" else 0.0
    values = values + additive_constant
    max_error = float(np.max(np.abs(values - truth)))

    destination = Path(
        os.environ.get("CHUNKIE_EXAMPLE_OUTPUT_DIR", Path(__file__).resolve().parent)
    )
    if output_dir is not None:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"smooth_laplace_{side}_{condition}"

    field = np.full(x_grid.shape, np.nan)
    field[mask] = values
    error = np.full(x_grid.shape, np.nan)
    error[mask] = np.log10(np.maximum(np.abs(values - truth), 1.0e-16))
    solution_path = destination / f"{stem}_solution.png"
    error_path = destination / f"{stem}_error_log10.png"
    title_prefix = f"Smooth Laplace {side} {condition}"
    save_scalar_plot(
        x_grid,
        y_grid,
        field,
        solution_path,
        title=f"{title_prefix}: solution",
        colorbar_label="u",
        cmap="viridis",
    )
    save_scalar_plot(
        x_grid,
        y_grid,
        error,
        error_path,
        title=f"{title_prefix}: log10 abs error",
        colorbar_label="log10(abs(error))",
        cmap="magma",
    )
    return {
        "boundary": boundary,
        "residual_norm": float(np.linalg.norm(solution.residual)),
        "max_error": max_error,
        "solution_path": solution_path,
        "error_path": error_path,
    }


def main() -> None:
    for side, condition in CASES:
        result = solve_case(side, condition)
        print(
            f"unit circle: {result['boundary'].panel_count} panels, {result['boundary'].point_count} nodes"
        )
        print(
            f"{side} {condition} residual: {result['residual_norm']:.3e}; max field error: {result['max_error']:.3e}"
        )
        print(f"solution figure: {result['solution_path']}")
        print(f"log10 abs error figure: {result['error_path']}")


if __name__ == "__main__":
    main()
