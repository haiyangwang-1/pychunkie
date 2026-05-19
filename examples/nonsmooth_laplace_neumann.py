"""Run both square-corner Laplace Neumann examples."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.polynomial.legendre import leggauss

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chunkie.geometry import (
    Chunker,
    ChunkGraph,
    GraphEdge,
    GraphRegion,
    GraphVertex,
    RegionCycle,
    SignedEdge,
)
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

CASES = (("interior", "neumann"), ("exterior", "neumann"))
QUADRATURE_ORDER = 12
GRID_SIZE = 80
RCIP_SUBDIVISIONS = 20
SQUARE_DEPTH = 2
SQUARE_VERTICES = np.array([[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]])
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


def analytic_solution(side: str, targets):
    points = np.asarray(targets, dtype=float)
    if side == "interior":
        return points[0]
    return points[0] / np.sum(points * points, axis=0)


def analytic_gradient(side: str, targets):
    points = np.asarray(targets, dtype=float)
    if side == "interior":
        return np.vstack((np.ones(points.shape[1]), np.zeros(points.shape[1])))
    x = points[0]
    y = points[1]
    radius2 = x * x + y * y
    radius4 = radius2 * radius2
    return np.vstack(((y * y - x * x) / radius4, -2.0 * x * y / radius4))


def make_square_graph(depth: int = SQUARE_DEPTH, quadrature_order: int = QUADRATURE_ORDER):
    panel_count = 2 ** int(depth)
    vertices = [
        GraphVertex(
            id=vertex_id,
            position=SQUARE_VERTICES[:, vertex_id],
            incident_edges=tuple(np.where(SQUARE_EDGES == vertex_id)[1]),
        )
        for vertex_id in range(SQUARE_VERTICES.shape[1])
    ]
    edges = []
    for edge_id, (start_vertex, end_vertex) in enumerate(SQUARE_EDGES.T):
        edges.append(
            GraphEdge(
                id=edge_id,
                chunker=line_chunker(
                    SQUARE_VERTICES[:, start_vertex],
                    SQUARE_VERTICES[:, end_vertex],
                    quadrature_order=quadrature_order,
                    panel_count=panel_count,
                ),
                start_vertex=int(start_vertex),
                end_vertex=int(end_vertex),
                orientation=1,
                left_region=1,
                right_region=0,
            )
        )
    exterior = GraphRegion(0, (), bounded=False, label="exterior")
    interior = GraphRegion(
        1,
        (RegionCycle(tuple(SignedEdge(edge_id=edge_id, orientation=1) for edge_id in range(4))),),
        bounded=True,
        label="interior",
    )
    return ChunkGraph(vertices=vertices, edges=edges, regions=[exterior, interior])


def line_chunker(start, end, *, quadrature_order: int, panel_count: int):
    nodes, reference_weights = leggauss(int(quadrature_order))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    breaks = np.linspace(0.0, 1.0, int(panel_count) + 1)
    positions = np.empty((2, nodes.size, int(panel_count)), dtype=float)
    derivatives = np.empty_like(positions)
    second = np.zeros_like(positions)
    weights = np.empty((nodes.size, int(panel_count)), dtype=float)
    for panel_id in range(int(panel_count)):
        midpoint = 0.5 * (breaks[panel_id] + breaks[panel_id + 1])
        half_width = 0.5 * (breaks[panel_id + 1] - breaks[panel_id])
        parameters = midpoint + half_width * nodes
        positions[:, :, panel_id] = start[:, None] + direction[:, None] * parameters[None, :]
        derivatives[:, :, panel_id] = direction[:, None] * half_width
        weights[:, panel_id] = reference_weights * np.linalg.norm(direction * half_width)
    speed = np.linalg.norm(derivatives, axis=0)
    normals = np.empty_like(derivatives)
    normals[0] = derivatives[1] / speed
    normals[1] = -derivatives[0] / speed
    adjacency = np.full((2, int(panel_count)), -1, dtype=np.int64)
    for panel_id in range(int(panel_count)):
        if panel_id > 0:
            adjacency[0, panel_id] = panel_id - 1
        if panel_id + 1 < int(panel_count):
            adjacency[1, panel_id] = panel_id + 1
    return Chunker(
        positions=positions,
        derivatives=derivatives,
        second_derivatives=second,
        normals=normals,
        weights=weights,
        nodes=nodes,
        reference_weights=reference_weights,
        adjacency=adjacency,
        closed=False,
        orientation="open",
    )


def normal_derivative(side: str, boundary):
    gradient = analytic_gradient(side, boundary.pointinfo.flat_positions)
    values = np.sum(gradient * boundary.pointinfo.flat_normals, axis=0)
    return values.reshape(boundary.panel_count, boundary.quadrature_order).T


def build_system(side: str, condition: str, quadrature_order: int = QUADRATURE_ORDER):
    graph = make_square_graph(quadrature_order=quadrature_order)
    boundary = graph.boundary(1, side="interior")
    density = DensitySpace("sigma", boundary)
    if condition == "dirichlet":
        trace_kernel = kernel("laplace", selector="d")
        trace_coefficient = -2.0 if side == "interior" else 2.0
        field_kernel = trace_kernel
        field_coefficient = trace_coefficient
        rhs = analytic_solution(side, boundary.pointinfo.positions)
        layer_name = "double"
    else:
        trace_kernel = kernel("laplace", selector="sp")
        trace_coefficient = 2.0 if side == "interior" else -2.0
        field_kernel = kernel("laplace", selector="s")
        field_coefficient = 1.0
        rhs = trace_coefficient * normal_derivative(side, boundary)
        layer_name = "single"

    trace_layer = LayerPotential(
        f"{layer_name}_trace",
        boundary,
        trace_kernel,
        "sigma",
        coefficient=trace_coefficient,
    )
    field_layer = LayerPotential(
        layer_name,
        boundary,
        field_kernel,
        "sigma",
        coefficient=field_coefficient,
    )
    trace = BoundaryTrace(trace_layer, boundary, side, jump=JumpTerm(1.0, "sigma"))
    system = IntegralSystem(
        f"nonsmooth_laplace_{side}_{condition}",
        graph,
        (density,),
        (BoundaryEquation(condition, boundary, (trace,), rhs),),
        fields={"u": (field_layer,)},
    )
    return graph, boundary, system


def field_grid(side: str, grid_size: int):
    bounds = (-1.15, 1.15) if side == "interior" else (-2.0, 2.0)
    axis = np.linspace(bounds[0], bounds[1], int(grid_size))
    x_grid, y_grid = np.meshgrid(axis, axis)
    if side == "interior":
        mask = (np.abs(x_grid) <= 0.995) & (np.abs(y_grid) <= 0.995)
    else:
        mask = (np.abs(x_grid) >= 1.005) | (np.abs(y_grid) >= 1.005)
    return x_grid, y_grid, mask


def save_scalar_plot(
    x_grid, y_grid, data, path: Path, *, title: str, cmap, vmin, vmax, ticks, colorbar_label
):
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
    ax.plot(outline[0], outline[1], color="black", linestyle="-", linewidth=0.7)
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


def solve_case(
    side: str,
    condition: str,
    *,
    quadrature_order: int = QUADRATURE_ORDER,
    grid_size: int = GRID_SIZE,
    output_dir: Path | str | None = None,
):
    graph, boundary, system = build_system(side, condition, quadrature_order)
    solve_config = SystemConfig(rcip_subdivisions=RCIP_SUBDIVISIONS)
    evaluation_config = SystemConfig(rcip_subdivisions=RCIP_SUBDIVISIONS)
    solution = system.solve(config=solve_config)
    x_grid, y_grid, mask = field_grid(side, grid_size)
    grid_targets = np.vstack((x_grid[mask], y_grid[mask]))
    grid_values = np.asarray(
        solution.evaluate(grid_targets, config=evaluation_config).values[0],
        dtype=complex,
    ).real
    grid_truth = analytic_solution(side, grid_targets)
    additive_constant = float(np.mean(grid_truth - grid_values)) if condition == "neumann" else 0.0
    grid_values = grid_values + additive_constant

    target_points = INTERIOR_TARGETS if side == "interior" else EXTERIOR_TARGETS
    target_values = np.asarray(
        solution.evaluate(target_points, config=evaluation_config).values[0],
        dtype=complex,
    ).real
    target_values = target_values + additive_constant
    target_truth = analytic_solution(side, target_points)

    destination = Path(
        os.environ.get("CHUNKIE_EXAMPLE_OUTPUT_DIR", Path(__file__).resolve().parent)
    )
    if output_dir is not None:
        destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"nonsmooth_laplace_{side}_{condition}"
    field = np.full(x_grid.shape, np.nan)
    field[mask] = grid_values
    error = np.full(x_grid.shape, np.nan)
    error[mask] = np.log10(np.maximum(np.abs(grid_values - grid_truth), 1.0e-16))
    solution_path = destination / f"{stem}_solution.png"
    error_path = destination / f"{stem}_error_log10.png"
    title_prefix = f"Nonsmooth Laplace {side} {condition}"
    save_scalar_plot(
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
    save_scalar_plot(
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
    rcip_state = (
        solution.operator.diagnostics.get("rcip") if solution.operator is not None else None
    )
    return {
        "graph": graph,
        "boundary": boundary,
        "rcip_corner_count": len(rcip_state.corners) if rcip_state is not None else 0,
        "residual_norm": float(np.linalg.norm(solution.residual)),
        "max_error": float(np.max(np.abs(target_values - target_truth))),
        "grid_max_error": float(np.max(np.abs(grid_values - grid_truth))),
        "solution_path": solution_path,
        "error_path": error_path,
    }


def main() -> None:
    for side, condition in CASES:
        result = solve_case(side, condition)
        print(
            f"square graph: {len(result['graph'].edges)} edges, {result['boundary'].point_count} nodes, "
            f"{result['rcip_corner_count']} RCIP corners"
        )
        print(
            f"{side} {condition} residual: {result['residual_norm']:.3e}; "
            f"target max error: {result['max_error']:.3e}; grid max error: {result['grid_max_error']:.3e}"
        )
        print(f"solution figure: {result['solution_path']}")
        print(f"log10 abs error figure: {result['error_path']}")


if __name__ == "__main__":
    main()
