"""Shared utilities for the nonsmooth Laplace examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from chunkie import chunkerkerneval, chunkerkernevalmat, chunkerpoly, kernel


DEFAULT_DEPTH = 40
DEFAULT_GRID_SIZE = 80

SQUARE_VERTS = np.array(
    [
        [-1.0, 1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0, 1.0],
    ]
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


def make_square(depth: int = DEFAULT_DEPTH):
    """Build the true-corner square with dyadic corner refinement."""

    return chunkerpoly(
        SQUARE_VERTS,
        {"ifclosed": True, "dyadic": True, "depth": depth, "widths": 0.25},
        {"k": 12, "nchmax": max(2000, 16 * depth)},
    )


def boundary_nodes(chnkr) -> np.ndarray:
    return chnkr.r.reshape(chnkr.dim, chnkr.npt, order="F")


def boundary_normals(chnkr) -> np.ndarray:
    return chnkr.n.reshape(chnkr.dim, chnkr.npt, order="F")


def interior_solution(targets: np.ndarray) -> np.ndarray:
    return np.asarray(targets)[0]


def interior_gradient(targets: np.ndarray) -> np.ndarray:
    pts = np.asarray(targets)
    return np.vstack((np.ones(pts.shape[1]), np.zeros(pts.shape[1])))


def exterior_solution(targets: np.ndarray) -> np.ndarray:
    pts = np.asarray(targets, dtype=float)
    r2 = np.sum(pts**2, axis=0)
    return pts[0] / r2


def exterior_gradient(targets: np.ndarray) -> np.ndarray:
    pts = np.asarray(targets, dtype=float)
    x = pts[0]
    y = pts[1]
    r2 = x**2 + y**2
    r4 = r2**2
    return np.vstack(((y**2 - x**2) / r4, -2.0 * x * y / r4))


def corrected_single_layer(chnkr, sigma: np.ndarray, const: float, targets: np.ndarray) -> np.ndarray:
    """Evaluate S sigma + const using MATLAB-style sparse near corrections."""

    lap_s = kernel("lap", "s")
    cormat = chunkerkernevalmat(chnkr, lap_s, targets, {"corrections": True, "fac": 1.0})
    vals = chunkerkerneval(
        chnkr,
        lap_s,
        sigma,
        targets,
        {"forcesmooth": True, "cormat": cormat},
    )
    return vals.reshape(-1) + const


def constant_fit(chnkr, sigma: np.ndarray, targets: np.ndarray, truth: np.ndarray) -> float:
    """Fit the additive constant left undetermined by a Neumann solve."""

    vals = corrected_single_layer(chnkr, sigma, 0.0, targets)
    return float(np.mean(truth - vals))


def target_error(chnkr, sigma: np.ndarray, const: float, targets: np.ndarray, truth_fn) -> float:
    vals = corrected_single_layer(chnkr, sigma, const, targets)
    return float(np.max(np.abs(vals - truth_fn(targets))))


def write_solution_plots(
    output_dir: Path,
    stem: str,
    title: str,
    chnkr,
    sigma: np.ndarray,
    const: float,
    side: str,
    truth_fn,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> dict[str, Path]:
    """Write solution and log10(abs(error)) PNGs for a single layer potential."""

    xs, ys, domain, targets, truth = _grid(side, truth_fn, grid_size)
    values = np.full(domain.shape, np.nan)
    values[domain] = corrected_single_layer(chnkr, sigma, const, targets)
    error = np.log10(np.maximum(np.abs(values - truth), 1e-16))

    paths = {
        "solution": output_dir / f"{stem}_solution.png",
        "error": output_dir / f"{stem}_error_log10.png",
    }
    _plot_field(
        paths["solution"],
        values,
        xs,
        ys,
        f"{title}: layer potential",
        cmap=SOLUTION_CMAP,
        vmin=-1.0,
        vmax=1.0,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        label="u",
    )
    _plot_field(
        paths["error"],
        error,
        xs,
        ys,
        f"{title}: log10 abs error",
        cmap=ERROR_CMAP,
        vmin=-14.0,
        vmax=-3.0,
        ticks=[-14, -11, -8, -5, -3],
        label="log10 |error|",
    )
    return paths


def _grid(side: str, truth_fn, grid_size: int):
    bounds = (-1.15, 1.15) if side == "interior" else (-2.0, 2.0)
    xs = np.linspace(bounds[0], bounds[1], grid_size)
    ys = np.linspace(bounds[0], bounds[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    if side == "interior":
        domain = (np.abs(xx) <= 0.995) & (np.abs(yy) <= 0.995)
    elif side == "exterior":
        domain = (np.abs(xx) >= 1.005) | (np.abs(yy) >= 1.005)
    else:
        raise ValueError("side must be 'interior' or 'exterior'")
    targets = np.vstack((xx[domain], yy[domain]))
    truth = np.full(xx.shape, np.nan)
    truth[domain] = truth_fn(targets)
    return xs, ys, domain, targets, truth


def _plot_field(
    path: Path,
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    *,
    cmap,
    vmin: float,
    vmax: float,
    ticks: list[float],
    label: str,
) -> None:
    cmap = cmap.copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
    mesh = ax.pcolormesh(
        xs,
        ys,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    outline = np.column_stack((SQUARE_VERTS, SQUARE_VERTS[:, :1]))
    ax.plot(outline[0], outline[1], color="black", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(xs[0]), float(xs[-1]))
    ax.set_ylim(float(ys[0]), float(ys[-1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax, ticks=ticks, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
