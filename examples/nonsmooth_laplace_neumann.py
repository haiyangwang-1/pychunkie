"""Neumann Laplace BVPs on a square with true corners.

Run from the repository root:

    uv run python examples/nonsmooth_laplace_neumann.py

This example uses a single-layer representation. For a Neumann problem, the
normal derivative of S sigma jumps by +/- 1/2 sigma, so the boundary system is

    (jump * I + S' + onesmat) sigma = du/dn.

The onesmat term removes the constant-density nullspace. Interior Neumann
solutions are determined only up to an additive constant, so the constant is
fitted at a few target points after the density is solved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chunkie import chunkermat, kernel

from _nonsmooth_laplace_common import (
    DEFAULT_DEPTH,
    DEFAULT_GRID_SIZE,
    EXTERIOR_TARGETS,
    INTERIOR_TARGETS,
    boundary_nodes,
    boundary_normals,
    constant_fit,
    exterior_gradient,
    exterior_solution,
    interior_gradient,
    interior_solution,
    make_square,
    target_error,
    write_solution_plots,
)


def single_layer_neumann_matrix(chnkr, kprime: np.ndarray, side: str) -> np.ndarray:
    """Build jump * I + S' + onesmat for a polygonal Neumann solve."""

    jump = 0.5 if side == "interior" else -0.5
    return jump * np.eye(chnkr.npt) + kprime + chnkr.onesmat()


def run_case(
    *,
    side: str,
    title: str,
    chnkr,
    system: np.ndarray,
    solution_fn,
    gradient_fn,
    check_targets: np.ndarray,
    output_dir: Path,
    grid_size: int,
) -> None:
    boundary = boundary_nodes(chnkr)
    normal_data = np.sum(gradient_fn(boundary) * boundary_normals(chnkr), axis=0)
    sigma = np.linalg.solve(system, normal_data)

    if side == "interior":
        const = constant_fit(chnkr, sigma, check_targets, solution_fn(check_targets))
    else:
        const = 0.0

    boundary_residual = np.max(np.abs(system @ sigma - normal_data))
    check_error = target_error(chnkr, sigma, const, check_targets, solution_fn)
    net_charge = float(np.dot(chnkr.wts.reshape(-1, order="F"), sigma))

    # The plotting helper evaluates close grid targets with opts["cormat"].
    pngs = write_solution_plots(
        output_dir,
        f"{side}_neumann",
        title,
        chnkr,
        sigma,
        const,
        side,
        solution_fn,
        grid_size,
    )

    print(f"{title} boundary residual: {boundary_residual:.3e}")
    print(f"{title} target max error: {check_error:.3e}")
    print(f"{title} net charge after stabilization: {net_charge:.3e}")
    for label, path in pngs.items():
        print(f"{title} {label} PNG: {path}")


def run_demo(output_dir: Path, depth: int, grid_size: int) -> None:
    chnkr = make_square(depth)
    kprime = chunkermat(chnkr, kernel("lap", "sp"))
    kprime[np.diag_indices_from(kprime)] = 0.0
    print(f"dyadic square: depth {depth}, {chnkr.nch} chunks, {chnkr.npt} nodes")

    run_case(
        side="interior",
        title="Interior Neumann",
        chnkr=chnkr,
        system=single_layer_neumann_matrix(chnkr, kprime, "interior"),
        solution_fn=interior_solution,
        gradient_fn=interior_gradient,
        check_targets=INTERIOR_TARGETS,
        output_dir=output_dir,
        grid_size=grid_size,
    )
    run_case(
        side="exterior",
        title="Exterior Neumann",
        chnkr=chnkr,
        system=single_layer_neumann_matrix(chnkr, kprime, "exterior"),
        solution_fn=exterior_solution,
        gradient_fn=exterior_gradient,
        check_targets=EXTERIOR_TARGETS,
        output_dir=output_dir,
        grid_size=grid_size,
    )


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "output" / "nonsmooth_laplace_neumann"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_demo(args.output_dir, args.depth, args.grid_size)


if __name__ == "__main__":
    main()
