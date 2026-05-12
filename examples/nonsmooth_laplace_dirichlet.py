"""Dirichlet Laplace BVPs on a square with true corners.

Run from the repository root:

    uv run python examples/nonsmooth_laplace_dirichlet.py

This is the shortest useful recipe for a nonsmooth Dirichlet solve:

1. Build a dyadically refined polygon.
2. Solve S sigma + c = g with zero net charge.
3. Evaluate the single-layer potential off boundary with sparse near
   corrections.
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
    exterior_solution,
    interior_solution,
    make_square,
    target_error,
    write_solution_plots,
)


def solve_dirichlet(s_mat: np.ndarray, weights: np.ndarray, boundary_values: np.ndarray):
    """Solve the 2D Laplace single-layer Dirichlet system."""

    system = np.block(
        [
            [s_mat, np.ones((s_mat.shape[0], 1))],
            [weights[None, :], np.zeros((1, 1))],
        ]
    )
    sigma_and_const = np.linalg.solve(system, np.concatenate((boundary_values, [0.0])))
    return sigma_and_const[:-1], float(sigma_and_const[-1])


def run_case(
    *,
    side: str,
    title: str,
    chnkr,
    s_mat: np.ndarray,
    weights: np.ndarray,
    truth_fn,
    check_targets: np.ndarray,
    output_dir: Path,
    grid_size: int,
) -> None:
    boundary_values = truth_fn(boundary_nodes(chnkr))
    sigma, const = solve_dirichlet(s_mat, weights, boundary_values)

    boundary_residual = np.max(np.abs(s_mat @ sigma + const - boundary_values))
    check_error = target_error(chnkr, sigma, const, check_targets, truth_fn)

    # The plotting helper evaluates close grid targets with opts["cormat"].
    pngs = write_solution_plots(
        output_dir,
        f"{side}_dirichlet",
        title,
        chnkr,
        sigma,
        const,
        side,
        truth_fn,
        grid_size,
    )

    print(f"{title} boundary residual: {boundary_residual:.3e}")
    print(f"{title} target max error: {check_error:.3e}")
    for label, path in pngs.items():
        print(f"{title} {label} PNG: {path}")


def run_demo(output_dir: Path, depth: int, grid_size: int) -> None:
    chnkr = make_square(depth)
    weights = chnkr.wts.reshape(-1, order="F")

    # Both interior and exterior Dirichlet cases use the same single-layer
    # matrix; only the boundary data changes.
    s_mat = chunkermat(chnkr, kernel("lap", "s"))

    print(f"dyadic square: depth {depth}, {chnkr.nch} chunks, {chnkr.npt} nodes")
    run_case(
        side="interior",
        title="Interior Dirichlet",
        chnkr=chnkr,
        s_mat=s_mat,
        weights=weights,
        truth_fn=interior_solution,
        check_targets=INTERIOR_TARGETS,
        output_dir=output_dir,
        grid_size=grid_size,
    )
    run_case(
        side="exterior",
        title="Exterior Dirichlet",
        chnkr=chnkr,
        s_mat=s_mat,
        weights=weights,
        truth_fn=exterior_solution,
        check_targets=EXTERIOR_TARGETS,
        output_dir=output_dir,
        grid_size=grid_size,
    )


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "output" / "nonsmooth_laplace_dirichlet"
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
