"""Run the split nonsmooth-square Laplace examples.

For learning from the code, read these smaller scripts first:

    examples/nonsmooth_laplace_dirichlet.py
    examples/nonsmooth_laplace_neumann.py
    examples/nonsmooth_laplace_rcip.py

This wrapper is kept for the old command:

    uv run python examples/nonsmooth_laplace_polygon.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _nonsmooth_laplace_common import DEFAULT_DEPTH, DEFAULT_GRID_SIZE
from nonsmooth_laplace_dirichlet import run_demo as run_dirichlet_demo
from nonsmooth_laplace_neumann import run_demo as run_neumann_demo
from nonsmooth_laplace_rcip import run_demo as run_rcip_demo


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent / "output" / "nonsmooth_laplace_polygon"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--rcip-nsub", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("RCIP diagnostic")
    run_rcip_demo(args.rcip_nsub)
    print("\nDirichlet BVPs")
    run_dirichlet_demo(args.output_dir / "dirichlet", args.depth, args.grid_size)
    print("\nNeumann BVPs")
    run_neumann_demo(args.output_dir / "neumann", args.depth, args.grid_size)


if __name__ == "__main__":
    main()
