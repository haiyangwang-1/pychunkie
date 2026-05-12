"""RCIP corner-compression diagnostic on a square chunkgraph.

Run from the repository root:

    uv run python examples/nonsmooth_laplace_rcip.py

This intentionally small script only shows how to call the exposed RCIP driver.
The Dirichlet and Neumann BVP examples keep their linear solves separate.
"""

from __future__ import annotations

import argparse

import numpy as np

from chunkie import chunkgraph, kernel
from chunkie.quadrature import rcip

from _nonsmooth_laplace_common import SQUARE_EDGES, SQUARE_VERTS


def run_demo(nsub: int) -> None:
    cg = chunkgraph(
        SQUARE_VERTS,
        SQUARE_EDGES,
        pref={"k": 8, "nchmax": 1000},
        cparams={"nchmin": 2},
    )
    result = rcip.chunkgraph_rcip(
        cg,
        kernel("lap", "d"),
        1,
        opts={"nsub": nsub, "rcip_savedepth": nsub},
    )
    deviations = [np.linalg.norm(rmat - np.eye(rmat.shape[0]), ord="fro") for rmat in result.R]
    block_size = int(result.R[0].shape[0] if result.R else 0)

    print(f"corner blocks: {result.vertices.size}")
    print(f"RCIP block size: {block_size}")
    print(f"max ||R-I||_F: {max(deviations, default=0.0):.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsub", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    run_demo(parse_args().nsub)


if __name__ == "__main__":
    main()
