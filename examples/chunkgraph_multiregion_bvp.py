"""Run the split chunkgraph multi-region examples.

For learning from the code, read these smaller scripts first:

    examples/chunkgraph_region_classification.py
    examples/chunkgraph_annular_dirichlet.py

This wrapper is kept for the old command:

    uv run python examples/chunkgraph_multiregion_bvp.py
"""

from __future__ import annotations

from chunkgraph_annular_dirichlet import main as run_annular_dirichlet
from chunkgraph_region_classification import main as run_region_classification


def main() -> None:
    print("Region classification")
    run_region_classification()
    print("\nAnnular Dirichlet solve")
    run_annular_dirichlet()


if __name__ == "__main__":
    main()
