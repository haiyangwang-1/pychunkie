"""Run the split smooth-circle Laplace BVP examples.

For learning from the code, read these smaller scripts first:

    examples/smooth_laplace_dirichlet.py
    examples/smooth_laplace_neumann.py

This wrapper is kept for the old command:

    uv run python examples/smooth_laplace_bvp.py
"""

from __future__ import annotations

from smooth_laplace_dirichlet import main as run_dirichlet
from smooth_laplace_neumann import main as run_neumann


def main() -> None:
    print("Dirichlet BVPs")
    run_dirichlet()
    print("\nNeumann BVPs")
    run_neumann()


if __name__ == "__main__":
    main()
