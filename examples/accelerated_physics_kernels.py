"""Run the split accelerated-kernel examples.

For learning from the code, read these smaller scripts first:

    examples/accelerated_fmm_kernels.py
    examples/accelerated_flam_laplace.py

This wrapper is kept for the old command:

    uv run python examples/accelerated_physics_kernels.py
"""

from __future__ import annotations

from accelerated_flam_laplace import main as run_flam
from accelerated_fmm_kernels import main as run_fmm


def main() -> None:
    print("FMM target evaluations")
    run_fmm()
    print("\nFLAM shifted solve")
    run_flam()


if __name__ == "__main__":
    main()
