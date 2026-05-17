from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("script_name", "expected_output"),
    [
        ("accelerated_fmm_laplace.py", "Laplace single layer FMM relative error"),
        ("accelerated_fmm_helmholtz.py", "Helmholtz single layer FMM relative error"),
        ("accelerated_fmm_stokes.py", "Stokes single layer FMM relative error"),
        ("accelerated_fmm_kernels.py", "Helmholtz double layer FMM relative error"),
        ("accelerated_fmm_biharmonic.py", "Biharmonic FMM backend unavailable"),
        ("accelerated_flam_laplace.py", "FLAM Laplace system solve residual"),
    ],
)
def test_accelerated_example_script_runs(script_name: str, expected_output: str):
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script_name)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_output in completed.stdout
