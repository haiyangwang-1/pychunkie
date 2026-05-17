from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
SPEC = importlib.util.spec_from_file_location(
    "nonsmooth_laplace_common",
    EXAMPLES_DIR / "_nonsmooth_laplace_common.py",
)
assert SPEC is not None
nonsmooth_laplace_common = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nonsmooth_laplace_common
assert SPEC.loader is not None
SPEC.loader.exec_module(nonsmooth_laplace_common)


@pytest.mark.parametrize(
    ("side", "condition", "tolerance"),
    [
        ("interior", "dirichlet", 1.0e-6),
        ("exterior", "dirichlet", 1.0e-3),
        ("interior", "neumann", 1.0e-3),
        ("exterior", "neumann", 1.0e-6),
    ],
)
def test_nonsmooth_laplace_example_solves_and_writes_figures(
    tmp_path: Path, side: str, condition: str, tolerance: float
):
    result = nonsmooth_laplace_common.solve_case(
        side,
        condition,
        quadrature_order=24,
        grid_size=28,
        output_dir=tmp_path,
    )

    assert result.rcip_corner_count == 4
    assert result.residual_norm < 1.0e-10
    assert result.max_error < tolerance
    assert result.solution_path.exists()
    assert result.solution_path.stat().st_size > 0
    assert result.error_path.exists()
    assert result.error_path.stat().st_size > 0
