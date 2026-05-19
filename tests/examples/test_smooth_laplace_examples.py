from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_example(script_name: str):
    spec = importlib.util.spec_from_file_location(script_name[:-3], EXAMPLES_DIR / script_name)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "side", "condition", "tolerance"),
    [
        ("smooth_laplace_interior_dirichlet.py", "interior", "dirichlet", 5.0e-7),
        ("smooth_laplace_exterior_dirichlet.py", "exterior", "dirichlet", 5.0e-7),
        ("smooth_laplace_interior_neumann.py", "interior", "neumann", 5.0e-3),
        ("smooth_laplace_exterior_neumann.py", "exterior", "neumann", 5.0e-3),
    ],
)
def test_smooth_laplace_example_solves_and_writes_figures(
    tmp_path: Path,
    script_name: str,
    side: str,
    condition: str,
    tolerance: float,
):
    example = load_example(script_name)

    result = example.solve_case(
        side,
        condition,
        grid_size=32,
        output_dir=tmp_path,
    )

    assert result["residual_norm"] < 1.0e-10
    assert result["max_error"] < tolerance
    assert result["solution_path"].exists()
    assert result["solution_path"].stat().st_size > 0
    assert result["error_path"].exists()
    assert result["error_path"].stat().st_size > 0
