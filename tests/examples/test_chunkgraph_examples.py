from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize(
    ("script_name", "expected_output"),
    [
        ("chunkgraph_region_classification.py", "regions at sample targets: [1, 1, 1, 0, 2]"),
        ("chunkgraph_annular_dirichlet.py", "annular-region Dirichlet max error"),
    ],
)
def test_chunkgraph_example_script_runs(script_name: str, expected_output: str):
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script_name)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_output in completed.stdout
