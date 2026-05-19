from __future__ import annotations

import ast
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def test_example_scripts_are_standalone_without_private_helpers():
    for path in EXAMPLES_DIR.glob("*.py"):
        assert not path.name.startswith("_")
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "__future__":
                    continue
                assert not node.module.startswith("_"), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("_"), f"{path.name} imports {alias.name}"
