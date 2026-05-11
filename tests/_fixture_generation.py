from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"

_GENERATOR_BY_FIXTURE = {
    "chunker_circle.mat": "generate_chunker_circle_fixture.m",
    "chunker_ops.mat": "generate_chunker_ops_fixture.m",
    "devtools_easy.mat": "generate_devtools_easy_fixture.m",
    "geometry_core.mat": "generate_geometry_core_fixture.m",
    "kernel_pointinfo.mat": "generate_kernel_pointinfo_fixture.m",
    "lege_basic.mat": "generate_lege_basic_fixture.m",
    "lege_extended.mat": "generate_lege_extended_fixture.m",
    "operator_parity.mat": "generate_operator_parity_fixture.m",
    "quadggq.mat": "generate_quadggq_fixture.m",
    "rcip.mat": "generate_rcip_fixture.m",
}


def ensure_matlab_fixture(name: str) -> Path:
    if name not in _GENERATOR_BY_FIXTURE:
        pytest.fail(f"no MATLAB fixture generator registered for tests/golden/{name}")

    path = GOLDEN / name
    if path.exists():
        return path

    generator = ROOT / "scripts" / "matlab" / _GENERATOR_BY_FIXTURE[name]
    if not generator.exists():
        pytest.fail(f"missing MATLAB fixture generator: {generator.relative_to(ROOT)}")

    matlab_startup = ROOT / "external" / "chunkie-matlab" / "startup.m"
    if not matlab_startup.exists():
        pytest.fail(
            f"cannot generate tests/golden/{name}: missing MATLAB reference checkout at "
            "external/chunkie-matlab/startup.m"
        )

    matlab = os.environ.get("PYCHUNKIE_MATLAB", "matlab")
    if shutil.which(matlab) is None and not Path(matlab).exists():
        pytest.fail(
            f"cannot generate tests/golden/{name}: MATLAB executable {matlab!r} was not found. "
            "Set PYCHUNKIE_MATLAB to the MATLAB executable path."
        )

    timeout = int(os.environ.get("PYCHUNKIE_MATLAB_FIXTURE_TIMEOUT", "1800"))
    batch_path = generator.as_posix().replace("'", "''")
    batch = f"run('{batch_path}')"
    result = subprocess.run(
        [matlab, "-batch", batch],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"generating tests/golden/{name} failed with exit code {result.returncode}.\n"
            f"Command: {matlab} -batch {batch}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not path.exists():
        pytest.fail(
            f"generator {_GENERATOR_BY_FIXTURE[name]} completed but did not create "
            f"tests/golden/{name}"
        )
    return path


def load_generated_mat_fixture(name: str, **kwargs):
    return loadmat(ensure_matlab_fixture(name), **kwargs)
