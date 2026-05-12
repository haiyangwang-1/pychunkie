from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import numpy as np
from scipy.io import loadmat

from chunkie import Chunker
from chunkie.operators import PointInfo


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


def point_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def pointinfo_from_mat(obj) -> PointInfo:
    return PointInfo(
        r=point_array(obj.r),
        d=point_array(obj.d) if hasattr(obj, "d") else None,
        d2=point_array(obj.d2) if hasattr(obj, "d2") else None,
        n=point_array(obj.n) if hasattr(obj, "n") else None,
        data=point_array(obj.data) if hasattr(obj, "data") else None,
    )


def chunker_from_fields(fields, *, nchmax: int | None = None) -> Chunker:
    k = int(fields.k)
    nch = int(fields.nch)
    dim = int(fields.dim)
    if nchmax is None:
        nchmax = max(2 * nch, nch + 16, 1)
    chnkr = Chunker(
        {"k": k, "dim": dim, "nchstor": nch, "nchmax": int(nchmax)},
        np.asarray(fields.tstor).reshape(-1),
        np.asarray(fields.wstor).reshape(-1),
    )
    chnkr.addchunk(nch)
    chnkr.r = np.asarray(fields.r)
    chnkr.d = np.asarray(fields.d)
    chnkr.d2 = np.asarray(fields.d2)
    chnkr.n = np.asarray(fields.n)
    chnkr.wts = np.asarray(fields.wts)
    chnkr.adj = np.asarray(fields.adj, dtype=int)
    return chnkr


def assert_chunker_matches_fields(
    chnkr: Chunker,
    fields,
    label: str = "",
    *,
    rtol: float = 1e-7,
    atol: float = 1e-12,
    check_area: bool = True,
    check_chunklen: bool = True,
) -> None:
    prefix = f"{label}: " if label else ""
    np.testing.assert_allclose(chnkr.r, fields.r, rtol=rtol, atol=atol, err_msg=f"{prefix}r")
    np.testing.assert_allclose(chnkr.d, fields.d, rtol=rtol, atol=atol, err_msg=f"{prefix}d")
    np.testing.assert_allclose(chnkr.d2, fields.d2, rtol=rtol, atol=atol, err_msg=f"{prefix}d2")
    np.testing.assert_allclose(chnkr.n, fields.n, rtol=rtol, atol=atol, err_msg=f"{prefix}n")
    np.testing.assert_allclose(chnkr.wts, fields.wts, rtol=rtol, atol=atol, err_msg=f"{prefix}wts")
    np.testing.assert_array_equal(chnkr.adj, np.asarray(fields.adj, dtype=int), err_msg=f"{prefix}adj")
    if check_area:
        np.testing.assert_allclose(chnkr.area(), fields.area, rtol=rtol, atol=atol, err_msg=f"{prefix}area")
    if check_chunklen:
        np.testing.assert_allclose(
            chnkr.chunklen(),
            fields.chunklen,
            rtol=rtol,
            atol=atol,
            err_msg=f"{prefix}chunklen",
        )
