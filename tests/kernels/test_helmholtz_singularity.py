from types import SimpleNamespace

import numpy as np

from chunkie.kernels import kernel


def test_helmholtz_single_layer_declares_smooth_log_remainder():
    source, target = _near_source_target()
    helmholtz_s = kernel("helmholtz", selector="s", wavenumber=1.7 + 0.2j)

    remainder = helmholtz_s(source, target) - helmholtz_s.singularity.expansion.evaluate(source, target)

    assert helmholtz_s.singularity.remainder_regular == "smooth"
    assert helmholtz_s.singularity.legacy_strength == "log"
    assert np.all(np.isfinite(remainder))
    assert np.max(np.abs(remainder)) < 1.0


def test_helmholtz_derivative_selectors_subtract_finite_remainders():
    source, target = _near_source_target(with_normals=True)

    for selector in ("sg", "d", "sp", "dg", "dp"):
        helmholtz_kernel = kernel("helmholtz", selector=selector, wavenumber=1.3)
        remainder = helmholtz_kernel(source, target) - helmholtz_kernel.singularity.expansion.evaluate(
            source,
            target,
        )

        assert np.all(np.isfinite(remainder)), selector
        assert np.max(np.abs(remainder)) < 100.0, selector


def _near_source_target(*, with_normals: bool = False):
    eps = np.array([1.0e-2, 3.0e-3, 1.0e-3])
    angles = np.array([0.0, 0.4, 1.2, 2.1])
    radii, theta = np.meshgrid(eps, angles, indexing="ij")
    target_positions = np.vstack((radii.ravel() * np.cos(theta).ravel(), radii.ravel() * np.sin(theta).ravel()))
    source_positions = np.array([[0.0], [0.0]])
    if not with_normals:
        return source_positions, target_positions

    source = SimpleNamespace(positions=source_positions, normals=np.array([[1.0], [0.0]]))
    target = SimpleNamespace(
        positions=target_positions,
        normals=np.tile(np.array([[0.0], [1.0]]), (1, target_positions.shape[1])),
    )
    return source, target
