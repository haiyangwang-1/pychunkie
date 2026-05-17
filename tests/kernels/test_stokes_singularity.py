from types import SimpleNamespace

import numpy as np

from chunkie.kernels import kernel


def test_stokes_single_velocity_has_finite_smooth_remainder():
    source, target = _near_source_target()
    stokes_s = kernel("stokes", selector="s", viscosity=2.0)

    remainder = stokes_s(source, target) - stokes_s.singularity.expansion.evaluate(source, target)

    assert stokes_s.singularity.remainder_regular == "smooth"
    assert stokes_s.singularity.legacy_strength == "mixed"
    assert np.all(np.isfinite(remainder))
    np.testing.assert_allclose(remainder[0, 1], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(remainder[1, 0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(remainder[0, 0], -1.0 / (16.0 * np.pi), atol=1.0e-12)
    np.testing.assert_allclose(remainder[1, 1], -1.0 / (16.0 * np.pi), atol=1.0e-12)


def test_stokes_double_velocity_matches_laplace_basis_expansion():
    source, target = _near_source_target(with_normals=True)
    stokes_d = kernel("stokes", selector="d")

    remainder = stokes_d(source, target) - stokes_d.singularity.expansion.evaluate(source, target)

    np.testing.assert_allclose(remainder, 0.0, atol=1.0e-12)
    assert stokes_d.singularity.legacy_strength == "mixed"
    assert stokes_d.singularity.boundary_limit == "pv"


def _near_source_target(*, with_normals: bool = False):
    eps = np.array([1.0e-2, 3.0e-3, 1.0e-3])
    angles = np.array([0.0, 0.6, 1.4, 2.8])
    radii, theta = np.meshgrid(eps, angles, indexing="ij")
    target_positions = np.vstack((radii.ravel() * np.cos(theta).ravel(), radii.ravel() * np.sin(theta).ravel()))
    source_positions = np.array([[0.0], [0.0]])
    if not with_normals:
        return source_positions, target_positions

    source = SimpleNamespace(positions=source_positions, normals=np.array([[1.0], [0.0]]))
    return source, target_positions
