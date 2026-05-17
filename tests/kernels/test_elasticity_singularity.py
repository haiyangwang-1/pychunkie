import numpy as np

from chunkie.kernels import kernel


def test_elasticity_single_displacement_matches_laplace_basis_expansion():
    source, target = _near_source_target()
    elasticity_s = kernel("elasticity", selector="s", lame_lambda=2.0, lame_mu=3.0)

    remainder = elasticity_s(source, target) - elasticity_s.singularity.expansion.evaluate(source, target)

    np.testing.assert_allclose(remainder, 0.0, atol=1.0e-12)
    assert elasticity_s.singularity.remainder_regular == "smooth"
    assert elasticity_s.singularity.legacy_strength == "mixed"


def _near_source_target():
    eps = np.array([1.0e-2, 3.0e-3, 1.0e-3])
    angles = np.array([0.0, 0.6, 1.4, 2.8])
    radii, theta = np.meshgrid(eps, angles, indexing="ij")
    target_positions = np.vstack((radii.ravel() * np.cos(theta).ravel(), radii.ravel() * np.sin(theta).ravel()))
    source_positions = np.array([[0.0], [0.0]])
    return source_positions, target_positions
