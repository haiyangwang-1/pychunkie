from types import SimpleNamespace

import numpy as np

from chunkie.kernels import kernel


def test_biharmonic_value_and_gradient_match_laplace_basis_expansion():
    source, target = _near_source_target()

    for selector in ("s", "sg", "hessian"):
        biharmonic_kernel = kernel("biharmonic", selector=selector)
        remainder = biharmonic_kernel(source, target) - biharmonic_kernel.singularity.expansion.evaluate(
            source,
            target,
        )

        np.testing.assert_allclose(remainder, 0.0, atol=1.0e-12)
        assert biharmonic_kernel.singularity.remainder_regular == "smooth"


def test_biharmonic_normal_derivatives_match_laplace_basis_expansion():
    source, target = _near_source_target(with_normals=True)

    for selector in ("sp", "d"):
        biharmonic_kernel = kernel("biharmonic", selector=selector)
        remainder = biharmonic_kernel(source, target) - biharmonic_kernel.singularity.expansion.evaluate(
            source,
            target,
        )

        np.testing.assert_allclose(remainder, 0.0, atol=1.0e-12)
        assert biharmonic_kernel.singularity.legacy_strength == "mixed"


def _near_source_target(*, with_normals: bool = False):
    eps = np.array([1.0e-2, 3.0e-3, 1.0e-3])
    angles = np.array([0.0, 0.6, 1.4, 2.8])
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
