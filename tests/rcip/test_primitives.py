import numpy as np
from numpy.polynomial.legendre import leggauss

from chunkie.rcip import (
    build_local_corner_geometry,
    build_prolongation,
    interpolate_density,
)


def test_local_corner_geometry_uses_dyadic_ray_panels():
    local = build_local_corner_geometry(
        [0.0, 0.0],
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        quadrature_order=4,
        levels=3,
        base_length=1.0,
    )

    radii = np.linalg.norm(local.positions, axis=0)

    assert local.positions.shape == (2, 4, 6)
    np.testing.assert_allclose(local.panel_scales[:3], np.array([0.5, 0.25, 0.125]))
    np.testing.assert_allclose(np.min(radii[:, 0]), 0.5 * (1.0 + local.nodes[0]) / 2.0 + 0.5)
    np.testing.assert_allclose(np.sum(local.weights), 2.0 * (1.0 - 2.0**-3))


def test_prolongation_interpolates_polynomials_exactly():
    source, _ = leggauss(5)
    target, _ = leggauss(9)
    prolongation = build_prolongation(source, target)
    values = 1.0 - 2.0 * source + 3.0 * source**4

    interpolated = prolongation @ values

    np.testing.assert_allclose(interpolated, 1.0 - 2.0 * target + 3.0 * target**4, atol=1.0e-13)


def test_interpolate_density_handles_component_rows():
    source, _ = leggauss(4)
    target, _ = leggauss(6)
    prolongation = build_prolongation(source, target)
    values = np.vstack((source, source**2))

    interpolated = interpolate_density(values, prolongation)

    assert interpolated.shape == (2, target.size)
    np.testing.assert_allclose(interpolated[0], target, atol=1.0e-13)
    np.testing.assert_allclose(interpolated[1], target**2, atol=1.0e-13)
