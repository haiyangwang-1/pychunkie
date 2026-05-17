import numpy as np
from numpy.polynomial.legendre import leggauss

from chunkie.rcip import (
    build_block_prolongation,
    build_local_corner_geometry,
    build_prolongation,
    build_split_panel_prolongation,
    interpolate_density,
    schur_compress_block,
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


def test_split_panel_prolongation_preserves_polynomial_integrals():
    source, weights = leggauss(8)
    target, target_weights, interpolation, weighted_transfer = build_split_panel_prolongation(source, weights)
    values = source**5 - 0.2 * source**3 + 0.7
    expected_target = target**5 - 0.2 * target**3 + 0.7

    np.testing.assert_allclose(interpolation @ values, expected_target, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(target_weights * expected_target), weights @ values, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(weighted_transfer @ (weights * values)), weights @ values, atol=1.0e-14)


def test_block_prolongation_lifts_edges_and_components():
    source, weights = leggauss(4)
    _, _, interpolation, weighted_transfer = build_split_panel_prolongation(source, weights)

    block = build_block_prolongation(interpolation, edge_count=3, component_count=2)
    weighted_block = build_block_prolongation(weighted_transfer, edge_count=3, component_count=2)

    np.testing.assert_allclose(block, np.kron(np.eye(3), np.kron(interpolation, np.eye(2))))
    np.testing.assert_allclose(weighted_block, np.kron(np.eye(3), np.kron(weighted_transfer, np.eye(2))))


def test_schur_compress_block_matches_direct_block_formula():
    rng = np.random.default_rng(1234)
    fine_star_count = 4
    fine_eliminated_count = 2
    size = fine_star_count + fine_eliminated_count
    prolongation = rng.random((fine_star_count, fine_star_count // 2))
    weighted_prolongation = rng.random((fine_star_count, fine_star_count // 2))
    local_matrix = rng.random((size, size))
    local_matrix[
        np.ix_(range(fine_star_count, size), range(fine_star_count, size))
    ] += 3.0 * np.eye(fine_eliminated_count)
    seed_inverse = np.eye(fine_star_count)
    fine_star = np.arange(fine_star_count)
    fine_eliminated = np.arange(fine_star_count, size)
    coarse_star = np.arange(fine_star_count // 2)
    coarse_eliminated = np.arange(fine_star_count // 2, fine_star_count)

    actual = schur_compress_block(
        prolongation,
        weighted_prolongation,
        local_matrix,
        seed_inverse,
        fine_star,
        fine_eliminated,
        coarse_star,
        coarse_eliminated,
    )

    expected = seed_inverse.copy()
    eliminated_to_star = local_matrix[np.ix_(fine_eliminated, fine_star)] @ expected
    weighted_star = weighted_prolongation.T @ expected
    weighted_coupling = weighted_star @ local_matrix[np.ix_(fine_star, fine_eliminated)]
    eliminated_inverse = np.linalg.inv(
        local_matrix[np.ix_(fine_eliminated, fine_eliminated)]
        - eliminated_to_star @ local_matrix[np.ix_(fine_star, fine_eliminated)]
    )
    eliminated_to_prolonged = eliminated_inverse @ (eliminated_to_star @ prolongation)
    expected[np.ix_(coarse_star, coarse_star)] = weighted_star @ prolongation + (
        weighted_coupling @ eliminated_to_prolonged
    )
    expected[np.ix_(coarse_eliminated, coarse_eliminated)] = eliminated_inverse
    expected[np.ix_(coarse_eliminated, coarse_star)] = -eliminated_to_prolonged
    expected[np.ix_(coarse_star, coarse_eliminated)] = -weighted_coupling @ eliminated_inverse
    np.testing.assert_allclose(actual, expected)


def test_interpolate_density_handles_component_rows():
    source, _ = leggauss(4)
    target, _ = leggauss(6)
    prolongation = build_prolongation(source, target)
    values = np.vstack((source, source**2))

    interpolated = interpolate_density(values, prolongation)

    assert interpolated.shape == (2, target.size)
    np.testing.assert_allclose(interpolated[0], target, atol=1.0e-13)
    np.testing.assert_allclose(interpolated[1], target**2, atol=1.0e-13)
