import numpy as np

from chunkie.system.backends.flam import factor_system


def test_flam_factor_apply_and_solve_match_dense_matrix():
    point_count = 24
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.vstack((np.cos(theta), np.sin(theta)))
    distances = np.linalg.norm(points[:, :, None] - points[:, None, :], axis=0)
    matrix = np.eye(point_count) + 0.1 / (1.0 + distances**2)
    vector = np.sin(theta)

    factor = factor_system(matrix, points, occupancy=16, tolerance=1.0e-12)
    applied = factor.apply(vector)
    solved = factor.solve(matrix @ vector)

    np.testing.assert_allclose(applied, matrix @ vector, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(solved, vector, rtol=1.0e-11, atol=1.0e-11)


def test_flam_factor_handles_complex_multiple_rhs_and_logdet():
    point_count = 16
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.vstack((np.cos(theta), np.sin(theta)))
    distances = np.linalg.norm(points[:, :, None] - points[:, None, :], axis=0)
    matrix = (1.0 + 0.2j) * np.eye(point_count) + 0.05 / (1.0 + distances**2)
    rhs = np.column_stack((np.sin(theta), np.cos(theta)))

    factor = factor_system(matrix, points, occupancy=8, tolerance=1.0e-12)
    applied = factor.apply(rhs)
    solved = factor.solve(matrix @ rhs)
    sign, logabs = np.linalg.slogdet(matrix)

    np.testing.assert_allclose(applied, matrix @ rhs, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(solved, rhs, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(factor.logdet(), np.log(sign) + logabs, rtol=1.0e-11, atol=1.0e-11)
