import numpy as np

from chunkie.geometry import (
    chunker_from_polygon,
    circle,
    flagnear,
    flagnear_rectangle,
    flagnear_rectangle_grid,
    nearest_point,
)


def test_flagnear_uses_bernstein_rectangle_for_straight_panel():
    boundary = chunker_from_polygon(np.array([[0.0, 2.0], [0.0, 0.0]]), closed=False)
    rho = 1.8
    semimajor = 0.5 * (rho + 1.0 / rho)
    semiminor = 0.5 * (rho - 1.0 / rho)
    points = np.array(
        [
            [1.0, 1.0 + 0.99 * semimajor, 1.0 + 1.01 * semimajor, 1.0],
            [0.0, 0.0, 0.0, 1.01 * semiminor],
        ],
    )

    actual = flagnear(boundary, points, rho=rho)

    np.testing.assert_array_equal(actual, np.array([[True], [True], [False], [False]]))


def test_flagnear_rectangle_is_compatibility_alias_for_flagnear():
    boundary = circle(quadrature_order=8, panel_count=4)
    points = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

    np.testing.assert_array_equal(
        flagnear_rectangle(boundary, points, rho=1.8),
        flagnear(boundary, points, rho=1.8),
    )


def test_flagnear_rectangle_grid_matches_flat_meshgrid_order():
    boundary = circle(quadrature_order=8, panel_count=4)
    x = np.linspace(-1.5, 1.5, 21)
    y = np.linspace(-1.25, 1.25, 17)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.ravel(), yy.ravel()))

    direct = flagnear_rectangle(boundary, points, rho=1.8)
    grid = flagnear_rectangle_grid(boundary, x, y, rho=1.8)

    assert grid.shape == (y.size, x.size, boundary.panel_count)
    np.testing.assert_array_equal(grid, direct.reshape(y.size, x.size, boundary.panel_count))


def test_flagnear_rectangle_uses_oriented_bernstein_rectangle():
    boundary = chunker_from_polygon(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        closed=False,
        quadrature_order=8,
    )
    rho = 1.8
    semiminor = 0.5 * (rho - 1.0 / rho)
    points = np.array(
        [
            [1.0, 1.0, 2.0, 2.0 + 0.51 * semiminor],
            [0.0, 1.01 * semiminor, 0.9, 0.5],
        ],
    )

    actual = flagnear_rectangle(boundary, points, rho=rho)

    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                [True, False],
                [False, False],
                [False, True],
                [False, False],
            ],
        ),
    )


def test_nearest_point_projects_to_panel_reference_coordinate():
    boundary = chunker_from_polygon(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        closed=False,
        quadrature_order=12,
    )

    nearest = nearest_point(boundary, np.array([[1.25], [0.6]]))

    np.testing.assert_allclose(nearest.positions[:, 0], [1.25, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(nearest.derivatives[:, 0], [1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(nearest.second_derivatives[:, 0], [0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(nearest.distances[0], 0.6, atol=1.0e-12)
    np.testing.assert_allclose(nearest.reference_coordinates[0], 0.25, atol=1.0e-12)
    assert nearest.panel_ids[0] == 0
