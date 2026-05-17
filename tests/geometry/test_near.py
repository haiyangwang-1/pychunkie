import numpy as np

from chunkie.geometry import (
    chunker_from_polygon,
    circle,
    flagnear,
    flagnear_rectangle,
    flagnear_rectangle_grid,
)


def test_flagnear_matches_bruteforce_panel_node_distance():
    boundary = circle(quadrature_order=8, panel_count=4)
    points = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])
    near_factor = 0.75

    actual = flagnear(boundary, points, near_factor=near_factor)
    expected = np.zeros_like(actual)
    for panel_id in range(boundary.panel_count):
        distances = np.linalg.norm(
            boundary.positions[:, :, panel_id][:, None, :] - points[:, :, None],
            axis=0,
        )
        expected[:, panel_id] = np.min(distances, axis=1) <= near_factor * boundary.panel_lengths[panel_id]

    np.testing.assert_array_equal(actual, expected)


def test_flagnear_rectangle_grid_matches_flat_meshgrid_order():
    boundary = circle(quadrature_order=8, panel_count=4)
    x = np.linspace(-1.5, 1.5, 21)
    y = np.linspace(-1.25, 1.25, 17)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack((xx.ravel(), yy.ravel()))

    direct = flagnear_rectangle(boundary, points, rho=0.25)
    grid = flagnear_rectangle_grid(boundary, x, y, rho=0.25)

    assert grid.shape == (y.size, x.size, boundary.panel_count)
    np.testing.assert_array_equal(grid, direct.reshape(y.size, x.size, boundary.panel_count))


def test_flagnear_rectangle_uses_per_panel_padding():
    boundary = chunker_from_polygon(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        closed=False,
        quadrature_order=8,
    )
    points = np.array([[1.0, 1.0, 2.25, 2.25], [0.1, 1.6, 0.5, 1.6]])
    rho = 0.2

    actual = flagnear_rectangle(boundary, points, rho=rho)
    expected = np.zeros_like(actual)
    for panel_id in range(boundary.panel_count):
        padding = rho * boundary.panel_lengths[panel_id]
        lower = np.min(boundary.positions[:, :, panel_id], axis=1) - padding
        upper = np.max(boundary.positions[:, :, panel_id], axis=1) + padding
        expected[:, panel_id] = np.all((points.T >= lower[None, :]) & (points.T <= upper[None, :]), axis=1)

    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual)
    assert not np.all(actual)
