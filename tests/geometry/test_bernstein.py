import numpy as np
import pytest

from chunkie.geometry import (
    bernstein_ellipse,
    bernstein_panel_image,
    bernstein_radius,
    chunker_from_polygon,
)


def test_bernstein_ellipse_matches_reference_conformal_map():
    points = bernstein_ellipse(8, 2.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    expected = 0.5 * (2.0 * np.exp(1j * theta) + 0.5 * np.exp(-1j * theta))

    np.testing.assert_allclose(points, expected)
    with pytest.raises(ValueError, match="rho"):
        bernstein_ellipse(8, 1.0)
    with pytest.raises(ValueError, match="point_count"):
        bernstein_ellipse(2, 2.0)


def test_bernstein_panel_image_maps_straight_panel_affinely():
    boundary = chunker_from_polygon(
        np.array([[0.0, 2.0], [0.0, 0.0]]), closed=False, quadrature_order=8
    )

    image = bernstein_panel_image(boundary, 0, rho=2.0, point_count=32)

    np.testing.assert_allclose(image.positions[0], 1.0 + image.references, atol=1.0e-13)
    np.testing.assert_allclose(image.positions[1], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(image.center, [1.0 + 0.0j, 0.0 + 0.0j], atol=1.0e-14)
    np.testing.assert_allclose(image.radius, np.max(np.abs(image.references)), atol=1.0e-14)


def test_bernstein_radius_accepts_complex_point_clouds():
    points = np.array(
        [
            [1.0 + 1.0j, -1.0 - 1.0j],
            [2.0 - 0.5j, -2.0 + 0.5j],
        ],
    )

    expected = np.sqrt(abs(1.0 + 1.0j) ** 2 + abs(2.0 - 0.5j) ** 2)

    np.testing.assert_allclose(bernstein_radius(points), expected)
