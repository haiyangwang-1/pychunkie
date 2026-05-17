import numpy as np
import pytest

from chunkie.geometry import circle
from chunkie.system import Density


def test_density_vector_layout_is_component_major_over_panel_major_points():
    boundary = circle(quadrature_order=3, panel_count=2)
    values = np.arange(2 * 3 * 2).reshape(2, 3, 2)
    density = Density("sigma", boundary, values, component_count=2)

    vector = density.to_vector()
    restored = Density.from_vector("sigma", boundary, vector, component_count=2)

    expected = values.swapaxes(1, 2).reshape(-1)
    np.testing.assert_array_equal(vector, expected)
    np.testing.assert_array_equal(restored.values, values)


def test_scalar_density_vector_layout_round_trips_panel_major_values():
    boundary = circle(quadrature_order=3, panel_count=2)
    values = np.arange(6).reshape(3, 2)
    density = Density("sigma", boundary, values)

    vector = density.to_vector()
    restored = Density.from_vector("sigma", boundary, vector)

    np.testing.assert_array_equal(vector, np.array([0, 2, 4, 1, 3, 5]))
    np.testing.assert_array_equal(restored.values, values)


def test_density_layout_rejects_incompatible_shapes_and_sizes():
    boundary = circle(quadrature_order=3, panel_count=2)

    with pytest.raises(ValueError, match="component_count=1"):
        Density("bad", boundary, np.zeros((3, 2)), component_count=2)
    with pytest.raises(ValueError, match="component density values"):
        Density("bad", boundary, np.zeros((2, 2, 3)), component_count=2)
    with pytest.raises(ValueError, match="incompatible size"):
        Density.from_vector("bad", boundary, np.arange(5))
