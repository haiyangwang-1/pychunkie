import numpy as np

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
