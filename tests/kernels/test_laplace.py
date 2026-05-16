import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import LaplaceBasis, kernel


def test_laplace_single_layer_returns_component_first_tensor():
    source = np.array([[0.0, 1.0], [0.0, 0.0]])
    target = np.array([[0.0], [1.0]])
    laplace_s = kernel("laplace", selector="s")

    values = laplace_s(source, target)

    assert values.shape == (1, 1, 1, 2)
    expected = -np.log(np.array([[1.0, 2.0]])) / (4.0 * np.pi)
    np.testing.assert_allclose(values[0, 0], expected)


def test_laplace_singularity_basis_matches_single_layer_kernel():
    source = np.array([[0.0], [0.0]])
    target = np.array([[2.0], [0.0]])
    laplace_s = kernel("laplace", selector="s")

    expansion = laplace_s.singularity.expansion.evaluate(source, target)
    values = laplace_s(source, target)

    np.testing.assert_allclose(expansion, values)
    assert LaplaceBasis(()).legacy_strength == "log"
    assert laplace_s.singularity.legacy_strength == "log"


def test_laplace_double_layer_requires_source_normals_and_has_pv_metadata():
    boundary = circle(quadrature_order=4, panel_count=4)
    laplace_d = kernel("laplace", selector="d")

    values = laplace_d(boundary.pointinfo, np.array([[2.0], [0.0]]))

    assert values.shape == (1, 1, 1, boundary.point_count)
    assert laplace_d.singularity.legacy_strength == "pv"
