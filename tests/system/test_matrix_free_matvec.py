import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
    SystemOperator,
    matrix_free_matvec,
)


def test_matrix_free_matvec_matches_dense_off_boundary_trace_blocks():
    source = circle(quadrature_order=7, panel_count=10)
    target = circle(radius=0.6, center=(2.0, -0.3), quadrature_order=7, panel_count=8)
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", source)
    layer = LayerPotential("single", source, single, "sigma", coefficient=0.75)
    equation = BoundaryEquation(
        "trace",
        target,
        (BoundaryTrace(layer, target, "left"),),
        np.zeros(target.point_count),
    )
    system = IntegralSystem("matrix_free", (source, target), (density,), (equation,))
    vector = 1.0 + np.sin(source.pointinfo.flat_positions[0])

    dense = system.assemble().matvec(vector)
    direct = matrix_free_matvec(system, vector)
    operator_direct = SystemOperator(system).matvec(vector)

    np.testing.assert_allclose(direct, dense, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(operator_direct, dense, rtol=1.0e-12, atol=1.0e-12)


def test_matrix_free_matvec_matches_dense_jump_only_block():
    boundary = circle(quadrature_order=6, panel_count=8)
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", boundary)
    zero_layer = LayerPotential("zero", boundary, single, "sigma", coefficient=0.0)
    equation = BoundaryEquation(
        "identity",
        boundary,
        (BoundaryTrace(zero_layer, boundary, "left", jump=JumpTerm(-0.5, "sigma")),),
        np.zeros(boundary.point_count),
    )
    system = IntegralSystem("matrix_free_jump", boundary, (density,), (equation,))
    vector = np.cos(boundary.pointinfo.flat_positions[1])

    np.testing.assert_allclose(matrix_free_matvec(system, vector), system.assemble().matvec(vector))
