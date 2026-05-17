import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    LayerPotential,
    fmm_matvec,
)


def test_fmm_matvec_matches_dense_for_off_boundary_laplace_traces():
    source = circle(quadrature_order=8, panel_count=16)
    target = circle(radius=0.7, center=(2.4, -0.2), quadrature_order=8, panel_count=12)
    single = kernel("laplace", selector="s")
    double = kernel("laplace", selector="d")
    density = DensitySpace("sigma", source)
    single_layer = LayerPotential("single", source, single, "sigma", coefficient=1.25)
    double_layer = LayerPotential("double", source, double, "sigma", coefficient=-0.4)
    system = IntegralSystem(
        name="off_boundary_fmm_matvec",
        geometry=(source, target),
        unknowns=(density,),
        equations=(
            BoundaryEquation(
                "trace",
                target,
                (
                    BoundaryTrace(single_layer, target, "left"),
                    BoundaryTrace(double_layer, target, "left"),
                ),
                np.zeros(target.point_count),
            ),
        ),
    )
    vector = np.cos(source.pointinfo.flat_positions[0])

    dense = system.assemble().matvec(vector)
    accelerated = fmm_matvec(system, vector, eps=1.0e-12)

    np.testing.assert_allclose(accelerated, dense, rtol=3.0e-11, atol=3.0e-12)
