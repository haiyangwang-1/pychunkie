import numpy as np

from chunkie.geometry import circle
from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    Constraint,
    ConstraintTerm,
    DensitySpace,
    IntegralSystem,
    LayerPotential,
)
from chunkie.system.assembly import rhs_vector


def test_dense_assembly_appends_scalar_density_constraint_rows():
    boundary = circle(quadrature_order=6, panel_count=5)
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", boundary)
    zero_layer = LayerPotential("zero", boundary, single, "sigma", coefficient=0.0)
    constraint = Constraint(
        "charge",
        (ConstraintTerm("sigma", boundary.pointinfo.flat_weights),),
        value=2.5,
    )
    system = IntegralSystem(
        "constrained",
        boundary,
        (density,),
        (
            BoundaryEquation(
                "identity",
                boundary,
                (BoundaryTrace(zero_layer, boundary, "left"),),
                np.zeros(boundary.point_count),
            ),
        ),
        constraints=(constraint,),
    )

    matrix = system.assemble().to_dense()
    rhs = rhs_vector(system)

    assert matrix.shape == (boundary.point_count + 1, boundary.point_count)
    np.testing.assert_allclose(matrix[-1], boundary.pointinfo.flat_weights)
    np.testing.assert_allclose(rhs[-1], 2.5)
    assert system.assemble().diagnostics["constraints"] == ("charge",)
