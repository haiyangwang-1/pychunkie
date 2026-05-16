"""Convenience Laplace boundary-value systems."""

from __future__ import annotations

from chunkie.kernels import kernel

from .density import DensitySpace
from .equation import BoundaryEquation, IntegralSystem
from .layer import LayerPotential
from .trace import BoundaryTrace, JumpTerm


def LaplaceExteriorDirichletSystem(
    geometry,
    boundary_data,
    *,
    name: str = "laplace_exterior_dirichlet",
    density_name: str = "sigma",
    field_name: str = "u",
) -> IntegralSystem:
    density = DensitySpace(density_name, geometry, component_count=1)
    layer = LayerPotential(
        name=f"{field_name}_double_layer",
        source=geometry,
        kernel=kernel("laplace", selector="d"),
        density=density_name,
    )
    trace = BoundaryTrace(
        layer=layer,
        target=geometry,
        side="exterior",
        jump=JumpTerm(coefficient=0.5, density=density_name),
    )
    equation = BoundaryEquation(name=f"{field_name}_dirichlet", target=geometry, terms=(trace,), rhs=boundary_data)
    return IntegralSystem(
        name=name,
        geometry=geometry,
        unknowns=(density,),
        equations=(equation,),
        fields={field_name: (layer,)},
    )
