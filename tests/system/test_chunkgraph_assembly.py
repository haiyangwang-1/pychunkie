import numpy as np

from chunkie.geometry import ChunkGraph
from chunkie.kernels import kernel
from chunkie.quadrature import dense_panel_operator_matrix
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)


def _square_boundary_part(quadrature_order=5):
    vertices = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
    )
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    graph = ChunkGraph.from_vertices(vertices, edges, quadrature_order=quadrature_order)
    return graph.boundary(1, side="left")


def test_dense_assembly_accepts_chunkgraph_boundary_part_blocks():
    boundary = _square_boundary_part(quadrature_order=5)
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", boundary)
    layer = LayerPotential("sigma_single", boundary, single, "sigma", coefficient=1.5)
    system = IntegralSystem(
        name="chunkgraph_boundary_part_reference",
        geometry=boundary.graph,
        unknowns=(density,),
        equations=(
            BoundaryEquation(
                "dirichlet",
                boundary,
                (BoundaryTrace(layer, boundary, "left"),),
                np.zeros(boundary.point_count),
            ),
        ),
    )

    matrix = system.assemble().to_dense()
    expected = 1.5 * dense_panel_operator_matrix(boundary.pointinfo, boundary.pointinfo, single)

    np.testing.assert_allclose(matrix, expected)


def test_dense_solve_reconstructs_chunkgraph_boundary_part_density():
    boundary = _square_boundary_part(quadrature_order=4)
    single = kernel("laplace", selector="s")
    density = DensitySpace("sigma", boundary)
    zero_layer = LayerPotential("sigma_zero", boundary, single, "sigma", coefficient=0.0)
    rhs = boundary.pointinfo.positions[0]
    system = IntegralSystem(
        name="chunkgraph_boundary_part_identity",
        geometry=boundary.graph,
        unknowns=(density,),
        equations=(
            BoundaryEquation(
                "identity",
                boundary,
                (BoundaryTrace(zero_layer, boundary, "left", jump=JumpTerm(1.0, "sigma")),),
                rhs,
            ),
        ),
    )

    solution = system.solve()

    np.testing.assert_allclose(solution.densities["sigma"].values, rhs)
    assert np.linalg.norm(solution.residual) < 1.0e-12
