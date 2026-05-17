"""Dirichlet solve on the annular region of a square-annulus ChunkGraph."""

from _chunkgraph_square_annulus_common import SAMPLE_TARGETS, make_square_annulus

from chunkie.kernels import kernel
from chunkie.system import (
    BoundaryEquation,
    BoundaryTrace,
    DensitySpace,
    IntegralSystem,
    JumpTerm,
    LayerPotential,
)


def main() -> None:
    graph = make_square_annulus()
    boundary = graph.boundary(1, side="interior")
    density = DensitySpace("sigma", boundary)
    double = LayerPotential("double", boundary, kernel("laplace", selector="d"), "sigma")
    trace = BoundaryTrace(double, boundary, "interior", jump=JumpTerm(-0.5, "sigma"))
    system = IntegralSystem(
        "chunkgraph_annular_dirichlet",
        graph,
        (density,),
        (BoundaryEquation("dirichlet", boundary, (trace,), boundary.pointinfo.positions[0]),),
        fields={"u": (double,)},
    )
    solution = system.solve()

    region_ids = graph.classify_points(SAMPLE_TARGETS)
    annular_targets = SAMPLE_TARGETS[:, region_ids == 1]
    values = solution.evaluate(annular_targets).values[0].real
    error = abs(values - annular_targets[0]).max()

    print(f"chunkgraph: {len(graph.edges)} edges, {boundary.point_count} annular boundary nodes")
    print(f"annular target count: {annular_targets.shape[1]}")
    print(f"annular-region Dirichlet max error: {error:.3e}")


if __name__ == "__main__":
    main()
