"""Classify targets in a square-annulus chunkgraph.

Run from the repository root:

    uv run python examples/chunkgraph_region_classification.py
"""

from __future__ import annotations

from chunkie import chunkgraphinregion, find_edge_regions

from _chunkgraph_square_annulus_common import SAMPLE_TARGETS, make_square_annulus


def main() -> None:
    cg = make_square_annulus()
    region_ids = chunkgraphinregion(cg, SAMPLE_TARGETS)
    edge_regions = find_edge_regions(cg)

    print(f"chunkgraph: {len(cg.echnks)} edges, {cg.npt} nodes")
    print(f"regions at sample targets: {region_ids.tolist()}")
    print(f"edge regions on positive side: {edge_regions[0].tolist()}")
    print(f"edge regions on negative side: {edge_regions[1].tolist()}")


if __name__ == "__main__":
    main()
