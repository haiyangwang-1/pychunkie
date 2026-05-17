"""Classify targets in a square-annulus ChunkGraph."""

from _chunkgraph_square_annulus_common import SAMPLE_TARGETS, make_square_annulus


def main() -> None:
    graph = make_square_annulus()
    region_ids = graph.classify_points(SAMPLE_TARGETS)
    left_regions = [edge.left_region for edge in graph.edges]
    right_regions = [edge.right_region for edge in graph.edges]

    print(f"chunkgraph: {len(graph.edges)} edges, {graph.point_count} nodes")
    print(f"regions at sample targets: {region_ids.tolist()}")
    print(f"edge left regions: {left_regions}")
    print(f"edge right regions: {right_regions}")


if __name__ == "__main__":
    main()
