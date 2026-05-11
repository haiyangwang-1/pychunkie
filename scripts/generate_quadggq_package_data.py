"""Convert MATLAB GGQ quadrature tables into package-owned NumPy assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


CELL_TABLE_RE = re.compile(r"(xs0|ws0)\{\s*(\d+)\s*\}\s*=\s*\[(.*?)\];", flags=re.S)
SKIPPED_TABLES = {
    # Present upstream, but incomplete and not reachable from MATLAB getlogquad.m.
    "ggqself_nnode040_npoly080",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matlab_quadggq_dir",
        type=Path,
        help="Path to chunkIE's chunkie/+chnk/+quadggq directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/chunkie/data/quadggq"),
        help="Output directory for .npz package assets.",
    )
    parser.add_argument(
        "--source-commit",
        default="unknown",
        help="Source chunkIE commit recorded into each .npz file.",
    )
    args = parser.parse_args()

    source_dir = args.matlab_quadggq_dir.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(source_dir.glob("ggqnear*.m")):
        if path.stem == "ggqnear":
            continue
        _write_near_table(path, out_dir / f"{path.stem}.npz", args.source_commit)
        count += 1

    for pattern in ("ggqself_*.m", "hsupp_*.m", "hqsupp_*.m"):
        for path in sorted(source_dir.glob(pattern)):
            if path.stem in SKIPPED_TABLES:
                continue
            _write_cell_table(path, out_dir / f"{path.stem}.npz", args.source_commit)
            count += 1

    np.savez_compressed(
        out_dir / "metadata.npz",
        source_commit=np.array(args.source_commit),
        table_count=np.array(count, dtype=np.int64),
    )
    print(f"wrote {count} quadrature tables to {out_dir}")


def _write_near_table(path: Path, out_path: Path, source_commit: str) -> None:
    text = path.read_text(encoding="utf-8")
    np.savez_compressed(
        out_path,
        x=_parse_vector_assignment(text, "x"),
        w=_parse_vector_assignment(text, "w"),
        source_file=np.array(path.name),
        source_commit=np.array(source_commit),
    )


def _write_cell_table(path: Path, out_path: Path, source_commit: str) -> None:
    text = path.read_text(encoding="utf-8")
    xs = _parse_cells(text, "xs0")
    ws = _parse_cells(text, "ws0")
    if len(xs) != len(ws):
        raise ValueError(f"malformed MATLAB GGQ table: {path}")

    arrays: dict[str, np.ndarray] = {
        "count": np.array(len(xs), dtype=np.int64),
        "source_file": np.array(path.name),
        "source_commit": np.array(source_commit),
    }
    for i, (x, w) in enumerate(zip(xs, ws)):
        arrays[f"xs0_{i:03d}"] = x
        arrays[f"ws0_{i:03d}"] = w
    np.savez_compressed(out_path, **arrays)


def _parse_vector_assignment(text: str, name: str) -> np.ndarray:
    match = re.search(rf"\b{name}\s*=\s*\[(.*?)\];", text, flags=re.S)
    if match is None:
        raise ValueError(f"no {name} vector found")
    return _parse_numbers(match.group(1))


def _parse_cells(text: str, name: str) -> list[np.ndarray]:
    matches = [(int(idx), block) for var, idx, block in CELL_TABLE_RE.findall(text) if var == name]
    if not matches:
        raise ValueError(f"no {name} cells found")
    out: list[np.ndarray] = [np.array([], dtype=float)] * max(idx for idx, _ in matches)
    for idx, block in matches:
        out[idx - 1] = _parse_numbers(block)
    return out


def _parse_numbers(block: str) -> np.ndarray:
    return np.fromstring(block.replace("D", "E").replace("d", "e"), sep=" ")


if __name__ == "__main__":
    main()
