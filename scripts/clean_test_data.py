"""Remove generated test fixture data from tests/golden."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
PATTERNS = ("*.mat", "*.npz")


def generated_test_data() -> list[Path]:
    paths: list[Path] = []
    for pattern in PATTERNS:
        paths.extend(path for path in GOLDEN.glob(pattern) if path.is_file())
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list files without deleting them")
    args = parser.parse_args()

    paths = generated_test_data()
    if not paths:
        print("no generated test data files found")
        return

    action = "would remove" if args.dry_run else "removed"
    for path in paths:
        if not args.dry_run:
            path.unlink()
        print(f"{action} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
