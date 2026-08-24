#!/usr/bin/env python3
"""Run a deterministic, disjoint shard of test files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def select_test_files(root: str | Path, shard_index: int, shard_count: int) -> list[Path]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    root = Path(root)
    files = sorted(root.rglob("test_*.py"))
    return files[shard_index::shard_count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="tests")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args, pytest_args = parser.parse_known_args()

    selected = select_test_files(args.root, args.shard_index, args.shard_count)
    if not selected:
        raise SystemExit("selected shard contains no test files")
    command = [sys.executable, "-m", "pytest", *(str(path) for path in selected), *pytest_args]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
