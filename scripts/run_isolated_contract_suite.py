#!/usr/bin/env python3
"""Run every contract test file in a fresh process and record exact coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def test_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("test_*.py"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def collect_nodeids(path: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"test-item collection failed for {path}: {result.stderr.strip()}"
        )
    nodeids = [
        f"{path}::{line.strip().split('::', 1)[1]}"
        for line in result.stdout.splitlines()
        if ".py::" in line.strip()
    ]
    if not nodeids:
        raise ValueError(f"test-item isolation collected no tests: {path}")
    return nodeids


def run_suite(
    root: str | Path,
    manifest_path: str | Path,
    item_isolate_files: list[str | Path] | tuple[str | Path, ...] = (),
) -> dict:
    files = test_files(root)
    if not files:
        raise ValueError("isolated contract suite selected no test files")
    started = time.time()
    records = []
    manifest_path = Path(manifest_path)
    junit_dir = manifest_path.with_name(f"{manifest_path.stem}_junit")
    junit_dir.mkdir(parents=True, exist_ok=True)
    item_isolate = {str(Path(path)) for path in item_isolate_files}
    missing = item_isolate - {str(path) for path in files}
    if missing:
        raise ValueError(f"item-isolated files are outside the suite: {sorted(missing)}")
    for index, path in enumerate(files):
        print(f"ISOLATED_FILE {index + 1}/{len(files)} {path}", flush=True)
        file_started = time.time()
        targets = collect_nodeids(path) if str(path) in item_isolate else [str(path)]
        results = []
        junit_paths = []
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        for item_index, target in enumerate(targets):
            if len(targets) > 1:
                print(
                    f"ISOLATED_ITEM {item_index + 1}/{len(targets)} {target}",
                    flush=True,
                )
            junit_path = junit_dir / f"{index:03d}_{item_index:03d}.xml"
            result = subprocess.run([
                sys.executable, "-m", "pytest", "-q", target,
                f"--junitxml={junit_path}",
            ])
            results.append(result)
            junit_paths.append(junit_path)
            item_counts = junit_counts(junit_path) if junit_path.exists() else {
                "tests": 0, "failures": 0, "errors": 1, "skipped": 0,
            }
            for key, value in item_counts.items():
                counts[key] += value
        records.append({
            "index": index,
            "path": str(path),
            "sha256": sha256(path),
            "returncode": max(int(result.returncode) for result in results),
            "elapsed_seconds": time.time() - file_started,
            "isolation": "item" if len(targets) > 1 else "file",
            "items_run": len(targets),
            "junit_paths": [str(path) for path in junit_paths],
            "junit_sha256": [
                sha256(path) if path.exists() else None for path in junit_paths
            ],
            **counts,
        })
        atomic_json(manifest_path, {
            "schema_version": 1,
            "protocol": "file-isolated full contract suite",
            "root": str(root),
            "complete": False,
            "files_total": len(files),
            "files_run": len(records),
            "records": records,
        })
    failures = [record for record in records if record["returncode"] != 0]
    totals = {
        f"{key}_total": sum(record[key] for record in records)
        for key in ("tests", "failures", "errors", "skipped")
    }
    payload = {
        "schema_version": 1,
        "protocol": "file-isolated full contract suite",
        "root": str(root),
        "complete": True,
        "passed": not failures,
        "files_total": len(files),
        "files_run": len(records),
        "unique_files": len({record["path"] for record in records}),
        "failed_files": [record["path"] for record in failures],
        "elapsed_seconds": time.time() - started,
        **totals,
        "records": records,
    }
    atomic_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="tests/drafts")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--item-isolate-file", action="append", default=[])
    args = parser.parse_args()
    payload = run_suite(args.root, args.manifest, args.item_isolate_file)
    print(json.dumps({
        key: payload[key] for key in (
            "protocol", "complete", "passed", "files_total", "files_run",
            "unique_files", "failed_files", "elapsed_seconds",
            "tests_total", "failures_total", "errors_total", "skipped_total",
        )
    }, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
