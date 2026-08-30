#!/usr/bin/env python3
"""Compose an immutable failed full-suite run with one corrected test rerun.

The original file-isolated suite is retained as evidence.  This verifier only
accepts a single explicitly named source correction, proves that every other
test file is byte-identical to the original run and passed there, then executes
the corrected file in a fresh process with retained JUnit output.
"""

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


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def junit_counts(path: str | Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def verify_original(
    manifest: dict,
    corrected_path: str,
    expected_failed_source_sha256: str,
    expected_failed_junit_sha256: str,
) -> tuple[dict[str, dict], dict]:
    if not manifest.get("complete"):
        raise ValueError("original suite did not reach a terminal manifest")
    if manifest.get("passed") is not False:
        raise ValueError("repair protocol requires the preserved failed suite")
    records = manifest.get("records", [])
    if int(manifest.get("files_total", -1)) != len(records):
        raise ValueError("original suite file count is inconsistent")
    by_path = {record["path"]: record for record in records}
    if len(by_path) != len(records):
        raise ValueError("original suite contains duplicate file records")
    if corrected_path not in by_path:
        raise ValueError("corrected file is absent from the original suite")
    failed = by_path[corrected_path]
    if failed.get("sha256") != expected_failed_source_sha256:
        raise ValueError("original corrected-file source hash changed")
    junit_hashes = failed.get("junit_sha256", [])
    if junit_hashes != [expected_failed_junit_sha256]:
        raise ValueError("original corrected-file JUnit hash changed")
    if (
        int(failed.get("returncode", 0)) == 0
        or int(failed.get("failures", 0)) + int(failed.get("errors", 0)) == 0
    ):
        raise ValueError("original corrected-file record did not fail")
    for path, record in by_path.items():
        if path == corrected_path:
            continue
        if int(record.get("returncode", 1)) != 0:
            raise ValueError(f"unrepaired original failure: {path}")
        if any(int(record.get(key, 0)) for key in ("failures", "errors", "skipped")):
            raise ValueError(f"unrepaired non-pass outcome: {path}")
    return by_path, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-manifest", required=True)
    parser.add_argument("--expected-original-manifest-sha256", required=True)
    parser.add_argument("--corrected-file", required=True)
    parser.add_argument("--expected-failed-source-sha256", required=True)
    parser.add_argument("--expected-failed-junit-sha256", required=True)
    parser.add_argument("--expected-corrected-tests", required=True, type=int)
    parser.add_argument("--junit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original_path = Path(args.original_manifest)
    original_hash = sha256(original_path)
    if original_hash != args.expected_original_manifest_sha256:
        raise ValueError("original suite manifest hash changed")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    corrected_path = str(Path(args.corrected_file))
    by_path, failed = verify_original(
        original,
        corrected_path,
        args.expected_failed_source_sha256,
        args.expected_failed_junit_sha256,
    )

    current_paths = sorted(str(path) for path in Path(original["root"]).rglob("test_*.py"))
    if current_paths != sorted(by_path):
        raise ValueError("current test-file inventory differs from the original suite")
    for path in current_paths:
        if path != corrected_path and sha256(path) != by_path[path]["sha256"]:
            raise ValueError(f"uncovered source change outside corrected file: {path}")
    corrected_hash = sha256(corrected_path)
    if corrected_hash == failed["sha256"]:
        raise ValueError("corrected file is byte-identical to the failed source")

    junit_path = Path(args.junit)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-q", corrected_path,
        f"--junitxml={junit_path}",
    ])
    counts = junit_counts(junit_path) if junit_path.exists() else {
        "tests": 0, "failures": 0, "errors": 1, "skipped": 0,
    }
    passed = (
        result.returncode == 0
        and counts == {
            "tests": args.expected_corrected_tests,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
    )
    payload = {
        "schema_version": 1,
        "protocol": "immutable full-suite plus single-file correction rerun",
        "complete": True,
        "passed": passed,
        "files_total": len(current_paths),
        "files_reused": len(current_paths) - 1,
        "files_rerun": 1,
        "tests_reused": sum(
            int(record.get("tests", 0))
            for path, record in by_path.items()
            if path != corrected_path
        ),
        "corrected_file": corrected_path,
        "corrected_file_sha256": corrected_hash,
        "corrected_junit": str(junit_path),
        "corrected_junit_sha256": sha256(junit_path) if junit_path.exists() else None,
        "corrected_counts": counts,
        "corrected_returncode": int(result.returncode),
        "elapsed_seconds": time.time() - started,
        "original_manifest": str(original_path),
        "original_manifest_sha256": original_hash,
        "original_failed_source_sha256": failed["sha256"],
        "original_failed_junit_sha256": failed["junit_sha256"][0],
        "claim_boundary": (
            "Composite regression evidence only: all unchanged files reuse the "
            "byte-identified original isolated results; only the corrected file "
            "is rerun. This is not policy-performance evidence."
        ),
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
