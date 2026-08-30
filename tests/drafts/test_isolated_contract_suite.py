import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from run_isolated_contract_suite import run_suite, test_files as select_files  # noqa: E402


def test_file_selection_is_sorted_unique_and_complete(tmp_path):
    (tmp_path / "z").mkdir()
    (tmp_path / "test_b.py").write_text("def test_b(): pass\n")
    (tmp_path / "z" / "test_a.py").write_text("def test_a(): pass\n")
    (tmp_path / "not_a_test.py").write_text("def test_hidden(): pass\n")
    selected = select_files(tmp_path)
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == 2


def test_runner_records_every_file_and_fails_closed(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_pass.py").write_text("def test_ok(): assert True\n")
    (tests / "test_fail.py").write_text(
        "def test_bad(): assert False\n\ndef test_other(): assert True\n"
    )
    manifest = tmp_path / "manifest.json"
    payload = run_suite(tests, manifest, [tests / "test_fail.py"])
    stored = json.loads(manifest.read_text())
    assert payload == stored
    assert stored["complete"] is True
    assert stored["passed"] is False
    assert stored["files_total"] == stored["files_run"] == stored["unique_files"] == 2
    assert stored["failed_files"] == [str(tests / "test_fail.py")]
    assert stored["tests_total"] == 3
    assert stored["failures_total"] == 1
    assert stored["errors_total"] == stored["skipped_total"] == 0
    assert all(
        Path(path).exists()
        for record in stored["records"] for path in record["junit_paths"]
    )
    assert all(
        digest
        for record in stored["records"] for digest in record["junit_sha256"]
    )
    isolated = next(record for record in stored["records"] if record["path"].endswith("test_fail.py"))
    assert isolated["isolation"] == "item"
    assert isolated["items_run"] == 2
