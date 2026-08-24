"""Focused tests for deterministic CI test-file sharding."""

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_test_shard.py"
_SPEC = importlib.util.spec_from_file_location("run_test_shard", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
select_test_files = _MODULE.select_test_files


def test_shards_form_an_exact_deterministic_partition(tmp_path):
    for name in ("test_delta.py", "test_alpha.py", "test_charlie.py", "test_bravo.py"):
        (tmp_path / name).write_text("")

    shards = [select_test_files(tmp_path, index, 3) for index in range(3)]
    flattened = [path for shard in shards for path in shard]

    assert sorted(flattened) == sorted(tmp_path.glob("test_*.py"))
    assert len(flattened) == len(set(flattened))
    assert shards == [select_test_files(tmp_path, index, 3) for index in range(3)]


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    ((0, 0), (-1, 2), (2, 2)),
)
def test_invalid_shard_coordinates_fail(shard_index, shard_count, tmp_path):
    with pytest.raises(ValueError):
        select_test_files(tmp_path, shard_index, shard_count)
