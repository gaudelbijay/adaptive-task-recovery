"""Static checks for the V4 state input adapter."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/build_v4_padded_state_checkpoint.py").read_text()


def test_adapter_preserves_old_columns_and_zero_initializes_only_appended_columns():
    assert "padded[:, :old.shape[1]] = old" in SOURCE
    assert "old.new_zeros" in SOURCE
    assert 'adapted.pop("optimizer", None)' in SOURCE
