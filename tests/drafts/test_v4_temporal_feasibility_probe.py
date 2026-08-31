"""Static checks for the held-out-mechanism V4 probe."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/probe_v4_temporal_feasibility.py"


def test_probe_keeps_reverse_ejection_out_of_training():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'training_kinds = ("nominal", "ejection", "permanent_block", "temporary_block")' in source
    assert '"heldout_intervention": "reverse_ejection"' in source
    assert '"ejection", "permanent_block", "temporary_block", "reverse_ejection",' in source


def test_probe_uses_multiple_temporal_horizons_and_goal_conditioning():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "HORIZONS = (1, 4, 8, 16, 32, 48)" in source
    tree = ast.parse(source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"visual_deltas", "pixel_deltas", "goal_condition", "collect"} <= functions
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"dinov2_axis"' in source
    assert '"pixel_axis"' in source


def test_temporary_block_is_a_negative_label():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'expected = kind in ("ejection", "reverse_ejection", "permanent_block")' in source
    assert "incorrectly authorized goal skipping" in source
