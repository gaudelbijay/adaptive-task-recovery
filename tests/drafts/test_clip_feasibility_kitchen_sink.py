"""Tests for clip_feasibility.py's "kitchen_sink" scene variant (D-027) --
added specifically to answer the caveat that clip_feasibility.py was only
ever validated on one scene layout ("kitchen_cabinet"). Same underlying
claim as test_clip_feasibility.py: does a rendered-frame CLIP judgment
match oracle privileged state?

Deliberately uses subprocess-isolated capture (`capture_episode_subprocess.py`,
the same mechanism dinov2_probe.py uses), not in-process rendering like
test_clip_feasibility.py: that file already spends the entire per-process
render-producing-reset budget D-022 leaves safe (2, in one process) on
"kitchen_cabinet". Testing a second variant in the same process would push
past that -- each capture here gets its own fresh subprocess instead, so it
never competes for that budget.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("open_clip")

import atr.envs.capture_episode_subprocess as _capture_module  # noqa: E402
from atr.feasibility.clip_feasibility import visual_object_exists  # noqa: E402

# capture_episode_subprocess.py promoted to src/atr/envs/ (D-052).
_CAPTURE_SCRIPT = Path(_capture_module.__file__)


def _capture(seed: int, steps: int) -> dict:
    out_path = Path(f"/tmp/_vision_kitchen_sink_test_{seed}.npz")
    subprocess.run(
        [
            sys.executable, str(_CAPTURE_SCRIPT),
            "--seed", str(seed), "--steps", str(steps), "--out", str(out_path),
            "--scene-variant", "kitchen_sink",
        ],
        check=True, capture_output=True,
    )
    data = dict(np.load(out_path))
    out_path.unlink()
    return data


class TestKitchenSinkMatchesOracle:
    def test_both_objects_visually_present_before_intervention(self):
        data = _capture(seed=300, steps=0)
        assert visual_object_exists(data["frame"], "master_chef_can", "kitchen_sink") is True
        assert visual_object_exists(data["frame"], "potted_meat_can", "kitchen_sink") is True

    def test_destroyed_object_visually_absent_survivor_still_present(self):
        data = _capture(seed=300, steps=6)
        assert bool(data["exists_master_chef_can"]) is False  # oracle: destroyed
        assert bool(data["exists_potted_meat_can"]) is True  # oracle: untouched
        assert visual_object_exists(data["frame"], "master_chef_can", "kitchen_sink") is False
        assert visual_object_exists(data["frame"], "potted_meat_can", "kitchen_sink") is True


class TestFailsLoudlyOnUncalibratedVariant:
    def test_unknown_scene_variant_raises(self):
        with pytest.raises(ValueError, match="no calibrated visual config"):
            visual_object_exists(
                np.zeros((512, 512, 3), dtype=np.uint8), "master_chef_can", "living_room"
            )
