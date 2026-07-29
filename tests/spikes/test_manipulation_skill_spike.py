"""Fast smoke tests for the manipulation-skill + RGB-D spike.

See spikes/maniskill_humanoid_spike/README.md "Manipulation skill + RGB-D
findings" for context: mplib-based canned solutions don't build on this
machine, so this uses ManiSkill3's built-in Cartesian IK controller instead.
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402, F401  (registers PickCube-v1 etc.)

from maniskill_humanoid_spike.device_utils import resolve_sim_backend  # noqa: E402
from maniskill_humanoid_spike.manipulation_skill_spike import (  # noqa: E402
    rgbd_obs_shapes,
    scripted_pick_and_lift,
)


class TestRgbdObservations:
    def test_rgb_and_depth_are_well_formed(self):
        env = gym.make(
            "PickCube-v1", num_envs=1, obs_mode="rgbd", render_mode=None,
            sim_backend=resolve_sim_backend(),
        )
        try:
            result = rgbd_obs_shapes(env)
            assert result["rgb_shape"][-1] == 3
            assert result["rgb_dtype"] == "torch.uint8"
            assert result["rgb_range"][1] > result["rgb_range"][0]  # not a blank/constant image
            assert result["depth_shape"][-1] == 1
            assert result["depth_range"][1] > 0  # some non-zero depth was captured
        finally:
            env.close()


class TestScriptedPickAndLift:
    def test_pick_and_lift_succeeds(self):
        """Not a guarantee for every seed (this is proportional control, not
        motion planning), but should succeed reliably from a fixed spawn
        region — 5/5 succeeded in manual runs across seeds 0-4."""
        env = gym.make(
            "PickCube-v1", num_envs=1, obs_mode="state", render_mode=None,
            sim_backend=resolve_sim_backend(), control_mode="pd_ee_delta_pos",
        )
        try:
            assert scripted_pick_and_lift(env, seed=0)
        finally:
            env.close()
