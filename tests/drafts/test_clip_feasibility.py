"""Tests for clip_feasibility.py -- stage 3 of docs/00-project-overview.md's build-up
order ("replace the privileged-state oracle with a feasibility judgment
from images"). Runs the real G1-in-ReplicaCAD-apartment env
(tidy_up_env_replicacad_humanoid.py), renders an actual camera frame, and
checks visual_object_exists() against the same privileged _exists state the
oracle uses -- privileged state is the label here, not the input, which is
exactly the comparison docs/01's "Success criteria" calls "headroom."

Two seed-related things this file used to work around, status as of D-021/
D-022:

- **Scene-layout-depends-on-seed (D-020's finding #4): fixed.** Used to be
  seed=0-only because other seeds loaded a different apartment entirely.
  Now pinned regardless of seed (D-021) -- confirmed via
  `test_scene_layout_reproducible_across_seeds` in
  test_tidy_up_env_replicacad_humanoid.py.
- **Render-producing-reset desync (D-022): NOT fixed, still why this file
  stays conservative.** Independently of seed, rendered frames for this env
  have been observed to desync from the actual scene after roughly the
  second render-producing reset within one process (object positions stay
  correct; the image doesn't). Root cause not found -- reproduces
  regardless of seed, of forcing `reconfigure=True`, and of
  `sapien.render.clear_cache()`; looks like a SAPIEN/ManiSkill CPU-renderer
  state leak. The env now warns past that point
  (`tidy_up_env_replicacad_humanoid.py`'s `_render_producing_reset_count`
  guard). This file keeps exactly two render-producing resets total (one
  per test below) and both have been visually spot-checked against saved
  frames, not just trusted from the CLIP score alone.

Slow: each case renders a frame and runs a real CLIP forward pass on CPU
(no CUDA on this dev machine).
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("open_clip")

import task_schema_draft  # noqa: E402, F401
from atr.feasibility.clip_feasibility import visual_object_exists  # noqa: E402

def _make_env(seed):
    env = gym.make(
        "TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind="chef_can_destroyed", onset_step_range=(2, 3),
    )
    env.reset(seed=seed)
    return env


class TestVisualFeasibilityMatchesOracle:
    """Same claim as D-014's H2 test, but the feasibility signal now comes
    from a rendered image and a pretrained model instead of privileged
    state -- the real thing this whole stage exists to test."""

    def test_both_objects_visually_present_before_intervention(self):
        env = _make_env(0)
        try:
            frame = env.render()[0].cpu().numpy()
            assert visual_object_exists(frame, "master_chef_can") is True
            assert visual_object_exists(frame, "potted_meat_can") is True
        finally:
            env.close()

    def test_destroyed_object_visually_absent_survivor_still_present(self):
        env = _make_env(0)
        try:
            for _ in range(4):
                env.step(env.action_space.sample() * 0)
            frame = env.render()[0].cpu().numpy()
            assert env.unwrapped._exists["master_chef_can"] is False  # oracle: destroyed
            assert env.unwrapped._exists["potted_meat_can"] is True  # oracle: untouched
            assert visual_object_exists(frame, "master_chef_can") is False
            assert visual_object_exists(frame, "potted_meat_can") is True
        finally:
            env.close()


class TestFailsLoudlyOnUncalibratedObject:
    def test_unknown_object_raises(self):
        import numpy as np

        with pytest.raises(ValueError, match="no calibrated visual config"):
            visual_object_exists(np.zeros((512, 512, 3), dtype=np.uint8), "bowl")
