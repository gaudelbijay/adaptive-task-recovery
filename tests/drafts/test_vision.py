"""Tests for vision.py -- stage 3 of docs/00-project-overview.md's build-up
order ("replace the privileged-state oracle with a feasibility judgment
from images"). Runs the real G1-in-ReplicaCAD-apartment env
(tidy_up_env_replicacad_humanoid.py), renders an actual camera frame, and
checks visual_object_exists() against the same privileged _exists state the
oracle uses -- privileged state is the label here, not the input, which is
exactly the comparison docs/01's "Success criteria" calls "headroom."

Seed=0 only, deliberately: rendering real frames for this test is what
surfaced a real, previously-unknown bug in tidy_up_env_replicacad_humanoid.py
-- G1's hardcoded base pose and camera are calibrated against seed=0's
apartment layout specifically. `ReplicaCADSetTableTrain` loads a genuinely
different room per seed (confirmed by rendering seed=2: G1 ends up next to
a couch and a bicycle, nowhere near the cans), so every existing test for
that env (D-018, all seed=0) was accidentally validating one scene layout,
not the general case. That's a real finding from this stage, not a
shortcut -- see D-020 in ai-notes/decisions.md. Fixing the general case is
a separate, later problem; this test scopes itself to the one layout
that's actually known to place G1 sensibly.

Slow: each case renders a frame and runs a real CLIP forward pass on CPU
(no CUDA on this dev machine).
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")
pytest.importorskip("open_clip")

import task_schema_draft  # noqa: E402, F401
from task_schema_draft.vision import visual_object_exists  # noqa: E402

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
