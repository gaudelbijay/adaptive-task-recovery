"""Tests for frame_diff.py (D-063) -- the "simple pixel-difference change
detector plus rules" required baseline (docs/10-evaluation-and-benchmarks.md),
compared against the same oracle labels CLIP (test_clip_feasibility.py) and
DINOv2 (test_dinov2_probe.py) are checked against, on the same crop
regions, so the three are directly comparable.

Same D-022 discipline as test_clip_feasibility.py: this env's rendered
frames desync from the actual scene after roughly the second render-
producing reset in one process. Kept to ONE reset total (both the
"before" and "after" frame come from the same env instance/episode, two
`render()` calls, not two resets -- "same-instance reuse tolerated a bit
more" per the env's own warning) across the whole file, well inside the
verified-safe budget.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.feasibility.clip_feasibility import _OBJECT_VISUAL_CONFIG  # noqa: E402
from atr.feasibility.frame_diff import frame_difference_score, object_changed  # noqa: E402

# Empirically measured (not guessed), same seed=0/onset_step_range=(2,3)
# scenario test_clip_feasibility.py uses -- master_chef_can (destroyed):
# 1.052; potted_meat_can (survivor): 0.593. Confirmed identical across 5
# reruns (deterministic given this env's pinned scene layout, D-021), not
# just a lucky single sample -- though onset_step_range=(2,3) only ever
# samples onset_step=2, so this is one scenario measured repeatedly, not
# several independent ones. Threshold picked at the midpoint: real
# evidence, not a round guess, and deliberately not tuned closer to
# either measured value.
_THRESHOLD = 0.8


class TestFrameDifferenceMatchesOracle:
    """Same claim as test_clip_feasibility.py's TestVisualFeasibilityMatchesOracle
    and test_dinov2_probe.py's live-loop tests -- does a feasibility signal
    derived from pixels match privileged state? -- from the simplest
    possible detector this time: no learned parameters, no supervision."""

    def test_destroyed_object_scores_higher_than_survivor(self):
        env = gym.make(
            "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
            render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos",
            intervention_kind="chef_can_destroyed", onset_step_range=(2, 3),
        )
        try:
            env.reset(seed=0)
            before = env.render()[0].cpu().numpy()
            for _ in range(6):
                env.step(env.action_space.sample() * 0)
            after = env.render()[0].cpu().numpy()
            assert env.unwrapped._exists["master_chef_can"] is False  # oracle: destroyed
            assert env.unwrapped._exists["potted_meat_can"] is True  # oracle: untouched

            def _crop(frame, obj):
                y0, y1, x0, x1 = _OBJECT_VISUAL_CONFIG["kitchen_cabinet"][obj].crop
                return frame[y0:y1, x0:x1]

            scores = {
                obj: frame_difference_score(_crop(before, obj), _crop(after, obj))
                for obj in ("master_chef_can", "potted_meat_can")
            }

            # The real, measured margin: real but not clean -- destroyed
            # scores ~1.8x the survivor's, not CLIP/DINOv2's near-100%
            # separations. Worth stating plainly: this baseline is weaker,
            # which is exactly the point of having it.
            assert scores["master_chef_can"] > scores["potted_meat_can"]
            assert object_changed(_crop(before, "master_chef_can"), _crop(after, "master_chef_can"), _THRESHOLD) is True
            assert object_changed(_crop(before, "potted_meat_can"), _crop(after, "potted_meat_can"), _THRESHOLD) is False
        finally:
            env.close()


class TestFrameDifferenceScore:
    def test_identical_frames_score_zero(self):
        import numpy as np

        frame = np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8)
        assert frame_difference_score(frame, frame) == 0.0

    def test_raises_on_shape_mismatch(self):
        import numpy as np

        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.zeros((20, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape mismatch"):
            frame_difference_score(a, b)
