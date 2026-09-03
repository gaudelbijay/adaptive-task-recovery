"""Contract tests for the direction-deferred ejection environment.

The point of this environment is that ejection direction must not be readable
from a single early frame. These tests check that property structurally rather
than by training a model against it.
"""

import inspect

from atr.envs import learned_recovery_v5 as v5


def test_both_directions_share_one_ejector_per_cube():
    """The v4 shortcut was that each direction had its own actor."""
    source = inspect.getsource(v5.LearnedRecoveryDeferredDirectionEnv)
    assert "red_ejector" in source and "blue_ejector" in source
    # Exactly two ejectors are constructed, not four directional sweepers.
    assert source.count('actors.build_box') == 2


def test_direction_is_applied_only_after_the_delay():
    source = inspect.getsource(
        v5.LearnedRecoveryDeferredDirectionEnv._before_simulation_step
    )
    assert "diverged" in source
    # The lateral (direction-bearing) force is gated on divergence; the axial
    # approach force is not.
    assert "active & diverged" in source


def test_v4_ejection_forces_are_suppressed():
    source = inspect.getsource(
        v5.LearnedRecoveryDeferredDirectionEnv._before_simulation_step
    )
    assert "self.intervention_force = 0.0" in source
    assert "finally:" in source, "inherited force must be restored on error"


def test_delay_range_is_validated():
    import pytest
    for bad in ((0, 10), (10, 10), (12, 4)):
        with pytest.raises(ValueError):
            v5.LearnedRecoveryDeferredDirectionEnv(direction_delay_range=bad)


def test_environment_is_registered_separately_from_v4():
    source = inspect.getsource(v5)
    assert '@register_env("LearnedRecovery-v5"' in source
    assert "LearnedRecovery-v4" not in source.split("register_env")[1][:80]


def test_ejection_window_outlasts_the_longest_direction_delay():
    """The window must still be open when the direction fires.

    The inherited window is 12 steps while the delay is drawn from [10, 34), so
    the two overlapped only for delays of 10 or 11. In 92% of episodes the
    direction force never ran, which is why two successive revisions measured
    0.0 direction correctness and were rejected.
    """
    source = inspect.getsource(v5.LearnedRecoveryDeferredDirectionEnv.__init__)
    assert "self.direction_delay_range[1] + self.push_steps" in source
    assert "self.intervention_steps = required" in source
    # The widening must come after the base constructor, which is what sets
    # intervention_steps in the first place.
    assert source.index("super().__init__") < source.index("required =")


def test_direction_is_not_applied_along_the_axis_separating_the_cubes():
    """Ejecting along y fires the target cube into the protected one.

    The cubes sit at y = -0.12 and y = +0.12, so a y-directed impulse sends one
    into the other: collateral target loss was 0.5938 against a 0.02 ceiling,
    and unchanged across a tenfold force sweep because the cause is geometric.
    x is the axis v4 ejects along and the one `_unavailable` reads.
    """
    source = inspect.getsource(
        v5.LearnedRecoveryDeferredDirectionEnv._before_simulation_step
    )
    assert "impulse[:, 0]" in source, "direction must act on x"
    assert "impulse[:, 1]" not in source, "y separates the two cubes"


def test_smoke_gate_measures_before_the_episode_is_truncated():
    """The gate must not read cube poses after an auto-reset.

    Passing --steps as max_episode_steps and then running exactly that many
    steps truncates the episode and restores every cube to its reset pose, so
    displacement reads as identically zero. Both prior rejections of this
    environment rest on that defect.
    """
    from pathlib import Path
    source = Path("scripts/smoke_learned_recovery_v5.py").read_text()
    assert "max_episode_steps=args.max_episode_steps" in source
    assert "max_episode_steps=args.steps" not in source
