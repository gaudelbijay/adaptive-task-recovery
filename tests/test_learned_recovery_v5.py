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
