"""Fast smoke tests for the object-level intervention spike.

See spikes/maniskill_humanoid_spike/README.md "Object-level intervention
findings" for what these confirm and why they're the actual gating test for
I-003/D-006 (not standing balance).
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import maniskill_humanoid_spike  # noqa: E402, F401  (registers ObjectInterventionSpike-v1)
from maniskill_humanoid_spike.device_utils import resolve_sim_backend  # noqa: E402


def _make_env(**kwargs):
    # Always CPU, never resolve_sim_backend() — object add/remove is
    # unsupported under GPU-batched sim by simulator design (see
    # object_intervention_spike.py's module docstring and the
    # test_gpu_sim_raises_clear_error test below), not a machine-specific
    # choice.
    return gym.make(
        "ObjectInterventionSpike-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend=resolve_sim_backend(prefer_gpu=False),
        **kwargs,
    )


class TestObjectRemovedIntervention:
    def test_entity_leaves_the_physics_scene(self):
        env = _make_env(intervention_kind="object_removed", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            sub_scene = env.unwrapped.scene.sub_scenes[0]
            target_entity = env.unwrapped._target._objs[0]
            assert target_entity in sub_scene.entities
            for _ in range(5):
                env.step(env.action_space.sample() * 0)
            assert target_entity not in sub_scene.entities
        finally:
            env.close()

    def test_actor_wrapper_goes_stale_after_removal(self):
        """Documents a real gap: don't trust the high-level Actor wrapper's
        state after remove_from_scene() — it keeps returning cached
        pre-removal values instead of erroring. Oracle/eval code must track
        existence itself (see InterventionRecord.exists_after)."""
        env = _make_env(intervention_kind="object_removed", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            target = env.unwrapped._target
            for _ in range(5):
                env.step(env.action_space.sample() * 0)
            # pose_before is captured at the moment of removal (after the
            # object has already settled under gravity for a few steps) —
            # comparing against that, not the t=0 spawn pose, isolates
            # "did removal change what the wrapper reports" from normal
            # pre-removal physics settling.
            record = env.unwrapped.last_intervention_record
            assert record is not None
            assert (target.pose.sp.p == record.pose_before).all()
        finally:
            env.close()

    def test_reproducible_given_seed(self):
        onset_steps = []
        for _ in range(2):
            env = _make_env(intervention_kind="object_removed")
            try:
                env.reset(seed=42)
                for _ in range(20):
                    env.step(env.action_space.sample() * 0)
                    record = env.unwrapped.last_intervention_record
                    if record is not None:
                        onset_steps.append(record.onset_step)
                        break
            finally:
                env.close()
        assert len(onset_steps) == 2
        assert onset_steps[0] == onset_steps[1]


class TestRouteBlockedIntervention:
    def test_new_entity_is_added_to_the_physics_scene(self):
        """The capability object-removal doesn't exercise: can new geometry
        be added to an already-built scene mid-episode, not just mutated?"""
        env = _make_env(intervention_kind="route_blocked", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            sub_scene = env.unwrapped.scene.sub_scenes[0]
            n_before = len(sub_scene.entities)
            for _ in range(5):
                env.step(env.action_space.sample() * 0)
            assert len(sub_scene.entities) == n_before + 1
            assert env.unwrapped._blocker is not None
        finally:
            env.close()


class TestGpuSimGuard:
    def test_gpu_sim_raises_clear_error_instead_of_silent_corruption(self):
        """Can't instantiate real GPU sim on this (CUDA-less) machine to
        test the true end-to-end path, but the guard itself is a plain
        attribute check — verify it fires correctly by faking the flag it
        reads, so the failure mode on a CUDA machine is a clear error, not a
        cryptic crash several calls deep into SAPIEN's GPU buffers."""
        env = _make_env(intervention_kind="object_removed", onset_step_range=(3, 4))
        try:
            env.reset(seed=0)
            env.unwrapped.scene.gpu_sim_enabled = True
            with pytest.raises(RuntimeError, match="requires CPU sim"):
                env.unwrapped._trigger_intervention()
        finally:
            env.unwrapped.scene.gpu_sim_enabled = False
            env.close()
