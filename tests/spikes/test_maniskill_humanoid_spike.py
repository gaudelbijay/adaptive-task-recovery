"""Fast smoke tests for the ManiSkill3 humanoid simulator-selection spike.

See spikes/maniskill_humanoid_spike/README.md for what this spike is and
isn't testing. Requires the ManiSkill3 stack (mani_skill, sapien, torch) —
skips cleanly if that stack isn't importable.
"""

import numpy as np
import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import maniskill_humanoid_spike  # noqa: E402, F401  (registers HumanoidStandSpike-*-v1)
from maniskill_humanoid_spike.device_utils import resolve_sim_backend  # noqa: E402
from maniskill_humanoid_spike.scripted_intervention import ScriptedPushIntervention  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "HumanoidStandSpike-G1-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend=resolve_sim_backend(),
        **kwargs,
    )


class TestScriptedPushIntervention:
    def test_same_seed_is_deterministic(self):
        a = ScriptedPushIntervention(np.random.default_rng(42))
        b = ScriptedPushIntervention(np.random.default_rng(42))
        assert a.onset_step == b.onset_step
        assert a.severity == b.severity
        assert np.allclose(a.force, b.force)

    def test_different_seed_differs(self):
        a = ScriptedPushIntervention(np.random.default_rng(1))
        b = ScriptedPushIntervention(np.random.default_rng(2))
        assert (a.onset_step, a.severity) != (b.onset_step, b.severity)

    def test_fires_exactly_once_at_onset_step(self):
        intervention = ScriptedPushIntervention(np.random.default_rng(0), onset_step_range=(10, 11))
        assert intervention.onset_step == 10
        for step in range(10):
            assert intervention.maybe_trigger(step) is None
        event = intervention.maybe_trigger(10)
        assert event is not None
        assert event.onset_step == 10
        assert intervention.maybe_trigger(11) is None  # only fires once per episode

    def test_severity_scales_force_magnitude(self):
        lo = ScriptedPushIntervention(
            np.random.default_rng(0), force_magnitude_range=(100.0, 300.0), severity=0.0
        )
        hi = ScriptedPushIntervention(
            np.random.default_rng(0), force_magnitude_range=(100.0, 300.0), severity=1.0
        )
        assert np.linalg.norm(lo.force) == pytest.approx(100.0, abs=1e-3)
        assert np.linalg.norm(hi.force) == pytest.approx(300.0, abs=1e-3)


class TestHumanoidStandSpikeEnv:
    def test_registered(self):
        assert "HumanoidStandSpike-G1-v1" in gym.envs.registry
        assert "HumanoidStandSpike-H1-v1" in gym.envs.registry

    def test_reset_and_step(self):
        env = _make_env()
        try:
            obs, info = env.reset(seed=0)
            assert obs.shape[0] == 1
            action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            assert "is_standing" in info
        finally:
            env.close()

    def test_push_reproducible_given_seed(self):
        """Same seed -> same scripted push (onset step, severity), across two
        independent env instances. This is the property the spike needs to
        confirm before trusting any determinism claim about the simulator."""
        events = []
        for _ in range(2):
            env = _make_env(push_onset_step_range=(5, 15))
            try:
                env.reset(seed=7)
                action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
                for _ in range(15):
                    env.step(action)
                events.append(env.unwrapped.last_intervention_event)
            finally:
                env.close()
        assert events[0] is not None and events[1] is not None
        assert events[0].onset_step == events[1].onset_step
        assert events[0].severity == events[1].severity
        assert np.allclose(events[0].force, events[1].force)

    def test_zero_force_range_never_perturbs(self):
        env = _make_env(push_force_range=(0.0, 0.0))
        try:
            env.reset(seed=3)
            action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            for _ in range(15):
                env.step(action)
            event = env.unwrapped.last_intervention_event
            if event is not None:
                assert np.linalg.norm(event.force) == pytest.approx(0.0, abs=1e-6)
        finally:
            env.close()
