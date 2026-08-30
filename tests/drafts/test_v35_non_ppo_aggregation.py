import pytest

from aggregate_visual_recovery import transition_accounting_view


def record():
    return {
        "ppo_accounting_applicable": False,
        "training_protocol": "supervised_translation_repair_v34",
        "translation_training_transitions": 128000,
        "checkpoint_global_step": 128000,
        "initialization_simulator_transitions": 1536000,
        "total_environment_transitions": 1664000,
        "protocol_environment_transitions_consumed": 1664000,
        "initialization_provenance": None,
        "online_ppo_environment_steps": 0,
        "initialization_ppo_environment_steps": 0,
        "ppo_environment_steps": 0,
        "online_protocol_ppo_environment_steps": 0,
        "initialization_protocol_ppo_environment_steps": 0,
        "protocol_ppo_environment_steps": 0,
        "local_bc_dagger_environment_transitions": 0,
        "initialization_bc_dagger_environment_transitions": 0,
        "bc_dagger_environment_transitions": 0,
    }


def experiment():
    return {"total_timesteps": 128000, "num_envs": 64, "num_steps": 1}


def test_v35_non_ppo_accounting_accepts_exact_complete_budget():
    source = record()
    view = transition_accounting_view(source, experiment())
    assert source["online_ppo_environment_steps"] == 0
    assert source["total_environment_transitions"] == 1664000
    assert view["online_ppo_environment_steps"] == 128000
    assert view["total_environment_transitions"] == 128000


@pytest.mark.parametrize(
    "field,value",
    [
        ("online_ppo_environment_steps", 1),
        ("checkpoint_global_step", 127999),
        ("total_environment_transitions", 1663999),
        ("protocol_environment_transitions_consumed", 1663999),
    ],
)
def test_v35_non_ppo_accounting_fails_closed(field, value):
    payload = record()
    payload[field] = value
    with pytest.raises(ValueError):
        transition_accounting_view(payload, experiment())
