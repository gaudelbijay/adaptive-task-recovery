import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_integrated_state_continuation_is_mirrored_curriculum_with_matched_endpoints():
    scratch = load("learned_recovery_ppo_v12_integrated_mixture.json")
    continuation = load("learned_recovery_ppo_v13_integrated_from_strict.json")
    left = dict(scratch["experiments"][0])
    right = dict(continuation["experiments"][0])
    initializer = right.pop("init_checkpoint")
    continuation_environment = dict(right["env_kwargs"])
    scratch_environment = dict(left["env_kwargs"])
    assert continuation_environment.pop("intervention_probability") == 0.2
    assert scratch_environment.pop("intervention_probability") == 0.8
    assert continuation_environment == scratch_environment
    right["env_kwargs"] = scratch_environment
    left["env_kwargs"] = scratch_environment
    left.pop("method")
    right.pop("method")
    assert right == left
    assert initializer.endswith("seed_{seed}/best.pt")
    assert "learned_recovery_ppo_v11_strict_removal" in initializer
    assert continuation["seeds"] == scratch["seeds"] == [9351, 4796, 1788]
    assert continuation["learning_rate"] == 0.0001
    assert continuation["experiments"][0]["eval_env_kwargs"] == (
        scratch["experiments"][0]["eval_env_kwargs"]
    )
    for key in scratch:
        if key not in {"name", "experiments", "learning_rate", "claim_boundary"}:
            assert continuation[key] == scratch[key]


def test_state_continuation_records_immutable_initializer_and_fresh_optimizer():
    source = (
        ROOT / "scripts/train_manipulation_continuation_ppo.py"
    ).read_text(encoding="utf-8")
    assert 'elif task.get("init_checkpoint"):' in source
    assert 'agent.load_state_dict(initialization["agent"], strict=True)' in source
    assert '"checkpoint_sha256": _file_sha256(initialization_path)' in source
    assert '"optimizer_reinitialized": True' in source
    assert 'source_task.get(key) != task.get(key)' in source


def test_state_continuation_smoke_and_24_hour_wrapper_are_fail_safe():
    smoke = load("learned_recovery_ppo_v13_integrated_from_strict_smoke.json")
    task = smoke["experiments"][0]
    batch = task["num_envs"] * task["num_steps"]
    assert task["total_timesteps"] == 10 * batch
    assert smoke["seeds"] == [9351]
    assert "never eligible" in smoke["claim_boundary"]
    wrapper = (
        ROOT / "scripts/slurm_learned_recovery_continuation_ppo.sh"
    ).read_text(encoding="utf-8")
    assert "train_manipulation_continuation_ppo.py" in wrapper
    assert "#SBATCH --signal=USR1@300" in wrapper
    assert "#SBATCH --signal=B:" not in wrapper
    assert "#SBATCH --requeue" in wrapper
    assert "scontrol requeue" in wrapper
    assert "buffer overflow detected" in wrapper


def test_failure_only_state_teacher_uses_same_frozen_gate_thresholds():
    scratch_gate = load("integrated_state_teacher_gate_v1.json")
    continuation_gate = load("integrated_from_strict_state_teacher_gate_v2.json")
    assert continuation_gate["thresholds"] == scratch_gate["thresholds"]
    strict = load("strict_removal_integrated_from_strict_state_gate_v2.json")
    assert len(strict["cohorts"]) == 1
    assert strict["cohorts"][0]["config"].endswith(
        "learned_recovery_ppo_v13_integrated_from_strict.json"
    )
    assert "failure-only" in continuation_gate["claim_boundary"]
