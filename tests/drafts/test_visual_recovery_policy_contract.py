"""Auditable restrictions for the deployed visual recovery policy."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "scripts" / "train_visual_recovery_ppo.py"
VISUAL_SLURM = ROOT / "scripts" / "slurm_visual_recovery_ppo.sh"
STATE_SLURM = ROOT / "scripts" / "slurm_learned_recovery_ppo.sh"
VISUAL_CAPTURE = ROOT / "scripts" / "capture_visual_recovery_policy.py"
HYPOTHESIS_WRAPPER = ROOT / "scripts/slurm_validate_visual_recovery_hypotheses.sh"
LEARNING_PLOT_WRAPPER = ROOT / "scripts/slurm_plot_visual_recovery_learning.sh"
STRICT_EVALUATOR = ROOT / "scripts/evaluate_visual_recovery_strict_removal.py"
RECOVERY_MONTAGE = ROOT / "scripts/build_recovery_montage.py"


def _module():
    return ast.parse(TRAINER.read_text(encoding="utf-8"))


def _assignment_literal(name):
    for node in _module().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"assignment not found: {name}")


def test_base_actor_extra_contract_contains_only_instruction():
    assert _assignment_literal("ACTOR_EXTRA_KEYS") == ("instruction",)


def test_deployed_action_api_cannot_receive_privileged_critic_state():
    for node in ast.walk(_module()):
        if isinstance(node, ast.ClassDef) and node.name == "VisualAgent":
            method = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "get_action")
            assert [argument.arg for argument in method.args.args] == [
                "self", "rgb", "proprio", "deterministic",
            ]
            return
    raise AssertionError("VisualAgent.get_action not found")


def test_actor_proprio_contains_only_robot_state_and_instruction():
    function = next(
        node for node in _module().body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_observation"
    )
    source = ast.unparse(function)
    assert "obs['agent']['qpos']" in source
    assert "obs['agent']['qvel']" in source
    assert "ACTOR_EXTRA_KEYS" in source
    actor_section = source.split("if asymmetric:", 1)[0]
    assert "tcp_pose" in actor_section
    assert "if actor_goal_progress:" in actor_section
    for forbidden in ("critic_red", "critic_blue", "critic_goal_resolved", "oracle"):
        assert forbidden not in actor_section


def test_policy_actions_are_tanh_bounded_and_never_posthoc_clipped():
    source = TRAINER.read_text(encoding="utf-8")
    assert "action = torch.tanh(pre_tanh_action)" in source
    rollout = source.split("for step in range(t):", 1)[1].split("with torch.no_grad():", 1)[0]
    assert "torch.clamp(action" not in rollout


def test_checkpoint_declares_restricted_observation_contract():
    source = TRAINER.read_text(encoding="utf-8")
    assert '"rgb_qpos_qvel_instruction_v1"' in source
    assert '"rgb_qpos_qvel_tcp_instruction_v2"' in source
    assert '"rgb_robot_proprio_instruction_progress_v3"' in source
    assert '"rgb_robot_proprio_instruction_visual_progress_v4"' in source


def test_learned_progress_is_inferred_from_rgb_not_appended_from_environment():
    source = TRAINER.read_text(encoding="utf-8")
    assert "self.goal_progress_predictor(latent)" in source
    assert 'task.get("actor_learned_goal_progress", False)' in source
    function = next(
        node for node in _module().body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_observation"
    )
    extractor = ast.unparse(function)
    assert "actor_learned_goal_progress" not in extractor
    assert "if actor_goal_progress:" in extractor
    assert 'obs["extra"]["critic_goal_resolved"]' in source


def test_terminal_success_does_not_bootstrap_value():
    source = TRAINER.read_text(encoding="utf-8")
    assert "bootstrap_mask = mask & truncated" in source
    assert "final_values[step, bootstrap_mask]" in source


def test_temporal_ssl_uses_executed_transition_and_masks_resets():
    source = TRAINER.read_text(encoding="utf-8")
    assert "next_rgbs[step] = next_rgb" in source
    assert "b_nonterminal = (1.0 - next_dones).reshape(-1)" in source
    assert "target = F.normalize(agent.encode(b_next_rgb[mb], augment=True)" in source
    assert "agent.temporal_predictor(torch.cat((latent, action), dim=1))" in source
    assert "weights = b_nonterminal[mb]" in source
    temporal = source.split("if temporal_coefficient:", 1)[1].split(
        "privileged_aux_loss", 1
    )[0]
    assert "with torch.no_grad():" in temporal


def test_24_hour_continuation_preserves_downstream_dependencies():
    wrappers = (
        VISUAL_SLURM,
        STATE_SLURM,
        ROOT / "scripts/slurm_manipulation_ppo.sh",
        ROOT / "scripts/slurm_learned_recovery_continuation_ppo.sh",
        ROOT / "scripts/slurm_visual_recovery_vicreg_ppo.sh",
        ROOT / "scripts/slurm_visual_recovery_dual_teacher_ppo.sh",
        ROOT / "scripts/slurm_visual_recovery_dual_teacher_vicreg_ppo.sh",
        ROOT / "scripts/slurm_vision_nedreamer.sh",
    )
    for wrapper in wrappers:
        source = wrapper.read_text(encoding="utf-8")
        # Slurm's B: prefix signals only the batch shell. The Python process
        # owns the atomic-save handler, so the job-step form is required.
        assert "#SBATCH --signal=USR1@300" in source
        assert "#SBATCH --signal=B:" not in source
        assert "#SBATCH --requeue" in source
        assert "scontrol requeue" in source
        assert "sbatch --array" not in source

    tabular = (ROOT / "scripts/slurm_rl_training.sh").read_text(encoding="utf-8")
    assert "#SBATCH --signal=USR1@180" in tabular
    assert "#SBATCH --signal=B:" not in tabular
    assert "#SBATCH --requeue" in tabular
    assert "scontrol requeue" in tabular


def test_state_strict_wrapper_accepts_recovery_config_alias():
    wrapper = (
        ROOT / "scripts/slurm_state_strict_removal_eval.sh"
    ).read_text(encoding="utf-8")
    assert 'ATR_STATE_CONFIG="${ATR_STATE_CONFIG:-${ATR_RECOVERY_CONFIG:-}}"' in wrapper
    assert 'set ATR_STATE_CONFIG or ATR_RECOVERY_CONFIG' in wrapper



def test_visual_capture_replays_restricted_actor_without_teleport():
    source = VISUAL_CAPTURE.read_text(encoding="utf-8")
    assert "extract_observation(" in source
    assert "agent.get_action(rgb, proprio, deterministic=True)" in source
    assert '"teleport_calls": 0' in source
    assert 'actual_removal_once |= _scalar(info, "goals_unavailable")' in source
    assert 'args.branch == "nominal" or actual_removal_once' in source
    assert '"actual_goal_unavailable": actual_removal_once' in source
    assert "set_pose" not in source
    assert "reconstruct_state_teacher_observation" not in source
    assert "event_reward_intervention_target_only_v3" in source
    assert "check_visualization_gate" in source
    assert '"visualization_gate": visualization_gate' in source
    assert '"predeclared integrated visual-policy selection"' in source
    assert 'not all(candidate["checks"].values())' in source
    assert 'candidate.get("method") != method' in source
    assert '"selection_sha256"' in source


def test_strict_removal_evaluator_is_separate_and_fails_closed():
    strict = json.loads((
        ROOT / "configs/visual_recovery_strict_removal_eval_v1.json"
    ).read_text(encoding="utf-8"))
    assert strict["intervention_overrides"]["onset_step_range"] == [0, 0]
    assert strict["require_actual_goal_unavailable_every_episode"] is True
    source = STRICT_EVALUATOR.read_text(encoding="utf-8")
    assert "import evaluate_visual_recovery_ppo as base" in source
    assert 'heldout_eval_strict_intervention.json' in source
    assert 'record["goals_unavailable"] >= 0.5' in source
    environment = (
        ROOT / "src/atr/envs/learned_recovery.py"
    ).read_text(encoding="utf-8")
    assert (
        "return physical & target & valid_target[:, None] "
        "& intervention_started[:, None]"
    ) in environment
    assert 'payload["condition"] = "strict_intervention"' in source
    state_source = (
        ROOT / "scripts/evaluate_state_recovery_strict_removal.py"
    ).read_text(encoding="utf-8")
    assert "import evaluate_manipulation_ppo as base" in state_source
    assert 'heldout_eval_strict_intervention.json' in state_source
    assert 'record["goals_unavailable"] >= 0.5' in state_source
    assert 'payload["condition"] = "strict_intervention"' in state_source


def test_zero_weight_auxiliary_head_can_preserve_initializer_architecture():
    for name in (
        "visual_recovery_strict_adaptive_v13_stable.json",
        "visual_recovery_strict_teacher_dagger_v14.json",
        "visual_recovery_integrated_teacher_dagger_v15.json",
    ):
        config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        task = config["experiments"][0]
        assert task["privileged_aux_coefficient"] == 0.0
        assert task["privileged_auxiliary_head"] is True
    source = TRAINER.read_text(encoding="utf-8")
    assert 'task.get("privileged_auxiliary_head", False)' in source


def test_strict_teacher_dagger_extension_is_gated_and_matched_by_seed():
    config = json.loads((
        ROOT / "configs/visual_recovery_strict_teacher_dagger_v14.json"
    ).read_text(encoding="utf-8"))
    task = config["experiments"][0]
    assert task["bc_teacher_checkpoint"].endswith("seed_{seed}/best.pt")
    assert task["bc_pretrain_updates"] * task["num_envs"] == 1_920_000
    assert task["bc_student_rollout_max"] == 0.8
    assert task["env_kwargs"]["intervention_probability"] == 0.8
    assert task["eval_env_kwargs"]["intervention_probability"] == 0.5
    assert config["num_eval_envs"] == 256
    gate = (ROOT / "scripts/slurm_check_state_teacher_nominal_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "--minimum-raw 0.70" in gate
    assert "--minimum-safe 0.70" in gate
    assert "--maximum-violation 0.05" in gate


def test_integrated_teacher_student_is_matched_and_preserves_rgb_contract():
    config = json.loads((
        ROOT / "configs/visual_recovery_integrated_teacher_dagger_v15.json"
    ).read_text(encoding="utf-8"))
    task = config["experiments"][0]
    v13 = json.loads((
        ROOT / "configs/visual_recovery_strict_adaptive_v13_stable.json"
    ).read_text(encoding="utf-8"))["experiments"][0]
    assert config["seeds"] == [9351, 4796, 1788]
    assert task["init_checkpoint"].endswith("seed_{seed}/best.pt")
    assert "visual_recovery_strict_adaptive_v13_stable" in task["init_checkpoint"]
    assert task["bc_teacher_checkpoint"].endswith("seed_{seed}/best.pt")
    assert "learned_recovery_ppo_v12_integrated_mixture" in task[
        "bc_teacher_checkpoint"
    ]
    assert task["bc_pretrain_updates"] * task["num_envs"] == 1_920_000
    assert task["bc_student_rollout_max"] == 0.8
    for field in (
        "env_kwargs", "eval_env_kwargs", "control_mode", "image_size",
        "actor_tcp_pose", "actor_learned_goal_progress", "asymmetric_critic",
    ):
        assert task[field] == v13[field]
    assert task.get("actor_goal_progress", False) is False
    assert "critic_" not in " ".join(task.keys())
    gate = json.loads((
        ROOT / "configs/integrated_state_teacher_gate_v1.json"
    ).read_text(encoding="utf-8"))
    assert gate["thresholds"] == json.loads((
        ROOT / "configs/integrated_visual_selection_v3.json"
    ).read_text(encoding="utf-8"))["thresholds"]


def test_recovery_montage_labels_require_strict_capture_opt_in():
    source = RECOVERY_MONTAGE.read_text(encoding="utf-8")
    assert '"--strict-removal-labels"' in source
    assert 'if args.strict_removal_labels' in source
    assert 'else "Sweeper targets first goal"' in source


def test_evaluation_and_aggregation_version_reward_semantics():
    evaluator = (ROOT / "scripts/evaluate_visual_recovery_ppo.py").read_text(
        encoding="utf-8"
    )
    state_evaluator = (ROOT / "scripts/evaluate_manipulation_ppo.py").read_text(
        encoding="utf-8"
    )
    aggregate = (ROOT / "scripts/aggregate_visual_recovery.py").read_text(
        encoding="utf-8"
    )
    comparison = (ROOT / "scripts/compare_visual_recovery_candidates.py").read_text(
        encoding="utf-8"
    )
    probe = (ROOT / "scripts/probe_visual_representation.py").read_text(
        encoding="utf-8"
    )
    marker = "event_reward_intervention_target_only_v3"
    assert marker in evaluator
    assert marker in state_evaluator
    assert marker in aggregate
    assert marker in probe
    assert "held-out records do not match configured benchmark semantics" in aggregate
    assert "state reference does not match configured benchmark semantics" in comparison


def test_final_hypothesis_job_is_strict_and_never_allows_missing_results():
    source = HYPOTHESIS_WRAPPER.read_text(encoding="utf-8")
    assert "validate_visual_recovery_hypotheses.py" in source
    assert "--allow-missing" not in source


def test_learning_plot_is_explicitly_training_only_and_covers_all_clean_cohorts():
    source = LEARNING_PLOT_WRAPPER.read_text(encoding="utf-8")
    for config in (
        "visual_recovery_ppo_gate_v2_event_reward.json",
        "visual_recovery_dagger_ablation_v7_event_reward.json",
        "visual_recovery_progress_dagger_v6_event_reward.json",
    ):
        assert config in source
    plotter = (ROOT / "scripts/plot_visual_recovery_learning.py").read_text(
        encoding="utf-8"
    )
    assert "training-stream diagnostics (not held-out)" in plotter


def test_v3_baselines_are_matched_factorial_ablations():
    gate = json.loads((
        ROOT / "configs/visual_recovery_ppo_gate_v2_event_reward.json"
    ).read_text(encoding="utf-8"))
    assert len(gate["experiments"]) == 3
    ignored = {"method", "asymmetric_critic", "temporal_ssl_coefficient"}
    templates = [
        {key: value for key, value in experiment.items() if key not in ignored}
        for experiment in gate["experiments"]
    ]
    assert templates[0] == templates[1] == templates[2]
    assert [item["asymmetric_critic"] for item in gate["experiments"]] == [
        False, True, True
    ]
    assert [item["temporal_ssl_coefficient"] for item in gate["experiments"]] == [
        0.0, 0.0, 0.05
    ]

    main = json.loads((
        ROOT / "configs/visual_recovery_progress_dagger_v6_event_reward.json"
    ).read_text(encoding="utf-8"))["experiments"][0]
    dagger = json.loads((
        ROOT / "configs/visual_recovery_dagger_ablation_v7_event_reward.json"
    ).read_text(encoding="utf-8"))["experiments"]
    temporal = dagger[1]
    ignored = {"method", "actor_learned_goal_progress", "goal_progress_aux_coefficient"}
    assert {key: value for key, value in main.items() if key not in ignored} == {
        key: value for key, value in temporal.items() if key not in ignored
    }
    assert main["bc_teacher_checkpoint"] == temporal["bc_teacher_checkpoint"]
    assert dagger[0]["temporal_ssl_coefficient"] == 0.0
    assert temporal["temporal_ssl_coefficient"] == 0.05


def test_v3_adaptive_dagger_followups_preserve_ssl_factorial():
    no_ssl = json.loads((
        ROOT / "configs/visual_recovery_dagger_intervention_v8_event_reward.json"
    ).read_text(encoding="utf-8"))["experiments"][0]
    temporal = json.loads((
        ROOT / "configs/visual_recovery_temporal_dagger_intervention_v9_event_reward.json"
    ).read_text(encoding="utf-8"))["experiments"][0]
    ignored = {"method", "init_checkpoint", "temporal_ssl_coefficient"}
    assert {key: value for key, value in no_ssl.items() if key not in ignored} == {
        key: value for key, value in temporal.items() if key not in ignored
    }
    assert no_ssl["temporal_ssl_coefficient"] == 0.0
    assert temporal["temporal_ssl_coefficient"] == 0.05
    assert "event_reward_pose_aux_dagger_visual_ppo" in no_ssl["init_checkpoint"]
    assert (
        "event_reward_pose_aux_temporal_dagger_visual_ppo"
        in temporal["init_checkpoint"]
    )


def test_confirmatory_configs_change_only_seeds_and_initializer_root():
    def config(name):
        return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))

    clean_screen = config("visual_recovery_progress_dagger_v6_event_reward.json")
    clean_confirm = config(
        "visual_recovery_progress_dagger_confirm_v10_event_reward.json"
    )
    assert clean_confirm["seeds"] == [71064, 84293]
    assert clean_screen["experiments"] == clean_confirm["experiments"]

    adaptive_screen = config(
        "visual_recovery_progress_intervention_v7_event_reward.json"
    )["experiments"][0]
    adaptive_confirm = config(
        "visual_recovery_progress_intervention_confirm_v11_event_reward.json"
    )["experiments"][0]
    ignored = {"init_checkpoint"}
    assert {k: v for k, v in adaptive_screen.items() if k not in ignored} == {
        k: v for k, v in adaptive_confirm.items() if k not in ignored
    }
    assert "confirm_v10" in adaptive_confirm["init_checkpoint"]

    state_screen = config("learned_recovery_ppo_v8_event_reward.json")
    state_confirm = config("learned_recovery_ppo_v10_event_reward_confirm.json")
    assert state_confirm["seeds"] == [71064, 84293]
    assert state_screen["experiments"] == state_confirm["experiments"]
    for key in state_screen:
        if key not in {"name", "seeds"}:
            assert state_screen[key] == state_confirm[key]
