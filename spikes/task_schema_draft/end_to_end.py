"""Stage 6 of docs/00-project-overview.md's build-up order: combine
everything -- language, vision, learned policy -- into one real pipeline
on one episode, instead of five separate demonstrations that never talk
to each other.

Concretely, for each goal in a parsed instruction:
  1. parse_instruction() (D-019/D-026) turns the instruction into a GoalGraph
  2. a real rendered frame + visual_object_exists() (D-020) judges whether
     the goal's target object still exists -- NOT a privileged-state read
  3. a Q-table trained by atr.policies.q_learning's train_q_table()
     (D-025/D-030, promoted D-041), retrained here for this env's goals,
     decides attempt vs. skip from that *perceived* feasibility
  4. attempt_goal() (unchanged, real arm motion) executes the decision

Privileged state still exists in this file -- but only in
`train_q_table_replicacad_humanoid()`, for *training* the Q-table, and in
tests, for checking the pipeline's decisions against ground truth. The live
`run_end_to_end_episode()` never reads it to make a decision. This split
is deliberate, not an oversight: training the decision *rule* ("attempt
iff feasible") doesn't need to happen against real pixels for this toy
case -- the rule itself is perception-independent -- so training stays
cheap (no rendering, ~2 min here since this env is heavier to construct
than the canonical one) while *evaluation* uses the real visual signal
throughout. Training against real rendered rollouts would need hundreds of
render-producing resets, which D-022's confirmed upstream rendering bug
makes impractical (safe budget: ~2 per process, or one subprocess each).

Exactly two render-producing calls per episode, matching D-022's verified-
safe budget: one frame per goal decision, and there are two goals.
"""

from __future__ import annotations

import gymnasium as gym

from atr.envs.tidy_up_env_replicacad_humanoid import replicacad_humanoid_example
from atr.envs.tidy_up_replicacad_humanoid_policies import (
    _TRAY_SLOTS,
    _summarize,
    attempt_goal,
)
from atr.feasibility.clip_feasibility import visual_object_exists
from atr.language.goal_graph import GoalGraph
from atr.language.instruction_parser import parse_instruction
from atr.policies.q_learning import ATTEMPT, SKIP, train_q_table

HUMANOID_OBJECTS = {"potted_meat_can", "master_chef_can", "cracker_box", "bowl"}


def _instruction_graph() -> GoalGraph:
    """The real thing this stage is testing: a GoalGraph produced by
    *parsing* the instruction, not the hand-authored replicacad_humanoid_example()
    directly -- goal ids come from the parser (D-019), so training and
    evaluation must both go through this, not the hand-authored version,
    or the Q-table's keys won't match at evaluation time."""
    return parse_instruction(replicacad_humanoid_example().instruction_text, HUMANOID_OBJECTS)


def _make_replicacad_humanoid_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    return gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode=None, sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def train_q_table_replicacad_humanoid(n_episodes: int = 30, seed: int = 0) -> dict:
    """atr.policies.q_learning's train_q_table() (D-025/D-030, promoted
    D-041), configured for this env's goals/ids -- privileged-state, no
    rendering. See module docstring
    for why training stays privileged while evaluation doesn't. Default
    n_episodes is lower than train_q_table_canonical()'s 120: this env is
    much heavier to construct per episode (a full ReplicaCAD apartment vs.
    a five-object tabletop), and the state space is just as trivial by
    construction, so fewer episodes still converge."""
    return train_q_table(
        make_env=_make_replicacad_humanoid_env, graph=_instruction_graph(), tray_slots=_TRAY_SLOTS,
        attempt_goal_fn=attempt_goal, intervention_kinds=("none", "chef_can_destroyed"),
        onset_step_bounds=(1, 3), n_episodes=n_episodes, seed=seed,
    )


def run_end_to_end_episode(env, q_table: dict, scene_variant: str = "kitchen_cabinet") -> dict:
    """The real integration: parse the instruction, then for each goal,
    render a frame, judge feasibility from it (not privileged state), and
    let the trained Q-table decide whether to attempt. Exactly one render
    per goal -- two goals, two renders total, D-022's verified-safe budget."""
    graph = _instruction_graph()
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        frame = env.render()[0].cpu().numpy()
        perceived_feasible = visual_object_exists(frame, goal.target_object, scene_variant)
        key = (goal.id, perceived_feasible)
        actions = q_table.get(key, {SKIP: 0.0, ATTEMPT: 0.0})
        action = max(actions, key=actions.get)

        if action == SKIP:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "perceived_feasible": perceived_feasible,
            }
        else:
            result = attempt_goal(env, goal, _TRAY_SLOTS[i])
            result["perceived_feasible"] = perceived_feasible
            per_goal[goal.id] = result

    return _summarize(per_goal)  # already includes "per_goal": per_goal
