"""One-episode ATR pipeline whose executor never teleports an object.

The policy parses the Fetch instruction, observes an irreversible cracker-box
removal through a fixed RGB camera, uses a reward-trained Q table for the
attempt/skip decision, validates attempted actions against the intent guard,
and executes accepted goals with contact-verified navigation, grasp, carry,
and release.  Privileged state is used only for final evaluation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from atr.constraints.intent_guard import validate_action
from atr.envs.tidy_up_replicacad_manipulation import attempt_goal_with_real_grasp
from atr.envs.tidy_up_replicacad_policies import _TRAY_SLOTS
from atr.feasibility.frame_diff import frame_difference_score
from atr.feasibility.oracle import evaluate_goal_graph
from atr.language.instruction_parser import parse_instruction
from atr.policies.baselines import _summarize
from atr.policies.q_learning import ATTEMPT, greedy_action

FETCH_OBJECTS = {"potted_meat_can", "bowl", "master_chef_can", "cracker_box"}
# Measured from the fixed Fetch intervention camera.  The removed cracker's
# changed pixels occupy this region; keeping the crop explicit makes the
# visual assumption inspectable and testable.
RECOVERY_CHANGE_CROP = (128, 384, 128, 384)

PHYSICAL_INSTRUCTION = (
    "Put the potted meat can and the cracker box on the table, and do not "
    "move the master chef can."
)


def settle_before_task(env, steps: int = 3) -> None:
    """Let spawned ReplicaCAD objects settle before the safety snapshot."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        env.step(action)


def instruction_graph():
    graph = parse_instruction(PHYSICAL_INSTRUCTION, FETCH_OBJECTS)
    # Match the established Fetch task's measured tolerance.  The generic
    # parser default is 2 cm, while this ReplicaCAD actor naturally settles
    # about 3.5 cm before any robot contact; the original Fetch graph uses
    # 5 cm for this exact master-chef-can constraint.
    constraints = tuple(
        replace(constraint, tolerance=0.05)
        if constraint.kind == "never_move" else constraint
        for constraint in graph.constraints
    )
    return replace(graph, constraints=constraints)


def recovery_change_score(reference_frame: np.ndarray, current_frame: np.ndarray) -> float:
    y0, y1, x0, x1 = RECOVERY_CHANGE_CROP
    return frame_difference_score(
        reference_frame[y0:y1, x0:x1], current_frame[y0:y1, x0:x1],
    )


def run_nonteleport_episode(
    env,
    q_table: dict,
    *,
    recovery_change_threshold: float,
    policy: str = "visual_learned_guarded",
) -> dict:
    """Run the complete hierarchy in one live Fetch episode.

    ``policy`` is one of ``static``, ``oracle``, or
    ``visual_learned_guarded``.  The live learned policy never reads
    ``_exists`` or ``_world_state`` to choose an action.  The final oracle
    read below is evaluation only.
    """
    if policy not in {"static", "oracle", "visual_learned_guarded"}:
        raise ValueError(f"unknown policy: {policy}")

    graph = instruction_graph()
    # The navigation-level intent guard reads the environment's graph while
    # screening the real route, so give it the same parsed contract used by
    # the high-level policy.
    env.unwrapped.goal_graph = graph
    initial_state = env.unwrapped._world_state()  # evaluation snapshot only
    reference_frame = env.render()[0].cpu().numpy()
    per_goal: dict[str, dict] = {}

    for i, goal in enumerate(graph.goals):
        change_score = None
        if goal.target_object == "cracker_box":
            current_frame = env.render()[0].cpu().numpy()
            change_score = recovery_change_score(reference_frame, current_frame)
            perceived_feasible = change_score <= recovery_change_threshold
        else:
            # The intervention family changes only the cracker box.  This is an
            # explicit task prior, not a privileged live-state query.
            perceived_feasible = True

        if policy == "static":
            action = ATTEMPT
            decision_source = "static"
        elif policy == "oracle":
            # Oracle is an evaluation-only headroom baseline.
            action = ATTEMPT if env.unwrapped._exists[goal.target_object] else 0
            decision_source = "privileged_oracle"
        else:
            action = greedy_action(q_table, (goal.id, perceived_feasible))
            decision_source = "rgb_change_plus_q"

        common = {
            "perceived_feasible": perceived_feasible,
            "visual_change_score": change_score,
            "decision_source": decision_source,
            "policy_action": "attempt" if action == ATTEMPT else "skip",
        }
        if action != ATTEMPT:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True, **common,
            }
            continue

        allowed, reason = validate_action(goal.target_object, graph)
        if not allowed:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "blocked_reason": reason, **common,
            }
            continue

        outcome = attempt_goal_with_real_grasp(env, goal, _TRAY_SLOTS[i])
        per_goal[goal.id] = {**outcome, "skipped": False, "guard_reason": reason, **common}

    final_state = env.unwrapped._world_state()  # evaluation only
    oracle = evaluate_goal_graph(graph, initial_state, final_state)
    result = _summarize(per_goal)
    result.update({
        "pipeline": "parsed_language+rgb_change+learned_q+intent_guard+physical_fetch",
        "teleport_calls": 0,
        "intervention_triggered": bool(env.unwrapped._triggered),
        "constraint_violations": oracle["constraint_violations"],
        "per_goal": per_goal,
    })
    return result
