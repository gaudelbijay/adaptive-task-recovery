"""D-104: manipulation requires verified navigation arrival."""

from types import SimpleNamespace

import numpy as np

from atr.envs import tidy_up_replicacad_policies as policies
from atr.envs.navigation import NavigationOutcome
from atr.language.goal_graph import Goal


def test_existing_object_is_not_teleported_when_navigation_did_not_arrive(monkeypatch):
    actor = SimpleNamespace(
        pose=SimpleNamespace(sp=SimpleNamespace(p=np.array([1.0, 0.0, 0.5]))),
        set_pose=lambda _pose: (_ for _ in ()).throw(AssertionError("teleported")),
    )
    unwrapped = SimpleNamespace(
        _exists={"potted_meat_can": True},
        _elapsed_control_steps=12,
        _get_actor=lambda _object_id: actor,
    )
    env = SimpleNamespace(unwrapped=unwrapped)
    monkeypatch.setattr(
        policies,
        "_navigate_to",
        lambda *_args, **_kwargs: NavigationOutcome(
            steps_used=250,
            reached_target=False,
        ),
    )

    result = policies.attempt_goal(
        env,
        Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can"),
        np.zeros(3),
    )

    assert result["achieved"] is False
    assert result["navigation_failed"] is True
    assert result["navigation_reached_target"] is False
