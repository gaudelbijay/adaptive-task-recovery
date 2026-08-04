"""Humanoid skill adapters and whole-body kinematics/safety tooling.

`ik_solver.py` (D-028, promoted D-051) is a real analytic-Jacobian IK
solver for G1's right arm -- built to retry D-024's grasp-confirmation
attempt with a principled tool, and ended up confirming a genuine
kinematic reachability limit (not a solver artifact) via a deterministic,
verified-against-ManiSkill's-own-kinematics search. Zero project-internal
dependency (`numpy`, `pinocchio`, `mani_skill.PACKAGE_ASSET_DIR` only) --
reusable independent of any specific TidyUp env variant.
"""

from __future__ import annotations
