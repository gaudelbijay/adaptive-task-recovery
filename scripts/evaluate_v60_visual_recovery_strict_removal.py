#!/usr/bin/env python3
"""Evaluate the independent V60 lineages under verified step-0 removal."""

import evaluate_visual_recovery_ppo as base
import evaluate_visual_recovery_strict_removal as strict
from evaluate_v44_visual_recovery import install_protocol_adapter
from v60_joint_residual_agent import JointResidualCompositionAgent


if __name__ == "__main__":
    install_protocol_adapter(
        "continuous_geometry_composition_v19", "geometry_training_transitions",
    )
    base.VisualAgent = JointResidualCompositionAgent
    strict.main()
