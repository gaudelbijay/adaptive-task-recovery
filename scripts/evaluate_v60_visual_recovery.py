#!/usr/bin/env python3
"""Evaluate the independent V60 lineages on the standard protocol."""

import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from v60_joint_residual_agent import JointResidualCompositionAgent


if __name__ == "__main__":
    install_protocol_adapter(
        "continuous_geometry_composition_v19", "geometry_training_transitions",
    )
    base.VisualAgent = JointResidualCompositionAgent
    base.main()
