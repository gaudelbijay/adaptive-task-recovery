#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from evaluate_v52_visual_recovery_unseen_ood import install_extensions
from v54_continuous_geometry_agent import ContinuousGeometryCompositionAgent

if __name__ == "__main__":
    install_extensions(); install_protocol_adapter("continuous_geometry_composition_v19", "geometry_training_transitions")
    base.VisualAgent = ContinuousGeometryCompositionAgent; base.main()
