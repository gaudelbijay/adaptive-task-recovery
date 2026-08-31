#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v41_visual_recovery_unseen_ood import install_extensions
from evaluate_v46_visual_recovery import configure_threshold
from evaluate_v44_visual_recovery import install_protocol_adapter
from v44_multiview_feature_agent import CalibratedHybridFeatureV41Agent


if __name__ == "__main__":
    install_extensions()
    configure_threshold()
    install_protocol_adapter("hybrid_calibrated_feature_adapter_v19")
    base.VisualAgent = CalibratedHybridFeatureV41Agent
    base.main()
