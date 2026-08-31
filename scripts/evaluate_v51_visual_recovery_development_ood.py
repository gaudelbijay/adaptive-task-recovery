#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v41_visual_recovery_unseen_ood import install_extensions
from evaluate_v44_visual_recovery import install_protocol_adapter
from v51_hierarchical_renderer_agent import HierarchicalRendererExpertAgent
if __name__ == "__main__":
    install_extensions(); install_protocol_adapter("hierarchical_renderer_experts_v19")
    base.VisualAgent = HierarchicalRendererExpertAgent
    base.main()
