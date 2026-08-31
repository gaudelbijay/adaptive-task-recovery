#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from v48_geometry_expert_agent import HierarchicalGeometryExpertAgent

if __name__ == "__main__":
    install_protocol_adapter("onpolicy_geometry_expert_v19")
    base.VisualAgent = HierarchicalGeometryExpertAgent
    base.main()
