#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from v47_renderer_expert_agent import RendererExpertV41Agent

if __name__ == "__main__":
    install_protocol_adapter("renderer_expert_adapter_v19")
    base.VisualAgent = RendererExpertV41Agent
    base.main()
