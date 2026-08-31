#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v41_visual_recovery_unseen_ood import install_extensions
from evaluate_v44_visual_recovery import install_protocol_adapter
from v52_subpixel_specialist_agent import SubpixelSpecialistAgent
if __name__ == "__main__":
    install_extensions(); install_protocol_adapter("subpixel_specialist_router_v19")
    base.VisualAgent = SubpixelSpecialistAgent
    base.main()
