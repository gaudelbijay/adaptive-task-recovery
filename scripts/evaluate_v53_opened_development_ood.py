#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v44_visual_recovery import install_protocol_adapter
from evaluate_v52_visual_recovery_unseen_ood import install_extensions
from v53_opened_renderer_agent import OpenedRendererExpertAgent
if __name__=="__main__": install_extensions();install_protocol_adapter("opened_renderer_experts_v19");base.VisualAgent=OpenedRendererExpertAgent;base.main()
