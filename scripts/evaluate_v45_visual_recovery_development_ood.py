#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v41_visual_recovery_unseen_ood import install_extensions
from v44_multiview_feature_agent import AlwaysMultiViewFeatureV41Agent
from evaluate_v44_visual_recovery import install_protocol_adapter
if __name__=="__main__": install_extensions(); install_protocol_adapter(); base.VisualAgent=AlwaysMultiViewFeatureV41Agent; base.main()
