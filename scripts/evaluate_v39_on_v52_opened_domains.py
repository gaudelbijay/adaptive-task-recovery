#!/usr/bin/env python3
import evaluate_visual_recovery_ppo as base
from evaluate_v39_visual_recovery import install_protocol_adapter
from evaluate_v52_visual_recovery_unseen_ood import install_extensions
if __name__ == "__main__": install_extensions(); install_protocol_adapter(); base.main()
