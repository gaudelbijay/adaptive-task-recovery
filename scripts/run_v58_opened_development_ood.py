#!/usr/bin/env python3
from run_v28_unseen_visual_ood import main
if __name__ == "__main__":
    main(evaluator_script="evaluate_v58_opened_development_ood.py",
         execution_protocol="V58 opened-domain development execution",
         execution_prefix="v58_opened_development_execution")
