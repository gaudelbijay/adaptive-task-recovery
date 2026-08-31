#!/usr/bin/env python3
from run_v28_unseen_visual_ood import main

if __name__ == "__main__":
    main(evaluator_script="evaluate_v57_opened_development_ood.py",
         execution_protocol="V57 opened-domain development execution",
         execution_prefix="v57_opened_development_execution")
