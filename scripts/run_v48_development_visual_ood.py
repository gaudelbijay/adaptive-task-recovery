#!/usr/bin/env python3
from run_v28_unseen_visual_ood import main

if __name__ == "__main__":
    main(evaluator_script="evaluate_v48_visual_recovery_development_ood.py",
         execution_protocol="V48 development-domain execution",
         execution_prefix="v48_development_ood_execution")
