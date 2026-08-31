#!/usr/bin/env python3
from run_v28_unseen_visual_ood import main

if __name__ == "__main__":
    main(
        evaluator_script="evaluate_v46_visual_recovery_development_ood.py",
        execution_protocol="V46 development-domain execution",
        execution_prefix="v46_development_ood_execution",
    )
